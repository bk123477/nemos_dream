"""Stage 1b — attach Korean cultural equivalents to each ``CulturalRef``.

Deterministic cascade: ``dict`` → ``retriever`` → ``web+LLM``. Runs the cheap
exact lookup first, then the semantic retriever over the same seed dictionary,
and finally grounds a Nemotron JSON call on Tavily search results when neither
upstream has a confident hit.

The ``dialogue`` argument, when provided, lets the web+llm path condition on
the surrounding utterances (e.g. "venti" → Starbucks size, not a Genshin NPC).
"""

from __future__ import annotations

import json
import os

from nemos_dream.nvidia_clients import NvidiaSyncClient
from nemos_dream.schemas import CulturalRef, MappedRef, Turn

from ._ref_filter import filter_refs
from ._validator import _check_rules
from .tools import dict_lookup, retriever_search, web_search

_RETRIEVER_THRESHOLD = 0.78
_RETRIEVER_MARGIN = 0.05

_TYPE_GUIDANCE = {
    "brand": "a Korean brand that serves the same role in Korean daily life",
    "service": "the Korean app/service most people use for the same purpose",
    "food": "a Korean food eaten in similar social contexts (not a transliteration)",
    "holiday": "a Korean holiday with similar social/cultural function",
    "event": "a Korean event that plays the same social role",
    "pop_culture": "a Korean equivalent in the same entertainment category",
    "slang": "the closest Korean slang used in the same register and situation",
    "other": "the closest Korean cultural analogue, not a literal translation",
}

_WEB_REASONING_PROMPT = (
    "You are localizing an English {type} to its closest KOREAN CULTURAL ANALOGUE.\n"
    'English term: "{term}"\n'
    'Used in this dialogue: "{context}"\n\n'
    "Goal: pick {guidance}.\n\n"
    "Rules:\n"
    "- SAME SEMANTIC CATEGORY. Map a school to a school, a vehicle to a vehicle,\n"
    "  a sport to a sport, a test to a test. Do not swap categories.\n"
    "- NOT a literal translation. The Korean term should be what a Korean speaker\n"
    "  would naturally mention in the same social situation.\n"
    "- NOT a fictional character, game item, or off-domain entity.\n"
    "- Prefer widely-recognized Korean names; if it's a loanword, use the accepted\n"
    "  Hangul form (e.g. 스타벅스, 카라멜 마키아토).\n"
    "- If no Korean analogue of the same category is plausible, return ko=\"\".\n\n"
    "Good examples (same role, same category):\n"
    "  thanksgiving (holiday) → 추석, venmo (service) → 토스,\n"
    "  superbowl (event) → 월드컵 결승, sat (event) → 수능\n\n"
    "Bad examples (do NOT do this):\n"
    "  college → 수능     ✗  college is an institution (→ 대학교), not an exam\n"
    "  clown → clown      ✗  never echo the English term — return ko=\"\"\n\n"
    "Web search results:\n{results}\n\n"
    "Respond with a single JSON object: "
    '{{"ko": "<Korean term or empty string>", "notes": "<one short sentence>"}}. '
    "JSON only, no prose."
)


class _MapperClient(NvidiaSyncClient):
    model_env = "NEMOTRON_MODEL"

    def call(self, prompt: str) -> str:
        model = self.model or "nvidia/nemotron-3-nano-30b-a3b"
        resp = self.openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"nvext": {"guided_json": {
                "type": "object",
                "properties": {"ko": {"type": "string"}, "notes": {"type": "string"}},
                "required": ["ko"],
            }}},
            temperature=0.2,
        )
        return resp.choices[0].message.content or "{}"


def _extract_context(term: str, dialogue: list[Turn] | None) -> str:
    """Return the 1-2 dialogue turns containing ``term`` (case-insensitive)."""
    if not dialogue:
        return ""
    needle = term.strip().lower()
    if not needle:
        return ""
    hits: list[str] = []
    for t in dialogue:
        if needle in (t.text or "").lower():
            hits.append(f"{t.speaker}: {t.text}")
            if len(hits) >= 2:
                break
    return " | ".join(hits)


def _web_then_llm(term: str, ref_type: str, context: str = "") -> MappedRef:
    results = web_search.search(
        term, max_results=3, ref_type=ref_type, context=context or None,
    )
    results_block = "\n".join(
        f"- {r.get('title','')}: {r.get('content','')[:300]}" for r in results
    ) or "(no web results)"

    guidance = _TYPE_GUIDANCE.get(ref_type, _TYPE_GUIDANCE["other"])
    prompt = _WEB_REASONING_PROMPT.format(
        term=term,
        type=ref_type,
        guidance=guidance,
        context=context or "(no surrounding dialogue available)",
        results=results_block,
    )

    base_url = os.environ.get("NVIDIA_API_BASE")
    client = _MapperClient(base_url=base_url) if base_url else _MapperClient()
    try:
        payload = json.loads(client.call(prompt))
    except json.JSONDecodeError:
        payload = {}

    ko = (payload.get("ko") or "").strip()
    notes = payload.get("notes", "")
    mapped = MappedRef(
        term=term,
        ko=ko or term,
        type=ref_type,
        source="web+llm",
        retrieved=True,
        notes=notes,
    )
    # Persist clean mappings to the seed dict so next run hits the dict path.
    # Stricter bar than in-flight output: any validation flag blocks the append.
    if ko and ko != term and not _check_rules(mapped):
        dict_lookup.append_entry(term, ko, ref_type=ref_type, notes=notes)
    return mapped


def _map_one(ref: CulturalRef, context: str = "") -> MappedRef:
    hit = dict_lookup.lookup(ref.term)
    if hit is not None:
        return MappedRef(
            term=ref.term,
            ko=hit["ko"],
            type=hit["type"],
            source="dict",
            retrieved=False,
            notes=hit.get("notes", ""),
        )

    try:
        # ``top_k=2`` lets us apply a margin test: a weak winner (top1 close to
        # top2) means the index doesn't really know the term, so we fall
        # through to web+llm instead of accepting a lukewarm match.
        retrieved = retriever_search.search(
            ref.term, top_k=2, threshold=_RETRIEVER_THRESHOLD,
        )
    except FileNotFoundError:
        retrieved = []
    if retrieved:
        top = retrieved[0]
        margin_ok = (
            len(retrieved) < 2
            or (top["score"] - retrieved[1]["score"]) >= _RETRIEVER_MARGIN
        )
        if margin_ok:
            return MappedRef(
                term=ref.term,
                ko=top["ko"],
                type=top["type"],
                source="retriever",
                retrieved=False,
                notes=f"cosine={top['score']:.3f}; {top.get('notes', '')}".strip("; "),
            )

    return _web_then_llm(ref.term, ref.type, context=context)


def map_refs(
    refs: list[CulturalRef],
    *,
    dialogue: list[Turn] | None = None,
) -> list[MappedRef]:
    """Return a Korean mapping for each reference; empty list if none fit.

    Passing ``dialogue`` lets the web+llm path disambiguate polysemous terms
    using the surrounding utterances.
    """
    refs = filter_refs(refs)
    return [_map_one(r, context=_extract_context(r.term, dialogue)) for r in refs]
