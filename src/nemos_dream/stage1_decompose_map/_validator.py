"""Post-mapping validation flags for cultural references.

Catches common failure modes on SNS seed data:

- ``ko_empty_or_same``: mapping fell through to echoing the English term.
- ``non_hangul_ko``: a brand/service/food/holiday mapping has no Hangul at all.
- ``corrupted_token``: the LLM emitted Python-ish tokens like ``식민지_rule``
  inside Korean prose (a tell-tale sign of broken generation).
- ``out_of_domain_leak``: a non-pop-culture reference was mapped with
  game/anime/fandom vocabulary (e.g. venti [Starbucks] → a Genshin character).
- ``llm_rejected``: optional second-pass Nemotron sanity check disagreed.

Rule checks are free and always run. The LLM check is opt-in and runs only on
references already flagged by the rule checks, so it is bounded work.

Flags are emitted as plain ``dict`` matching ``MappedRef.validation``'s schema.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from typing import Any

from nemos_dream.nvidia_clients import NvidiaSyncClient
from nemos_dream.schemas import MappedRef

_HANGUL_RE = re.compile(r"[가-힣]")
_UNDERSCORE_TOKEN_RE = re.compile(r"[A-Za-z가-힣]+_[A-Za-z가-힣]+")
_KO_REQUIRED_TYPES = {"brand", "service", "food", "holiday", "pop_culture", "event"}
# pop_culture entries often use romanized stage names (IU, BTS); keep them warn-only.
_KO_REQUIRED_WARN_TYPES = {"pop_culture"}

_OUT_OF_DOMAIN_KEYWORDS = {
    "brand": ("캐릭터", "게임", "애니메이션", "원신", "genshin", "캐릭", "만화"),
    "service": ("캐릭터", "게임", "애니메이션", "원신", "만화"),
    "food": ("캐릭터", "게임", "애니메이션", "원신", "만화"),
    "other": ("캐릭터", "게임", "원신"),
    "holiday": ("게임", "원신", "애니메이션"),
    "event": ("게임", "원신", "애니메이션"),
}

_LLM_VERIFY_PROMPT = (
    'Is "{ko}" a plausible Korean cultural equivalent or translation of the '
    'English term "{term}" (type: {type})?\n'
    "Context notes (may be empty): {notes}\n\n"
    "A plausible mapping either (a) shares the same referent in Korean, "
    "(b) is a widely-known Korean analogue, or (c) is the accepted loanword. "
    "Reject mappings that are unrelated characters, fictional entities, or "
    "random words.\n\n"
    'Respond with JSON only: {{"valid": true|false, '
    '"better_ko": "<alternative or empty string>", '
    '"reason": "<one short sentence>"}}'
)


def _flag(code: str, severity: str, message: str) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message}


def _has_hangul(s: str) -> bool:
    return bool(_HANGUL_RE.search(s or ""))


def _check_rules(ref: MappedRef) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    ko = ref.ko or ""
    notes = ref.notes or ""
    ref_type = ref.type or "other"

    if not ko.strip() or ko.strip().lower() == ref.term.strip().lower():
        flags.append(_flag(
            "ko_empty_or_same",
            "error",
            "ko is empty or identical to source term (mapping likely failed)",
        ))
        return flags

    if ref_type in _KO_REQUIRED_TYPES and not _has_hangul(ko):
        severity = "warn" if ref_type in _KO_REQUIRED_WARN_TYPES else "error"
        flags.append(_flag(
            "non_hangul_ko",
            severity,
            f'type "{ref_type}" expects a Korean mapping, but ko="{ko}" has no Hangul',
        ))

    corrupted = _UNDERSCORE_TOKEN_RE.findall(ko) + _UNDERSCORE_TOKEN_RE.findall(notes)
    if corrupted:
        flags.append(_flag(
            "corrupted_token",
            "warn",
            f"underscored tokens in output: {corrupted[:3]}",
        ))

    banned = _OUT_OF_DOMAIN_KEYWORDS.get(ref_type, ())
    if banned:
        haystack = f"{ko} {notes}".lower()
        hits = [kw for kw in banned if kw.lower() in haystack]
        if hits:
            flags.append(_flag(
                "out_of_domain_leak",
                "warn",
                f'type "{ref_type}" mapped with off-domain vocabulary: {hits}',
            ))

    return flags


class _VerifyClient(NvidiaSyncClient):
    model_env = "NEMOTRON_MODEL"

    def call(self, prompt: str) -> str:
        model = self.model or "nvidia/nemotron-3-nano-30b-a3b"
        resp = self.openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"nvext": {"guided_json": {
                "type": "object",
                "properties": {
                    "valid": {"type": "boolean"},
                    "better_ko": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["valid"],
            }}},
            temperature=0.0,
        )
        return resp.choices[0].message.content or "{}"


def _llm_verify(ref: MappedRef) -> dict[str, Any] | None:
    """Ask Nemotron whether ``ref.ko`` is plausible. Returns a flag on reject,
    None on accept or on any client error (fail-open)."""
    try:
        base_url = os.environ.get("NVIDIA_API_BASE")
        client = _VerifyClient(base_url=base_url) if base_url else _VerifyClient()
        raw = client.call(_LLM_VERIFY_PROMPT.format(
            term=ref.term, ko=ref.ko, type=ref.type, notes=ref.notes or "(none)",
        ))
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001 — fail-open; the rule checks still apply
        return None

    if payload.get("valid", True):
        return None

    reason = (payload.get("reason") or "").strip() or "llm judged mapping implausible"
    better = (payload.get("better_ko") or "").strip()
    msg = reason + (f' (suggest: "{better}")' if better else "")
    return _flag("llm_rejected", "error", msg)


def validate_ref(ref: MappedRef, *, use_llm: bool = False) -> MappedRef:
    """Return a copy of ``ref`` with ``.validation`` populated."""
    flags = _check_rules(ref)
    if use_llm and flags:
        llm_flag = _llm_verify(ref)
        if llm_flag is not None:
            flags.append(llm_flag)
    return ref.model_copy(update={"validation": flags})


def validate_refs(refs: Iterable[MappedRef], *, use_llm: bool = False) -> list[MappedRef]:
    return [validate_ref(r, use_llm=use_llm) for r in refs]
