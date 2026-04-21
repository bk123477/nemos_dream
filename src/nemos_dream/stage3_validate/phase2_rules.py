"""Phase 2 — rule-based hard gates on a stage-2 row.

Two layers:

1. **Text-level rules** (``ASCIIRatioFilter``, ``MappedRefSurfaceFilter``,
   ``MappedRefKoHangulFilter``) subclass Curator's
   ``nemo_curator.stages.text.filters.doc_filter.DocumentFilter`` so they
   drop straight into a Curator streaming pipeline when stage 3 is
   scaled out on GPU. Each exposes the Curator-mandated ``score_document``
   / ``keep_document`` pair; we wrap them in ``_text_rule_from_filter`` so
   the runner keeps the same ``Rule = Callable[[row, cfg], RejectReason
   | None]`` signature.

2. **Structured-row rules** (``cultural_ref_coverage``,
   ``turn_count_parity``, ``speaker_ref_integrity``,
   ``mapped_ref_type_consistency``) stay as plain Python callables on
   the Pydantic row. DocumentFilter's contract is single-text-column,
   so cross-field invariants (e.g. KR turn count == EN turn count)
   don't map to it — Pydantic row validation is the right primitive
   there.

All rules are cheap, deterministic, LLM-free. The runner wires them into
``Stage3Output.reject_reasons`` and flips ``valid`` off when any rule
fires.

Default rules (skip gracefully when the dependent data is absent — semi-
final stage-2 rows without a KR rewrite still get structural + mapping
checks):

- ``cultural_ref_coverage`` — every ``dialogue_decomposed.cultural_refs[i]``
  term has a matching ``mapped_refs[j]`` with the same term.
- ``mapped_ref_ko_hangul`` — each mapped_ref.ko is non-empty and contains
  at least one Hangul syllable (catches ``ko="Höm"`` etc.).
- ``mapped_ref_type_consistency`` — when a term appears in both
  ``dialogue_decomposed.cultural_refs`` and ``mapped_refs``, the ``type``
  agrees.
- ``turn_count_parity`` — KR turn count equals EN turn count (when KR is
  populated).
- ``turn_index_order`` — KR turn indices are a monotonic sequence starting
  at 0 (when KR is populated).
- ``speaker_ref_integrity`` — every persona / speaker_persona refers to a
  speaker_name_en that exists in ``row.speakers``.
- ``ascii_ratio`` — KR dialogue's ASCII-char ratio is under
  ``cfg["ascii_ratio_max"]`` (default 0.40) when KR is populated.
- ``mapped_ref_surface`` — each mapped_ref.ko appears as a substring in the
  KR dialogue (when KR is populated).

Opt-in rules (not in ALL_RULES, enable via explicit ``rules=`` list):

- ``mapped_ref_terms_in_source`` — each mapped_ref.term appears verbatim
  in ``source_dialogue``. Off by default because stage-1 decomposers often
  normalise concepts ("pet sitting" from "watch my cat"), so literal
  substring match produces false positives.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from nemo_curator.stages.text.filters.doc_filter import DocumentFilter

from nemos_dream.schemas import RejectReason, Stage3Output

Rule = Callable[[Stage3Output, dict], RejectReason | None]


_HANGUL_RE = re.compile(r"[가-힯]")


# ---- Curator DocumentFilter subclasses (text-level, GPU-scalable) ----


class ASCIIRatioFilter(DocumentFilter):
    """Curator DocumentFilter: keep rows whose KR ASCII-alnum ratio ≤ max."""

    def __init__(self, max_ratio: float = 0.40) -> None:
        super().__init__()
        self.max_ratio = max_ratio

    def score_document(self, text: str) -> float:
        if not text:
            return 0.0
        ascii_count = sum(1 for ch in text if ord(ch) < 128 and ch.isalnum())
        return ascii_count / len(text)

    def keep_document(self, scores: float | list[int | float]) -> bool:
        if isinstance(scores, list):
            scores = scores[0] if scores else 0.0
        return float(scores) <= self.max_ratio


class MappedRefKoHangulFilter(DocumentFilter):
    """Curator DocumentFilter: keep KR ko-terms containing ≥1 Hangul syllable."""

    def score_document(self, text: str) -> float:
        text = (text or "").strip()
        if not text:
            return 0.0
        return 1.0 if _HANGUL_RE.search(text) else 0.0

    def keep_document(self, scores: float | list[int | float]) -> bool:
        if isinstance(scores, list):
            scores = min(scores) if scores else 0.0
        return float(scores) >= 1.0


def _contains_hangul(text: str) -> bool:
    return bool(_HANGUL_RE.search(text))


def _kr_variants(row: Stage3Output) -> list[list]:
    """Every populated KR turn variant on the row.

    Structural rules check each variant so a stale copy — e.g. v4
    ``final_dialogue`` rewritten but legacy ``korean_dialogue`` not updated
    — trips the gate.
    """
    out = []
    if row.final_dialogue:
        out.append(list(row.final_dialogue))
    if row.korean_dialogue and (
        not row.final_dialogue or list(row.korean_dialogue) != list(row.final_dialogue)
    ):
        out.append(list(row.korean_dialogue))
    return out


def _kr_text(row: Stage3Output) -> str:
    parts: list[str] = []
    for variant in _kr_variants(row):
        parts.extend(t.text for t in variant)
    return " \n ".join(parts)


def cultural_ref_coverage(row: Stage3Output, cfg: dict) -> RejectReason | None:
    mapped_terms = {m.term.lower() for m in row.mapped_refs}
    missing = [
        c.term
        for c in row.dialogue_decomposed.cultural_refs
        if c.term.lower() not in mapped_terms
    ]
    if missing:
        return RejectReason(
            stage="stage3.phase2",
            rule="cultural_ref_coverage",
            detail=f"{len(missing)} cultural_refs lack a mapped_ref: {missing[:5]}",
            extra={"missing": missing},
        )
    return None


def mapped_ref_terms_in_source(row: Stage3Output, cfg: dict) -> RejectReason | None:
    joined = " \n ".join(t.text for t in row.source_dialogue).lower()
    missing = [m.term for m in row.mapped_refs if m.term.lower() not in joined]
    if missing:
        return RejectReason(
            stage="stage3.phase2",
            rule="mapped_ref_terms_in_source",
            detail=(
                f"{len(missing)} mapped terms not present in source_dialogue: "
                f"{missing[:5]}"
            ),
            extra={"missing": missing},
        )
    return None


def mapped_ref_ko_hangul(row: Stage3Output, cfg: dict) -> RejectReason | None:
    flt = MappedRefKoHangulFilter()
    bad = [m.term for m in row.mapped_refs if not flt.keep_document(flt.score_document(m.ko))]
    if bad:
        return RejectReason(
            stage="stage3.phase2",
            rule="mapped_ref_ko_hangul",
            detail=f"{len(bad)} mapped_refs have empty or non-Hangul ko: {bad[:5]}",
            extra={"offenders": bad},
        )
    return None


def mapped_ref_type_consistency(row: Stage3Output, cfg: dict) -> RejectReason | None:
    ref_type_by_term = {c.term.lower(): c.type for c in row.dialogue_decomposed.cultural_refs}
    bad: list[dict] = []
    for m in row.mapped_refs:
        decomposed_type = ref_type_by_term.get(m.term.lower())
        if decomposed_type is not None and decomposed_type != m.type:
            bad.append({"term": m.term, "decomposed": decomposed_type, "mapped": m.type})
    if bad:
        return RejectReason(
            stage="stage3.phase2",
            rule="mapped_ref_type_consistency",
            detail=(
                f"{len(bad)} mapped_refs disagree with dialogue_decomposed "
                f"cultural_ref type: {[b['term'] for b in bad][:5]}"
            ),
            extra={"offenders": bad},
        )
    return None


def turn_count_parity(row: Stage3Output, cfg: dict) -> RejectReason | None:
    variants = _kr_variants(row)
    if not variants:
        return None  # KR side not produced yet — skip
    en = len(row.source_dialogue)
    for kr in variants:
        if len(kr) != en:
            return RejectReason(
                stage="stage3.phase2",
                rule="turn_count_parity",
                detail=f"KR turn count {len(kr)} != EN turn count {en}",
                extra={"kr": len(kr), "en": en},
            )
    return None


def turn_index_order(row: Stage3Output, cfg: dict) -> RejectReason | None:
    variants = _kr_variants(row)
    if not variants:
        return None
    for kr in variants:
        expected = list(range(len(kr)))
        actual = [t.index for t in kr]
        if actual != expected:
            return RejectReason(
                stage="stage3.phase2",
                rule="turn_index_order",
                detail=f"KR turn indices {actual} != expected {expected}",
                extra={"actual": actual, "expected": expected},
            )
    return None


def speaker_ref_integrity(row: Stage3Output, cfg: dict) -> RejectReason | None:
    known = {s.name_en for s in row.speakers}
    unknown: list[str] = []
    # v4 path: persona[*].speaker_name_en
    if row.persona:
        unknown.extend(
            p.speaker_name_en for p in row.persona if p.speaker_name_en not in known
        )
    # v3 fallback
    elif row.speaker_personas:
        unknown.extend(
            p.speaker_ref for p in row.speaker_personas if p.speaker_ref not in known
        )
    if unknown:
        return RejectReason(
            stage="stage3.phase2",
            rule="speaker_ref_integrity",
            detail=f"persona refs not in stage-1 speakers: {unknown[:5]}",
            extra={"unknown": unknown, "known": sorted(known)},
        )
    return None


def ascii_ratio(row: Stage3Output, cfg: dict) -> RejectReason | None:
    variants = _kr_variants(row)
    if not variants:
        return None
    max_ratio = float(cfg.get("ascii_ratio_max", 0.40))
    flt = ASCIIRatioFilter(max_ratio=max_ratio) if ASCIIRatioFilter is not None else None
    for kr in variants:
        text = " \n ".join(t.text for t in kr)
        if not text:
            continue
        if flt is not None:
            ratio = flt.score_document(text)
            ok = flt.keep_document(ratio)
        else:  # pragma: no cover — Curator missing
            total = len(text)
            ascii_count = sum(1 for ch in text if ord(ch) < 128 and ch.isalnum())
            ratio = ascii_count / total if total else 0.0
            ok = ratio <= max_ratio
        if not ok:
            return RejectReason(
                stage="stage3.phase2",
                rule="ascii_ratio",
                detail=f"KR ASCII-alnum ratio {ratio:.2f} > max {max_ratio:.2f}",
                extra={"ratio": ratio, "max": max_ratio, "filter": "curator.DocumentFilter"},
            )
    return None


def mapped_ref_surface(row: Stage3Output, cfg: dict) -> RejectReason | None:
    variants = _kr_variants(row)
    if not variants or not row.mapped_refs:
        return None
    for kr in variants:
        text = " \n ".join(t.text for t in kr)
        missing = [m.ko for m in row.mapped_refs if m.ko and m.ko not in text]
        if missing:
            return RejectReason(
                stage="stage3.phase2",
                rule="mapped_ref_surface",
                detail=(
                    f"{len(missing)} mapped_ref.ko missing from KR text: {missing[:5]}"
                ),
                extra={"missing": missing},
            )
    return None


ALL_RULES: list[Rule] = [
    turn_count_parity,
    turn_index_order,
    speaker_ref_integrity,
    ascii_ratio,
    mapped_ref_surface,
    cultural_ref_coverage,
    # mapped_ref_terms_in_source,  # opt-in — see module docstring
    mapped_ref_ko_hangul,
    mapped_ref_type_consistency,
]


def apply(rows: list[Stage3Output], cfg: dict, rules: list[Rule] | None = None) -> None:
    """Mutate ``rows`` in place: append RejectReasons and flip ``valid``.

    Rules only run on rows that are still ``valid`` — a row rejected by
    phase 1 stays rejected and skips rule checks.
    """
    active = rules or ALL_RULES
    for row in rows:
        if not row.valid:
            continue
        for rule in active:
            rr = rule(row, cfg)
            if rr is not None:
                row.reject_reasons.append(rr)
                row.valid = False
                break  # one reject reason per row per phase is enough
