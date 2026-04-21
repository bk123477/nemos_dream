"""Derive ``RetryAction`` hints from reject_reasons + judge_reasoning.

These are hints for the NAT ReAct self-verify agent (phase 6) — the agent
may ignore, combine, or override them. Stage 3 never executes them; it
only emits the hint list on each row.

Mapping (first match wins, one action per hint group):

- Any ``stage3.phase3`` reject with rule="safety" → no retry (hard stop).
- ``stage3.phase1:near_dup`` / ``semantic_dup`` → ``none`` (drop dupes).
- ``stage3.phase2:cultural_ref_coverage`` /
  ``mapped_ref_terms_in_source`` / ``mapped_ref_ko_hangul`` /
  ``mapped_ref_type_consistency`` / low ``cultural_appropriateness`` /
  phase-4 cosine floor → ``maps_ref_redo`` (+ ``websearch_cultural``
  when culturally weak).
- ``stage3.phase2:turn_count_parity`` or low ``register_consistency`` /
  low ``persona_style_consistency`` / low ``naturalness`` /
  low ``property_preservation`` → ``stage2_rewrite``.
- Otherwise → ``none`` with a diagnostic summary.
"""

from __future__ import annotations

from nemos_dream.schemas import RejectReason, RetryAction, Stage3Output


def _has(rr: list[RejectReason], stage: str, rule: str) -> bool:
    return any(r.stage == stage and r.rule == rule for r in rr)


def derive(row: Stage3Output) -> list[RetryAction]:
    if not row.reject_reasons:
        return []

    # Hard-stop: safety rejects are not retried.
    if _has(row.reject_reasons, "stage3.phase3", "safety"):
        return [
            RetryAction(
                action="none",
                reason_summary="safety reject — drop, do not retry",
            )
        ]

    # Dedup rejects: drop the duplicate, don't regenerate.
    if _has(row.reject_reasons, "stage3.phase1", "near_dup") or _has(
        row.reject_reasons, "stage3.phase1", "semantic_dup"
    ):
        return [RetryAction(action="none", reason_summary="duplicate — drop")]

    actions: list[RetryAction] = []
    q = row.quality

    mapping_issues = (
        _has(row.reject_reasons, "stage3.phase2", "cultural_ref_coverage")
        or _has(row.reject_reasons, "stage3.phase2", "mapped_ref_terms_in_source")
        or _has(row.reject_reasons, "stage3.phase2", "mapped_ref_ko_hangul")
        or _has(row.reject_reasons, "stage3.phase2", "mapped_ref_type_consistency")
        or _has(row.reject_reasons, "stage3.phase2", "mapped_ref_surface")
        or _has(row.reject_reasons, "stage3.phase4", "en_ko_mapping_cosine")
        or (q.cultural_appropriateness or 5) < 3
    )
    if mapping_issues:
        actions.append(
            RetryAction(
                action="maps_ref_redo",
                reason_summary="mapped_refs failed surface/script/type/cosine gates",
                hints={"axis": "cultural_appropriateness"},
            )
        )
        actions.append(
            RetryAction(
                action="websearch_cultural",
                reason_summary="re-resolve cultural refs from updated web context",
            )
        )

    # Structural mismatch on turn count points at stage-1 decomposition.
    if _has(row.reject_reasons, "stage3.phase2", "turn_count_parity"):
        actions.append(
            RetryAction(
                action="stage1_redecompose",
                reason_summary="KR turn count disagrees with EN — re-decompose dialogue",
                hints={"rule": "turn_count_parity"},
            )
        )

    rewrite_issues = (
        _has(row.reject_reasons, "stage3.phase2", "turn_index_order")
        or _has(row.reject_reasons, "stage3.phase2", "ascii_ratio")
        or _has(row.reject_reasons, "stage3.phase2", "speaker_ref_integrity")
        or _has(row.reject_reasons, "stage3.phase4", "intra_kr_coherence")
        or (q.register_consistency or 5) < 3
        or (q.persona_style_consistency or 5) < 3
        or (q.naturalness or 5) < 3
        or (q.property_preservation or 5) < 3
    )
    if rewrite_issues:
        actions.append(
            RetryAction(
                action="stage2_rewrite",
                reason_summary=(
                    "turn parity / register / persona style / naturalness / "
                    "property_preservation below floor"
                ),
                hints={"axis": "register_consistency"},
            )
        )

    aggregate_floor = _has(row.reject_reasons, "stage3.phase5", "aggregate_floor")
    if aggregate_floor and not actions:
        actions.append(
            RetryAction(
                action="stage2_rewrite",
                reason_summary="aggregate judge score below floor",
                hints={"aggregate": q.aggregate},
            )
        )

    if not actions:
        actions.append(RetryAction(action="none", reason_summary="no actionable signal"))
    return actions


def apply(rows: list[Stage3Output]) -> None:
    for row in rows:
        row.retry_actions = derive(row)
