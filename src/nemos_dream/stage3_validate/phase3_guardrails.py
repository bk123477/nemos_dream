"""Phase 3 — NeMoGuard content-safety + Presidio PII guardrails.

Both checks hit NVIDIA-hosted NIM endpoints via
``stage3_validate.clients``:

- ``safety_fn(text) -> bool`` — wraps ``SafetyClient.call`` against
  ``nvidia/llama-3.1-nemoguard-8b-content-safety``.
- ``pii_fn(text) -> bool`` — Presidio AnalyzerEngine. (Nemo-Curator 1.1
  no longer ships a ``PiiModifier`` — the Python package moved its PII
  workflow behind the ``curator pii`` CLI / GPU Dask pipeline. Presidio
  is what Curator wraps under the hood, so invoking it directly is the
  same primitive with zero Dask overhead.)

The row text scanned is the joined narrative + source_dialogue +
final_dialogue + korean_dialogue so PII anywhere on the row trips the
gate. Both are async-ready (``safety_fn`` is awaited) so the runner can
fan them out under an ``asyncio.Semaphore`` just like phase 5.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from nemos_dream.schemas import RejectReason, Stage3Output

SafetyFn = Callable[[str], Awaitable[bool]]
PiiFn = Callable[[str], bool]


def _row_text(row: Stage3Output) -> str:
    parts: list[str] = []
    if row.scene and row.scene.narrative_en:
        parts.append(row.scene.narrative_en)
    parts.extend(t.text for t in row.source_dialogue)
    parts.extend(t.text for t in row.final_dialogue)
    parts.extend(t.text for t in row.korean_dialogue)
    return " \n ".join(parts)


_PII_ENTITIES = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "IBAN_CODE",
    "CRYPTO",
)


def make_pii_fn(score_threshold: float = 0.6) -> PiiFn:
    """Build a ``PiiFn`` backed by Presidio's ``AnalyzerEngine``.

    Returns a callable ``(text) -> bool`` — ``True`` when no PII entity
    at or above ``score_threshold`` is detected. We deliberately skip
    ``PERSON`` / ``LOCATION`` because the dialogue pipeline legitimately
    mentions character names (Mike, 철수) and city/country terms.
    """
    from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415

    analyzer = AnalyzerEngine()

    def _detect(text: str) -> bool:
        if not text:
            return True
        hits = analyzer.analyze(text=text, language="en", entities=list(_PII_ENTITIES))
        for h in hits:
            if h.score >= score_threshold:
                return False
        return True

    return _detect


async def apply_async(
    rows: list[Stage3Output],
    *,
    safety_fn: SafetyFn,
    pii_fn: PiiFn,
    concurrency: int = 4,
) -> None:
    """Async phase-3 mutator.

    ``safety_fn`` is awaited per row; ``pii_fn`` is sync (Curator's
    ``PiiModifier`` is not awaitable). Both populate
    ``quality.safety_pass`` / ``quality.pii_pass`` regardless of whether
    the row was already rejected upstream, so stage 4 can report
    guardrail pass-rates on the full population.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _safe(row: Stage3Output) -> bool:
        async with sem:
            return await safety_fn(_row_text(row))

    safety_results = await asyncio.gather(*(_safe(r) for r in rows))

    for row, safety_ok in zip(rows, safety_results, strict=True):
        text = _row_text(row)
        pii_ok = pii_fn(text)
        row.quality.safety_pass = safety_ok
        row.quality.pii_pass = pii_ok
        if not row.valid:
            continue
        if not safety_ok:
            row.reject_reasons.append(
                RejectReason(
                    stage="stage3.phase3",
                    rule="safety",
                    detail="content flagged by NeMoGuard safety check",
                )
            )
            row.valid = False
            continue
        if not pii_ok:
            row.reject_reasons.append(
                RejectReason(
                    stage="stage3.phase3",
                    rule="pii",
                    detail="PII entity detected by Curator PiiModifier",
                )
            )
            row.valid = False


def apply(
    rows: list[Stage3Output],
    *,
    safety_fn: SafetyFn,
    pii_fn: PiiFn,
    concurrency: int = 4,
) -> None:
    """Sync entry that wraps :func:`apply_async` via ``asyncio.run``."""
    asyncio.run(apply_async(rows, safety_fn=safety_fn, pii_fn=pii_fn, concurrency=concurrency))
