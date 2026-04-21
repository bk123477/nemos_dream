"""Phase 1 — Pydantic validation + NV-Embed semantic dedup (Curator-native).

Stage 3 keeps the Curator pipeline strictly on the NVIDIA stack. Curator's
fuzzy (MinHashLSH) and exact dedup workflows live under
``nemo_curator.stages.deduplication.fuzzy`` / ``.exact`` but both require
``cudf`` / ``cupy`` (GPU-only). On a CPU box we therefore lean on NV-Embed
(the same primitive Curator's ``SemanticClusterLevelDedup`` composes)
with a two-band pairwise cosine filter:

- ``cosine ≥ 0.99`` → ``rule="exact_dup"`` — near-identical surface
- ``0.92 ≤ cosine < 0.99`` → ``rule="semantic_dup"`` — meaning near-dup

A single NV-Embed pass covers both, so we pay one embedding-batch cost.
The caller wires ``embed_fn`` from
``stage3_validate.clients.EmbedClient.embed_fn()`` or an equivalent
NV-Embed-backed callable.

Dedup input document: ``scene.narrative_en`` + ``source_dialogue`` joined
— stable whether the KR rewrite is populated or not.

Outputs per row:
- ``valid=False`` + ``RejectReason(stage="stage3.phase1", rule="schema")``
  for schema parse failures (at the reader boundary).
- ``valid=False`` + ``RejectReason(stage="stage3.phase1", rule="exact_dup"
  | "semantic_dup")`` for dedup hits. First occurrence wins.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from nemos_dream.schemas import RejectReason, Stage2Output, Stage3Output

EmbedFn = Callable[[list[str]], list[list[float]]]


def _row_text(row: Stage2Output) -> str:
    parts: list[str] = []
    if row.scene and row.scene.narrative_en:
        parts.append(row.scene.narrative_en)
    parts.extend(t.text for t in row.source_dialogue)
    return " \n ".join(parts)


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def read_stage2(path: str | Path) -> tuple[list[Stage2Output], list[tuple[int, str]]]:
    """Return (validated rows, [(lineno, error_detail), …])."""
    p = Path(path)
    valid: list[Stage2Output] = []
    errors: list[tuple[int, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                valid.append(Stage2Output.model_validate(json.loads(line)))
            except Exception as exc:  # noqa: BLE001
                errors.append((lineno, f"{type(exc).__name__}: {exc}"))
    return valid, errors


def semantic_dedup_ids(
    rows: list[Stage2Output],
    *,
    embed_fn: EmbedFn,
    exact_threshold: float = 0.99,
    semantic_threshold: float = 0.92,
) -> tuple[set[str], set[str]]:
    """Pairwise NV-Embed cosine dedup. Returns ``(exact_dup_ids, semantic_dup_ids)``."""
    if not rows:
        return set(), set()
    texts = [_row_text(r) for r in rows]
    embs = embed_fn(texts)

    exact: set[str] = set()
    semantic: set[str] = set()
    for i in range(len(rows)):
        if rows[i].id in exact or rows[i].id in semantic:
            continue
        for j in range(i + 1, len(rows)):
            if rows[j].id in exact or rows[j].id in semantic:
                continue
            c = _cosine(embs[i], embs[j])
            if c >= exact_threshold:
                exact.add(rows[j].id)
            elif c >= semantic_threshold:
                semantic.add(rows[j].id)
    return exact, semantic


def to_stage3(
    row: Stage2Output,
    *,
    reject_reasons: list[RejectReason] | None = None,
) -> Stage3Output:
    rr = reject_reasons or []
    return Stage3Output(**row.model_dump(), valid=not rr, reject_reasons=list(rr))


def run(
    input_path: str | Path,
    *,
    embed_fn: EmbedFn,
    exact_threshold: float = 0.99,
    semantic_threshold: float = 0.92,
) -> tuple[list[Stage3Output], list[tuple[int, str]]]:
    """Phase 1 entrypoint. ``embed_fn`` is required (NV-Embed callable)."""
    valid_rows, parse_errors = read_stage2(input_path)
    exact, semantic = semantic_dedup_ids(
        valid_rows,
        embed_fn=embed_fn,
        exact_threshold=exact_threshold,
        semantic_threshold=semantic_threshold,
    )

    out: list[Stage3Output] = []
    for row in valid_rows:
        reasons: list[RejectReason] = []
        if row.id in exact:
            reasons.append(
                RejectReason(
                    stage="stage3.phase1",
                    rule="exact_dup",
                    detail=f"NV-Embed cosine ≥ {exact_threshold}",
                )
            )
        elif row.id in semantic:
            reasons.append(
                RejectReason(
                    stage="stage3.phase1",
                    rule="semantic_dup",
                    detail=f"NV-Embed cosine ≥ {semantic_threshold}",
                )
            )
        out.append(to_stage3(row, reject_reasons=reasons))
    return out, parse_errors
