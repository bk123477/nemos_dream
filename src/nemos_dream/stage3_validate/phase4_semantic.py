"""Phase 4 — intra-KR semantic flow via NV-Embed.

Mean cosine between adjacent Korean turns, computed with NV-Embed
(``llama-3.2-nv-embedqa-1b-v2`` by default). Replaces EN↔KR semantic
cosine (deprecated because cultural rewriting intentionally breaks
surface equivalence).

``embed_fn`` is required — the caller wires
``stage3_validate.clients.EmbedClient.embed_fn()`` or an equivalent
NV-Embed-backed callable. Rows without a populated KR dialogue record
``intra_kr_coherence_source="skipped_no_kr"`` and skip scoring; rows
below ``coherence_floor`` are rejected.
"""

from __future__ import annotations

from collections.abc import Callable

from nemos_dream.schemas import RejectReason, Stage3Output

EmbedFn = Callable[[list[str]], list[list[float]]]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _adjacent_cosine(turns: list[str], embed_fn: EmbedFn) -> float:
    if len(turns) < 2:
        return 1.0
    embeds = embed_fn(turns)
    scores = [_cosine(embeds[i], embeds[i + 1]) for i in range(len(embeds) - 1)]
    return sum(scores) / len(scores)


def apply(
    rows: list[Stage3Output],
    *,
    embed_fn: EmbedFn,
    coherence_floor: float = 0.55,
) -> None:
    """Populate ``row.quality.intra_kr_coherence`` and reject below floor."""
    for row in rows:
        kr = row.final_dialogue or row.korean_dialogue
        if not kr:
            row.quality.judge_reasoning = {
                **(row.quality.judge_reasoning or {}),
                "intra_kr_coherence_source": "skipped_no_kr",
            }
            continue

        turns = [t.text for t in kr]
        score = _adjacent_cosine(turns, embed_fn)
        row.quality.intra_kr_coherence = round(score, 4)
        row.quality.judge_reasoning = {
            **(row.quality.judge_reasoning or {}),
            "intra_kr_coherence_source": "nv_embed",
        }

        if not row.valid:
            continue
        if score < coherence_floor:
            row.reject_reasons.append(
                RejectReason(
                    stage="stage3.phase4",
                    rule="intra_kr_coherence",
                    detail=(
                        f"mean adjacent-turn cosine {score:.3f} "
                        f"< floor {coherence_floor:.3f}"
                    ),
                    extra={"score": score, "floor": coherence_floor},
                )
            )
            row.valid = False
