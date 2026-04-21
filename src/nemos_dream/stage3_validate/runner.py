"""Stage 3 entrypoint: ``Stage2Output`` rows → ``Stage3Output`` rows (+ splits).

The runner chains the five scoring phases (schema+dedup → rules →
guardrails → semantic → judge+reward), derives retry hints, then writes
the five README-mandated artifacts under ``output_dir``:

- ``accepted.jsonl`` — rows with ``valid=True``
- ``rejected.jsonl`` — rows with ``valid=False``
- ``retry_queue.jsonl`` — rejected rows with actionable ``retry_actions``
  (self-verify fodder; the NAT agent in phase 6 consumes this file)
- ``dataset_metrics.json`` — absolute quality/diversity telemetry
- ``reject_details.json`` — per-row reject diary for debugging
- ``parse_errors.json`` — line-level schema-parse failures

All LLM / embedding calls are required — stage 3 is NVIDIA-native and
has no offline stubs. The runner builds its clients through
``stage3_validate.clients.build_default_clients`` which reads
``NVIDIA_API_KEY`` from the environment (or ``.env`` if the caller
loaded it). Tests supply their own in-memory ``judge_fn`` / ``reward_fn``
/ ``safety_fn`` / ``embed_fn`` / ``pii_fn`` shims.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from nemos_dream.stage3_validate import (
    clients,
    config,
    dataset_metrics,
    phase1_schema_dedup,
    phase2_rules,
    phase3_guardrails,
    phase4_semantic,
    phase5_judge_reward,
    retry_hints,
)

EmbedFn = Callable[[list[str]], list[list[float]]]
JudgeFn = Callable[..., Awaitable[dict[str, Any]]]
RewardFn = Callable[..., Awaitable[dict[str, float]]]
SafetyFn = Callable[[str], Awaitable[bool]]
PiiFn = Callable[[str], bool]


def _reject_diary(rows: list[Any]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r.valid:
            continue
        out.append(
            {
                "id": r.id,
                "source_dialogue": [t.model_dump() for t in r.source_dialogue],
                "reject_reasons": [rr.model_dump() for rr in r.reject_reasons],
                "retry_actions": [a.model_dump() for a in r.retry_actions],
                "mapped_refs": [m.model_dump() for m in r.mapped_refs],
            }
        )
    return out


def _retry_queue(rows: list[Any]) -> list[Any]:
    """Rejected rows with at least one non-``none`` retry_action.

    These are what the NAT self-verify agent should work on. Safety
    rejects (hard-stop) have ``retry_actions=[{"action":"none", ...}]``
    so they're excluded.
    """
    out = []
    for r in rows:
        if r.valid:
            continue
        actionable = [a for a in r.retry_actions if a.action != "none"]
        if actionable:
            out.append(r)
    return out


async def run_async(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    cfg: config.Stage3Config | None = None,
    embed_fn: EmbedFn | None = None,
    judge_fn: JudgeFn | None = None,
    reward_fn: RewardFn | None = None,
    safety_fn: SafetyFn | None = None,
    pii_fn: PiiFn | None = None,
) -> dict[str, int]:
    """Async end-to-end stage-3 runner.

    If any of the five callables is ``None``, the runner constructs the
    default NVIDIA-native client from ``clients.build_default_clients``.
    A missing ``NVIDIA_API_KEY`` therefore fails loudly at first use —
    that's intentional: stage 3 is NIM-backed end-to-end.
    """
    input_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = cfg or config.load()
    defaults = clients.build_default_clients() if any(
        x is None for x in (embed_fn, judge_fn, reward_fn, safety_fn)
    ) else {}

    embed_fn = embed_fn or defaults.get("embed_fn")
    judge_fn = judge_fn or defaults.get("judge_fn")
    reward_fn = reward_fn or defaults.get("reward_fn")
    safety_fn = safety_fn or defaults.get("safety_fn")
    pii_fn = pii_fn or phase3_guardrails.make_pii_fn()

    if embed_fn is None or judge_fn is None or reward_fn is None or safety_fn is None:
        raise RuntimeError(
            "stage 3 requires NIM clients: set NVIDIA_API_KEY (and load .env "
            "if applicable) or pass embed_fn/judge_fn/reward_fn/safety_fn "
            "explicitly from a test harness."
        )

    rows, parse_errors = phase1_schema_dedup.run(
        input_path,
        embed_fn=embed_fn,
        semantic_threshold=cfg.semantic_cosine_threshold,
    )

    phase2_rules.apply(rows, {"ascii_ratio_max": cfg.ascii_ratio_max})
    await phase3_guardrails.apply_async(rows, safety_fn=safety_fn, pii_fn=pii_fn)
    phase4_semantic.apply(rows, embed_fn=embed_fn, coherence_floor=cfg.intra_kr_coherence_floor)
    await phase5_judge_reward.apply_async(
        rows,
        judge_fn=judge_fn,
        reward_fn=reward_fn,
        axis_floor=cfg.axis_floor,
        aggregate_floor=cfg.aggregate_floor,
        weights=cfg.quality_weights,
    )
    retry_hints.apply(rows)

    accepted = [r for r in rows if r.valid]
    rejected = [r for r in rows if not r.valid]
    retry_rows = _retry_queue(rows)

    (out_dir / "accepted.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in accepted) + ("\n" if accepted else ""),
        encoding="utf-8",
    )
    (out_dir / "rejected.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in rejected) + ("\n" if rejected else ""),
        encoding="utf-8",
    )
    (out_dir / "retry_queue.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in retry_rows) + ("\n" if retry_rows else ""),
        encoding="utf-8",
    )

    metrics = dataset_metrics.compute(accepted, rejected, embed_fn=embed_fn)
    (out_dir / "dataset_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "reject_details.json").write_text(
        json.dumps(_reject_diary(rows), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "parse_errors.json").write_text(
        json.dumps(
            [{"lineno": ln, "detail": d} for ln, d in parse_errors],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "retry_queue": len(retry_rows),
        "parse_errors": len(parse_errors),
    }


def run(input_path: str | Path, output_dir: str | Path, **kwargs: Any) -> dict[str, int]:
    """Sync wrapper over :func:`run_async`."""
    import asyncio

    return asyncio.run(run_async(input_path, output_dir, **kwargs))
