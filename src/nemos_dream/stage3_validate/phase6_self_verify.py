"""Phase 6 — NeMo Agent Toolkit (NAT) self-verify loop.

Per the stage-3 README, this phase lives *between* stages conceptually
but is implemented as a thin ReAct agent that reads
``retry_queue.jsonl`` (emitted by the runner) and re-runs stage 1/2/3
primitives via registered tools until each row is valid or
``max_iter`` is reached.

Tools (all NVIDIA-stack):
- ``stage1_redecompose`` — re-run dialogue decomposition for the row.
- ``maps_ref_redo`` — re-resolve cultural refs (dict → retriever → web).
- ``stage2_rewrite`` — regenerate the KR rewrite with adjusted persona.
- ``websearch_cultural`` — NAT's hosted web-search tool for fresh refs.
- ``revalidate`` — invoke stage-3 phases 1–5 on the repaired row.

The agent itself is constructed lazily via
``nvidia_nat.agent.react_agent.ReActAgent`` so the package stays
importable without NAT installed — stage 3 never executes the loop
during ``runner.run``; the caller (pipeline orchestrator) imports
``run_self_verify`` explicitly.

This module registers tool *adapters*; the actual stage-1 / stage-2
runners are expected to be imported by the caller and passed in as
``StageCallables`` so stage 3 stays decoupled (zero coupling to stage
1/2 per README principle 5).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from nemos_dream.schemas import Stage3Output


@dataclass
class StageCallables:
    """Adapter bundle the orchestrator passes in to wire agent tools.

    Each callable mutates and returns a repaired :class:`Stage3Output`.
    Callables that cannot act on the row (missing upstream context)
    should raise :class:`NotImplementedError` — the agent treats that as
    a failed tool and moves on to the next action.
    """

    stage1_redecompose: Callable[[Stage3Output], Awaitable[Stage3Output]]
    maps_ref_redo: Callable[[Stage3Output], Awaitable[Stage3Output]]
    stage2_rewrite: Callable[[Stage3Output], Awaitable[Stage3Output]]
    websearch_cultural: Callable[[Stage3Output], Awaitable[Stage3Output]]
    revalidate: Callable[[Stage3Output], Awaitable[Stage3Output]]


def build_tools(stages: StageCallables, *, enabled_actions: list[str]) -> list[Any]:
    """Construct NAT tool objects for the ReAct agent.

    Imports NAT lazily so the package stays importable when
    ``nvidia-nat`` is not installed (CI / offline). The ``enabled_actions``
    allowlist mirrors ``configs/stage3/filter.yaml::self_verify.enabled_actions``.
    """
    from nvidia_nat.tool import Tool  # noqa: PLC0415 — lazy NAT import

    registry = {
        "stage1_redecompose": stages.stage1_redecompose,
        "maps_ref_redo": stages.maps_ref_redo,
        "stage2_rewrite": stages.stage2_rewrite,
        "websearch_cultural": stages.websearch_cultural,
        "revalidate": stages.revalidate,
    }
    tools: list[Any] = []
    for name, fn in registry.items():
        if name not in enabled_actions and name != "revalidate":
            continue
        tools.append(
            Tool(
                name=name,
                description=_TOOL_DESCRIPTIONS[name],
                run=fn,
            )
        )
    return tools


_TOOL_DESCRIPTIONS = {
    "stage1_redecompose": (
        "Re-run stage-1 dialogue decomposition on the EN source. Use when "
        "turn_count_parity fails or speech_acts look wrong."
    ),
    "maps_ref_redo": (
        "Re-resolve mapped_refs via dict → retriever → web+llm chain. "
        "Use when cultural_ref_coverage / mapped_ref_ko_hangul / "
        "mapped_ref_type_consistency fails."
    ),
    "stage2_rewrite": (
        "Regenerate the KR rewrite with adjusted persona / register. "
        "Use when register_consistency / persona_style_consistency / "
        "naturalness / property_preservation is low."
    ),
    "websearch_cultural": (
        "Hit NAT's web-search tool for fresh cultural-ref evidence. "
        "Use alongside maps_ref_redo when the dict+retriever path fails."
    ),
    "revalidate": (
        "Re-run stage-3 phases 1–5 on the repaired row. The agent must "
        "call this after every repair to check if the row now passes."
    ),
}


async def run_self_verify(
    row: Stage3Output,
    *,
    stages: StageCallables,
    enabled_actions: list[str],
    max_iter: int = 2,
    model: str = "nvidia/nemotron-3-super-120b-a12b",
) -> Stage3Output:
    """Run the NAT ReAct loop on one invalid row up to ``max_iter`` times.

    Returns the (possibly now-valid) row with ``iter`` incremented per
    repair attempt. Raises if NAT is not installed — this phase is
    opt-in.
    """
    from nvidia_nat.agent.react_agent import ReActAgent  # noqa: PLC0415

    tools = build_tools(stages, enabled_actions=enabled_actions)
    agent = ReActAgent(model=model, tools=tools, max_iterations=max_iter)

    current = row
    for i in range(1, max_iter + 1):
        current.iter = i
        hint_ctx = {
            "reject_reasons": [rr.model_dump() for rr in current.reject_reasons],
            "retry_actions": [a.model_dump() for a in current.retry_actions],
            "judge_reasoning": current.quality.judge_reasoning or {},
        }
        current = await agent.arun(context=hint_ctx, row=current)
        if current.valid:
            break
    return current
