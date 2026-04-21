# Stage 3 — `validate`

> **Owner:** Stage 3 팀 · **Reference impl:** `../nemotron-test/` has a
> working Curator pipeline that seeds phases 1–3.

**Goal.** Consume `Stage2Output` rows and decide which ones are good enough
to keep. Attach per-row `QualityScores` + `RejectReason[]` + `RetryAction[]`,
split into `accepted.jsonl` / `rejected.jsonl`, emit a `retry_queue.jsonl`
that the NAT ReAct self-verify agent consumes, and feed dataset-level
diversity/quality signals to stage 4.

## Contract

| | Schema | Artifact |
|---|---|---|
| Input | `Stage2Output` | `data/stage2/*.jsonl` |
| Output (accepted) | `Stage3Output` | `data/stage3/accepted.jsonl` |
| Output (rejected) | `Stage3Output` | `data/stage3/rejected.jsonl` |
| Output (retry hints) | `Stage3Output` | `data/stage3/retry_queue.jsonl` |
| Output (dataset metrics) | — | `data/stage3/dataset_metrics.json` |

Every row keeps all `Stage2Output` fields verbatim — phase code only fills
in `quality`, flips `valid`, and appends to `reject_reasons` /
`retry_actions`. `iter` stays at 0 on first pass; the self-verify agent
increments it on re-runs.

## Design principles

1. **NVIDIA stack maximally.** Every phase is a NVIDIA component — NeMo
   Curator (`DocumentFilter`, `MinHashLSH`, `SemanticClusterLevelDedup`,
   `PiiModifier`), NeMoGuard content-safety, Nemotron-70B judge via NIM,
   NV-Embed (`llama-3.2-nv-embedqa-1b-v2`) for similarity, Nemotron-4-340B-
   Reward for absolute row scoring, NeMo Agent Toolkit for the self-verify
   loop. No Presidio, no Tavily direct, no EN↔KR cosine-preservation.
2. **Async only.** All LLM/embedding calls fan out through
   `nim_async_client()` + `asyncio.Semaphore` caps (judge/safety/reward
   each have their own semaphore). Curator itself runs its own dataflow;
   we only async-call the NIM-backed judge/reward phases.
3. **Cheap gates first.** Phase order minimises cost: hard rules
   (schema/rule/safety/dedup) kill bad rows before the expensive LLM judge
   and reward passes touch them.
4. **Cultural rewrite ≠ meaning preservation.** Dropped signals:
   EN↔KR embedding cosine (phase 4a), EN/KR length ratio (R1), and
   jailbreak-detect. A high-quality Korean rewrite is allowed to diverge
   semantically from the English source — that's the point.
5. **Zero coupling to stage 1/2.** Stage 3 never imports stage 1 or 2.
   Retry is a *hint* (`retry_actions` + `retry_queue.jsonl`); the NAT
   agent lives outside this package and owns the loop.
6. **Reject = data, not exception.** Data-level failures append a
   `RejectReason` + set `valid=False`. Infrastructure failures (HTTP,
   schema parse) raise and bubble up to the runner.

## Phase pipeline (6 phases, Curator-native dataflow)

```
  data/stage2/out.jsonl          (Stage2Output)
           │
           ▼
  ┌──────────────────────────────┐
  │ Phase 1 — schema + dedup     │  Curator
  │  • Pydantic validation       │  MinHashLSH (lexical, jaccard 0.8)
  │  • exact-match dedup         │  SemanticClusterLevelDedup (cosine ≥ 0.92)
  │  • near-dup collapse         │  NV-Embed (nv-embedqa-e5-v5 / 1b-v2)
  └──────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────┐
  │ Phase 2 — rule-based         │  Curator DocumentFilter subclasses:
  │  (hard gates, cheap)         │   • turn_count_parity (src vs KR length)
  │                              │   • speaker_ref integrity (persona/style)
  │                              │   • honorific/emoji ratio vs register
  │                              │   • ASCII ratio ceiling
  │                              │   • mapped_ref surface-present in KR
  └──────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────┐
  │ Phase 3 — guardrails         │  NeMoGuard content-safety (NIM)
  │                              │  Curator PiiModifier (KR/EN entities)
  │                              │   → quality.safety_pass / pii_pass
  └──────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────┐
  │ Phase 4 — semantic flow      │  NV-Embed: embed each KR turn, mean
  │                              │  cosine between adjacent turns
  │                              │   → quality.intra_kr_coherence
  │                              │  (floor in configs/stage3/filter.yaml)
  └──────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────┐
  │ Phase 5 — LLM judge + reward │  Nemotron-70B-instruct (judge):
  │  (expensive, async fan-out)  │   5 axes 1–5 — property_preservation,
  │                              │   naturalness, cultural_appropriateness,
  │                              │   register_consistency,
  │                              │   persona_style_consistency
  │                              │   + per-axis judge_reasoning
  │                              │  Nemotron-4-340B-Reward:
  │                              │   absolute per-row reward (correctness +
  │                              │   coherence; helpfulness dropped)
  │                              │  → quality.aggregate (weighted mean)
  └──────────────────────────────┘
           │
           ├──────── accepted ────────► data/stage3/accepted.jsonl
           │
           ├──────── rejected ────────► data/stage3/rejected.jsonl
           │
           ▼
  ┌──────────────────────────────┐
  │ Phase 6 — self-verify loop   │  NeMo Agent Toolkit (NAT) ReAct agent
  │  (runs over retry_queue)     │  Tools (all NVIDIA-stack):
  │                              │   • stage1_redecompose
  │                              │   • maps_ref_redo
  │                              │   • stage2_rewrite
  │                              │   • websearch_cultural  (NAT web-search)
  │                              │   • revalidate          (stage3 phase 1–5)
  │                              │  Agent reads quality.judge_reasoning +
  │                              │  reject_reasons + retry_actions (hints)
  │                              │  and decides which tools to run; loops
  │                              │  until valid OR iter == max_iter.
  └──────────────────────────────┘
           │
           ▼
  data/stage3/{accepted,rejected}.jsonl (final, with iter > 0 on retried rows)
  data/stage3/dataset_metrics.json      (absolute diversity/quality)
```

## Dataset-level metrics (phase 5 side-output → `dataset_metrics.json`)

Absolute, not relative. Computed once over the accepted corpus (and a
breakdown over rejected) so stage 4 can report on pipeline health:

| Metric | Source |
|---|---|
| `reward_distribution` | Nemotron-4-340B-Reward per-row, then aggregate |
| `embedding_diversity` | mean pairwise 1 − cosine across NV-Embed of KR dialogues |
| `distinct_n` (1/2/3-gram) | over `korean_dialogue` text |
| `persona_coverage_entropy` | entropy over 9-attribute Persona distribution |
| `scene_coverage_entropy` | entropy over (setting, relationship_type, topics) |
| `cultural_ref_diversity` | distinct terms / total refs across `mapped_refs` |
| `length_stats` | turn count, char count per row (mean/p50/p95) |
| `reject_breakdown` | count per (stage, rule) in `reject_reasons` |
| `retry_stats` | iter histogram, action counts, valid-after-retry rate |

## Module layout (suggested)

```
stage3_validate/
├── runner.py                 # run(input, output_dir) → {"accepted": N, "rejected": M, "retry": K}
├── phase1_schema_dedup.py    # Curator MinHashLSH + SemanticClusterLevelDedup
├── phase2_rules.py           # Curator DocumentFilter subclasses
├── phase3_guardrails.py      # NeMoGuard + PiiModifier
├── phase4_semantic.py        # NV-Embed intra-KR coherence
├── phase5_judge_reward.py    # Nemotron-70B judge + Nemotron-340B-Reward (async)
├── dataset_metrics.py        # phase-5 side-output aggregator
└── retry_hints.py            # map reject_reasons + judge_reasoning → RetryAction[]
```

The NAT agent does NOT live in this package — it's orchestrated from
`scripts/run_pipeline.py` (or a dedicated `scripts/self_verify.py`) and
imports stage 1/2/3 runners as plain functions. Stage 3 exposes only its
`runner.run(...)` + a `revalidate(row) → Stage3Output` entrypoint that the
agent tool-binds to.

## Config (`configs/stage3/filter.yaml`)

- Drop `semantic_cosine_floor` (deprecated — EN↔KR cosine isn't a signal).
- Add `intra_kr_coherence_floor` (e.g. 0.55 baseline; tune after first run).
- Add `self_verify.max_iter` (default 2) + `self_verify.enabled_actions`
  (allowlist for NAT agent).
- Judge + reward model IDs live here, not in code.

## Install + run

```bash
uv sync
uv run python -m scripts.run_stage --stage 3 \
    --input data/stage2/out.jsonl \
    --output-dir data/stage3/
```

## Testing

- `tests/stage3/` — phase-level unit tests (mock NIM where possible; use a
  tiny fixture set for end-to-end).
- `tests/test_schemas.py` must stay green after any contract touch.

## Invariants (repeat from `.claude/docs/stage-contracts.md`)

- A record never drops `Stage2Output` fields, regardless of `valid`.
- `valid == False` requires ≥ 1 `reject_reasons` entry.
- `quality` fields are optional (`None`) — populate what the phase
  measured. `semantic_cosine` is deprecated.
- `retry_actions` are **hints**, not commands; the NAT agent may ignore
  them.
- `iter == 0` on first pass; the agent increments on re-run.