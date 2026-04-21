# nemos_dream

> **NVIDIA Nemotron Hackathon 2026 — Track C (SDG).**
> English SNS datasets → Korean datasets for k-sovereign LLM training.
> **Not translation — cultural rewriting.**

A 4-stage synthetic-data-generation pipeline with one teammate per stage.
Stages share a single Pydantic contract (`src/nemos_dream/schemas.py`) and
layered JSONL artifacts under `data/stage{N}/`.

## Pipeline

```
┌──────────┐   ┌────────────────────┐   ┌─────────────────────┐   ┌───────────────┐   ┌──────────────┐
│ RawInput │ → │ Stage 1             │ → │ Stage 2              │ → │ Stage 3        │ → │ Stage 4       │
│ (EN SNS) │   │ 사회언어학적 분해 +  │   │ 번역 → rewrite       │   │ validate       │   │ report + SFT  │
│          │   │ 문화적 요소 추가     │   │ (metadata 추가 가능) │   │                │   │               │
│ data/raw/│   │ data/stage1/        │   │ data/stage2/         │   │ data/stage3/   │   │ data/reports/ │
└──────────┘   └────────────────────┘   └─────────────────────┘   └───────────────┘   └──────────────┘
                 nemo_dream_step1 ref     new                        nemotron-test ref   new
```

| Stage | Package | Owner | Input schema | Output schema |
|---|---|---|---|---|
| 1 | `stage1_decompose_map` | TBD | `RawInput` | `Stage1Output` |
| 2 | `stage2_translate_rewrite` | TBD | `Stage1Output` | `Stage2Output` |
| 3 | `stage3_validate` | TBD | `Stage2Output` | `Stage3Output` |
| 4 | `stage4_report` | TBD | `Stage3Output` | `Stage4Sft` + report |

Stage 2 also depends on a local persona bank under `data/persona_age_gender/`,
which is created by `uv run python -m nemos_dream.stage2_translate_rewrite.persona_downloader`.

**Stage owners start here:** `.claude/docs/stage-owner-guide.md` — one-page
playbook per stage. For deeper reference: `.claude/docs/architecture.md`
(data flow), `.claude/docs/stage-contracts.md` (schema per boundary), and
each stage's `README.md`.

## Quickstart

```bash
# 0. Clone, then copy env template
cp .env.example .env
# Edit .env: set NVIDIA_API_KEY (build.nvidia.com) and TAVILY_API_KEY (optional)

# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (single flat set — base + NVIDIA Nemotron stack)
uv sync

# 3. Download the stage-2 persona bank required for rewrite runs.
uv run python -m nemos_dream.stage2_translate_rewrite.persona_downloader

# 4. Run tests — the schema round-trip test must pass on a fresh clone.
uv run pytest

# 5. Run a single stage
uv run python scripts/run_stage.py --stage 1 \
    --input data/raw/sample_input.jsonl \
    --output data/stage1/out.jsonl

# Stage 2 requires data/persona_age_gender/, which the downloader creates.

# 6. End-to-end run
uv run python scripts/run_pipeline.py --input data/raw/sample_input.jsonl
```

## NVIDIA stack at a glance

| Stage | NVIDIA tools |
|---|---|
| 1 | NIM · NeMo Data Designer · NeMo Retriever · (opt) NeMo Agent Toolkit |
| 2 | NIM · HF `Nemotron-Personas-Korea` |
| 3 | NeMo Curator · NeMoGuard · NIM (70B judge) · Nemotron-4-340B-Reward |
| 4 | (none — pure analysis) |

Full model IDs, env vars, and endpoints: `.claude/docs/nvidia-stack.md`.

## Repo layout

```
nemos_dream/
├── pyproject.toml                   uv-managed, single flat dep list (base + NVIDIA stack)
├── configs/                         pipeline.yaml + stage{1..4}/*.yaml
├── data/                            raw/ + persona_age_gender/ + stage{1,2,3}/ + reports/
├── src/nemos_dream/
│   ├── schemas.py                    ★ canonical contract (only file with real logic)
│   ├── io_utils.py                   shared JSONL / HF loaders
│   ├── nvidia_clients.py             NIM / Retriever / judge / safety / reward client factories
│   ├── stage1_decompose_map/         사회언어학적 분해 + 문화적 요소
│   ├── stage2_translate_rewrite/     번역 → rewrite (post-processing)
│   ├── stage3_validate/              validate / filter / score
│   └── stage4_report/                report + SFT export
├── scripts/                         run_stage.py, run_pipeline.py
└── tests/                           schema round-trip (real) + per-stage tests (owners add)
```

## What's implemented right now

**Structure, not logic.** Every `.py` file outside `schemas.py` is a
signature stub that raises `NotImplementedError`. Stage owners are free to
add, rename, or split submodules inside their own stage — the only locked
interface is the schema contract between stages.

The schema contract (`schemas.py`) and the test fixtures
(`tests/fixtures/sample_rows.py`) are real — `uv run pytest
tests/test_schemas.py` passes on a fresh clone.

See `.claude/docs/conventions.md` for the stub-file policy.

## License & attribution

Code structure adapts two practice repos from the same team:

- `../nemo_dream_step1/` — stages 1+2 reference implementation
- `../nemotron-test/` — stage 3 reference implementation
