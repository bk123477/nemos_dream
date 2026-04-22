# Stage 2 — `translate_rewrite`

> **Owner:** Minki Hong

`Stage2Output` is produced in two internal passes, with selectable pipeline
variants:

1. `run_step3.py`
   First-pass Korean generation.
   - `default`: persona retrieval + mapped-ref-aware rewrite
   - `direct`: no persona, no mapped_refs, direct translation
   - `naive_persona`: persona name assignment without mapped_refs
   Output:
   - `data/stage2/stage2_rewrite.jsonl`
   - `data/stage2/stage2_direct.jsonl`
   - `data/stage2/stage2_naive_persona.jsonl`
2. `run_step4.py`
   Finalization into `final_dialogue`.
   - `default`: naturalization / consistency polish
   - `direct`, `naive_persona`: copy `step3_korean_dialogue` into `final_dialogue`
   Output:
   - `data/stage2/out.jsonl`
   - `data/stage2/out.direct.jsonl`
   - `data/stage2/out.naive_persona.jsonl`

The stage-level entrypoint is still `runner.py::run(input, output)`, so the
repo-wide commands in `scripts/run_stage.py` and `scripts/run_pipeline.py`
keep working.

## Files

| File | Role |
|---|---|
| `persona_downloader.py` | Download the public Google Drive persona bank into `data/persona_age_gender/` |
| `pipeline_modes.py` | Shared mode constants + default output naming for `default` / `direct` / `naive_persona` |
| `persona_retriever.py` | Persona selection logic adapted from the original NVIDIA step3 implementation |
| `run_step3.py` | `Stage1Output -> Stage2Output` intermediate (`persona` + `step3_korean_dialogue`) |
| `run_step4.py` | `stage2_rewrite.jsonl -> out.jsonl` final dialogue polish |
| `runner.py` | Stage-2 end-to-end wrapper used by the repo-wide stage runner |

## Contract

| | Schema | Artifact |
|---|---|---|
| Input | `Stage1Output` | `data/stage1/*.jsonl` |
| Intermediate | `Stage2Output` (partial) | `data/stage2/stage2_rewrite.jsonl` |
| Final | `Stage2Output` | `data/stage2/out.jsonl` |

The final artifact populates the repo’s current v4 fields:

- `step3_korean_dialogue`
- `persona`
- `final_dialogue`

`korean_dialogue` is mirrored automatically by `Stage2Output`’s validator in
`schemas.py`, so downstream readers that still use the legacy field name keep
working.

## Pipeline Modes

| Mode | Persona | mapped_refs | Step 4 |
|---|---|---|---|
| `default` | O | O | polish |
| `direct` | X | X | copy step3 -> final |
| `naive_persona` | O | X | copy step3 -> final |

## Download Persona Data

The persona bank is downloaded from the public Google Drive folder configured
for this stage and stored under `data/persona_age_gender/`.

```bash
uv sync
uv run python -m nemos_dream.stage2_translate_rewrite.persona_downloader
```

If you need a clean re-download:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.persona_downloader --force
```

## Run Only Step 3

This consumes stage-1 output and writes the first internal stage-2 artifact.

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.run_step3 \
  --input data/stage1/example_output.jsonl \
  --output data/stage2/stage2_rewrite.jsonl
```

For the repo-wide naming convention, use `data/stage1/out.jsonl` once stage 1
is wired in end-to-end:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.run_step3 \
  --input data/stage1/out.jsonl \
  --output data/stage2/stage2_rewrite.jsonl
```

Direct-translation ablation:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.run_step3 \
  --input data/stage1/out.jsonl \
  --pipeline-mode direct \
  --output data/stage2/stage2_direct.jsonl
```

Persona-only ablation:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.run_step3 \
  --input data/stage1/out.jsonl \
  --pipeline-mode naive_persona \
  --output data/stage2/stage2_naive_persona.jsonl
```

## Run Only Step 4

This consumes the intermediate rewrite file and produces the final stage-2
artifact used by downstream stages.

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.run_step4 \
  --input data/stage2/stage2_rewrite.jsonl \
  --output data/stage2/out.jsonl
```

For ablation modes, `run_step4.py` keeps the same CLI but simply mirrors
`step3_korean_dialogue` into `final_dialogue`:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.run_step4 \
  --input data/stage2/stage2_direct.jsonl \
  --output data/stage2/out.direct.jsonl \
  --pipeline-mode direct
```

## Run Stage 2 End-To-End

Repo-wide stage runner:

```bash
uv run python scripts/run_stage.py --stage 2 \
  --input data/stage1/out.jsonl \
  --output data/stage2/out.jsonl
```

Generate only a small sample for local testing:

```bash
uv run python scripts/run_stage.py --stage 2 \
  --input data/stage1/example_output.jsonl \
  --output data/stage2/example_output.jsonl \
  --num-records 3 \
  --no-resume
```

Run a specific ablation mode through the repo-wide stage runner:

```bash
uv run python scripts/run_stage.py --stage 2 \
  --input data/stage1/out.jsonl \
  --pipeline-mode naive_persona
```

Stage-2 package directly:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.runner \
  --input data/stage1/out.jsonl \
  --output data/stage2/out.jsonl
```

For local testing against the current mock data:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.runner \
  --input data/stage1/example_output.jsonl \
  --output data/stage2/out.jsonl
```

Choose the ablation pipeline end-to-end inside stage 2:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.runner \
  --input data/stage1/out.jsonl \
  --pipeline-mode direct
```

## Resume And Retry

Both `run_step3.py` and `run_step4.py` process one row at a time and append
results immediately, so interrupted runs can be resumed safely.

Generated side files:

- `<output>.retry-errors.jsonl`
  Rows that failed during Data Designer validation/generation and should be
  retried later.
- `<output>.invalid.jsonl`
  Rows where generation returned but normalization/schema checks failed.

Resume an interrupted run:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.run_step3 \
  --input data/stage1/out.jsonl \
  --output data/stage2/stage2_rewrite.jsonl \
  --resume
```

Retry the step-3 retry queue later:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.run_step3 \
  --retry-errors-from data/stage2/stage2_rewrite.jsonl
```

Retry the step-4 retry queue later:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.run_step4 \
  --retry-errors-from data/stage2/out.jsonl
```

If you want to consume the invalid queue instead of the retry-errors queue:

```bash
uv run python -m nemos_dream.stage2_translate_rewrite.run_step4 \
  --retry-errors-from data/stage2/out.jsonl \
  --retry-source invalid
```

## Notes

- All default paths are repo-relative from the `nemos_dream/` root.
- `run_step3.py` writes mode-specific intermediate names:
  `stage2_rewrite.jsonl`, `stage2_direct.jsonl`, `stage2_naive_persona.jsonl`.
- `runner.py` keeps the same public interface and chooses mode-specific
  internal/final filenames when `--output` is omitted.
- `data/persona_age_gender/` and `data/stage2/artifacts/` are ignored in git.
