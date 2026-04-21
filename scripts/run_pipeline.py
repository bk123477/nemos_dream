"""End-to-end pipeline: stages 1 → 2 → 3 → 4.

Chains the four stage runners with the file-path convention from
``.claude/docs/stage-owner-guide.md``::

    <output_dir>/stage1/stage1_output.jsonl
    <output_dir>/stage2/stage2_output.jsonl
    <output_dir>/stage3/{accepted,rejected}.jsonl
    <output_dir>/reports/{report.*, sft.jsonl}

Usage::

    uv run python scripts/run_pipeline.py \
        --input data/raw/sample_input.jsonl \
        --output-dir data/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/soda.jsonl",
        help="Path to the raw JSONL. If --hf-spec is set, the file is (re)materialized here before stage 1.",
    )
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument(
        "--hf-spec",
        default="allenai/soda",
        help="HuggingFace dataset id to download into --input. Pass empty string to skip the download.",
    )
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-limit", type=int, default=4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Stage 1: overwrite output; stage 2: run with --no-resume. Default resumes both.",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=None,
        help="Optional row cap forwarded to stage 2.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    stage1_out = out / "stage1" / "stage1_output.jsonl"
    stage2_out = out / "stage2" / "stage2_output.jsonl"
    stage3_dir = out / "stage3"
    stage4_dir = out / "reports"

    if args.hf_spec:
        from nemos_dream.io_utils import materialize_hf_to_jsonl

        n = materialize_hf_to_jsonl(
            args.hf_spec,
            args.input,
            limit=args.hf_limit,
            split=args.hf_split,
        )
        print(f"hf: wrote {n} rows from {args.hf_spec}:{args.hf_split} → {args.input}")

    from nemos_dream.stage1_decompose_map.runner import run as run_stage1
    from nemos_dream.stage2_translate_rewrite.runner import run as run_stage2
    from nemos_dream.stage3_validate.runner import run as run_stage3
    from nemos_dream.stage4_report.runner import run as run_stage4

    n1 = run_stage1(args.input, stage1_out, overwrite=args.overwrite)
    print(f"stage 1: {n1} rows → {stage1_out}")

    n2 = run_stage2(
        stage1_out,
        stage2_out,
        num_records=args.num_records,
        resume=not args.overwrite,
    )
    print(f"stage 2: {n2} rows → {stage2_out}")

    counts = run_stage3(stage2_out, stage3_dir)
    print(f"stage 3: {counts} → {stage3_dir}")

    artifacts = run_stage4(
        stage3_dir / "accepted.jsonl",
        stage3_dir / "rejected.jsonl",
        stage4_dir,
    )
    print(f"stage 4: {artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
