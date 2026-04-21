"""Run a single stage by number.

Thin dispatcher to each stage's ``runner.run(...)``. Stage owners define
the real work; this file only wires CLI flags to the runner signatures
agreed in ``.claude/docs/stage-owner-guide.md``.

Usage::

    uv run python scripts/run_stage.py --stage 1 --input ... --output ...
    uv run python scripts/run_stage.py --stage 2 --input ... --output ...
    uv run python scripts/run_stage.py --stage 3 --input ... --output-dir ...
    uv run python scripts/run_stage.py --stage 4 \
        --accepted ... --rejected ... --output-dir ...
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    parser.add_argument("--accepted")
    parser.add_argument("--rejected")
    parser.add_argument(
        "--hf-spec",
        default=None,
        help="Stage 1 only: HuggingFace dataset id to materialize into --input before running.",
    )
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-limit", type=int, default=4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Stage 1 only: overwrite the output file instead of resuming.",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=None,
        help="Optionally limit the number of input rows. Currently used by stage 2.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from existing output and queue files when supported by the stage.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing output and start a fresh run when supported by the stage.",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    if args.stage == 1:
        input_path = args.input or "data/raw/soda.jsonl"
        hf_spec = args.hf_spec
        if hf_spec is None and not args.input:
            hf_spec = "allenai/soda"
        if hf_spec:
            from nemos_dream.io_utils import materialize_hf_to_jsonl

            n = materialize_hf_to_jsonl(
                hf_spec,
                input_path,
                limit=args.hf_limit,
                split=args.hf_split,
            )
            print(f"hf: wrote {n} rows from {hf_spec}:{args.hf_split} → {input_path}")

        from nemos_dream.stage1_decompose_map.runner import run

        output_path = args.output or "data/stage1/stage1_output.jsonl"
        n = run(input_path, output_path, overwrite=args.overwrite)
        print(f"stage 1: wrote {n} rows → {output_path}")
    elif args.stage == 2:
        from nemos_dream.stage2_translate_rewrite.runner import run

        n = run(
            args.input,
            args.output,
            num_records=args.num_records,
            resume=args.resume,
        )
        print(f"stage 2: wrote {n} rows → {args.output}")
    elif args.stage == 3:
        from nemos_dream.stage3_validate.runner import run

        counts = run(args.input, args.output_dir)
        print(f"stage 3: {counts} → {args.output_dir}")
    elif args.stage == 4:
        from nemos_dream.stage4_report.runner import run

        artifacts = run(args.accepted, args.rejected, args.output_dir)
        print(f"stage 4: {artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
