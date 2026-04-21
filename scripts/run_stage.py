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
    args = parser.parse_args()

    if args.stage == 1:
        from nemos_dream.stage1_decompose_map.runner import run

        n = run(args.input, args.output)
        print(f"stage 1: wrote {n} rows → {args.output}")
    elif args.stage == 2:
        from nemos_dream.stage2_translate_rewrite.runner import run

        n = run(args.input, args.output)
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
