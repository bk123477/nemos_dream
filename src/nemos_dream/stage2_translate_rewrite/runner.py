"""Stage 2 entrypoint: ``Stage1Output`` rows → ``Stage2Output`` rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from nemos_dream.stage2_translate_rewrite.run_step3 import (
    count_jsonl_rows,
    run as run_step3,
)
from nemos_dream.stage2_translate_rewrite.run_step4 import run as run_step4


def run(
    input_path: str | Path,
    output_path: str | Path,
    *,
    num_records: int | None = None,
    resume: bool = True,
) -> int:
    """Run stage 2 end-to-end. Returns number of final rows written."""

    final_output_path = Path(output_path)
    intermediate_output_path = final_output_path.with_name("stage2_rewrite.jsonl")

    run_step3(
        input_path=input_path,
        output_path=intermediate_output_path,
        num_records=num_records,
        resume=resume,
    )
    run_step4(
        input_path=intermediate_output_path,
        output_path=final_output_path,
        num_records=num_records,
        resume=resume,
    )
    return count_jsonl_rows(final_output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stage 2 end-to-end.")
    parser.add_argument("--input", required=True, help="Input Stage1 JSONL path.")
    parser.add_argument("--output", required=True, help="Output Stage2 JSONL path.")
    parser.add_argument("--num-records", type=int, default=None, help="Optionally limit the number of input rows processed in stage 2.")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Resume from existing output and queue files.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore existing output and start a fresh run.")
    parser.set_defaults(resume=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    count = run(
        args.input,
        args.output,
        num_records=args.num_records,
        resume=args.resume,
    )
    print(f"Stage 2 wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
