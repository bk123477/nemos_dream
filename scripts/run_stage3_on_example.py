"""Run stage 3 on data/stage2/example_output.jsonl.

Writes outputs to data/stage3/.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from nemos_dream.stage3_validate.runner import run  # noqa: E402


def main() -> None:
    summary = run(
        input_path=Path("data/stage2/example_output.jsonl"),
        output_dir=Path("data/stage3"),
    )
    total = summary["accepted"] + summary["rejected"]
    print(
        f"total: {total}  accepted: {summary['accepted']}  "
        f"rejected: {summary['rejected']}  "
        f"retry_queue: {summary['retry_queue']}  "
        f"parse_errors: {summary['parse_errors']}"
    )


if __name__ == "__main__":
    main()
