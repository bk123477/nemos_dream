"""Run stage 3 end-to-end on the stage-2 sample via live NVIDIA NIM.

Reads ``data/stage2/sample_v3.jsonl``, pushes every row through
phase 1–5 (Presidio PII + NeMoGuard safety + NV-Embed + Nemotron
judge + Nemotron-4-340B-Reward), and writes the six README artefacts
under ``data/stage3/``.

Requires ``NVIDIA_API_KEY`` in ``.env`` (loaded on import).
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from nemos_dream.stage3_validate.runner import run  # noqa: E402


def main() -> None:
    summary = run(
        input_path=Path("data/stage2/sample_v3.jsonl"),
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
