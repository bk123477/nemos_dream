"""Stage 1 entrypoint: read ``RawInput`` rows → emit ``Stage1Output`` rows.

Orchestration::

    io_utils.read_jsonl(input)
        → decompose(rows)          # stage 1a: sociolinguistic meta
        → map_refs(refs, dialogue) # stage 1b: EN→KR cultural refs
        → validate_refs(mapped)    # attach validation flags
        → Stage1Output             # bundle + write
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from nemos_dream._proxy_patch import apply_proxy_patches
from nemos_dream.io_utils import read_jsonl, write_jsonl
from nemos_dream.schemas import RawInput, Stage1Output

from ._validator import validate_refs
from .cultural_map import map_refs
from .decompose import decompose


def _row_id(row: RawInput) -> str:
    """Preserve an existing id, else fall back to ``soda-<original_index>``
    to match the convention in ``data/stage1/example_output.jsonl``."""
    return row.id or f"soda-{row.original_index}"


def run(input_path: str | Path, output_path: str | Path) -> int:
    """Run stage 1 end-to-end. Returns number of rows written."""
    if input_path is None:
        raise ValueError("stage 1 requires --input")
    if output_path is None:
        raise ValueError("stage 1 requires --output")

    load_dotenv()  # populate NVIDIA_API_KEY / TAVILY_API_KEY from .env if present
    apply_proxy_patches()  # force httpx custom-transport (Data Designer) to honor HTTPS_PROXY
    use_llm_verify = os.environ.get("STAGE1_VALIDATE_LLM", "").lower() in {"1", "true", "yes"}

    rows = list(read_jsonl(input_path, RawInput))
    decomposed = decompose(rows)

    stage1_rows: list[Stage1Output] = []
    for row, dr in zip(rows, decomposed, strict=True):
        mapped = map_refs(
            list(dr.dialogue_decomposed.cultural_refs),
            dialogue=dr.turns,
        )
        mapped = validate_refs(mapped, use_llm=use_llm_verify)
        stage1_rows.append(Stage1Output(
            id=_row_id(row),
            original_index=row.original_index,
            source_dialogue=dr.turns,
            speakers=dr.speakers,
            scene=dr.scene,
            dialogue_decomposed=dr.dialogue_decomposed,
            mapped_refs=mapped,
        ))

    return write_jsonl(output_path, stage1_rows)
