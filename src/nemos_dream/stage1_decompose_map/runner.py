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
from nemos_dream.io_utils import append_jsonl, read_jsonl, read_processed_ids, write_jsonl
from nemos_dream.schemas import RawInput, Stage1Output

from ._validator import validate_refs
from .cultural_map import map_refs
from .decompose import decompose


def _row_id(row: RawInput) -> str:
    """Preserve an existing id, else fall back to ``soda-<original_index>``
    to match the convention in ``data/stage1/example_output.jsonl``."""
    return row.id or f"soda-{row.original_index}"


def run(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> int:
    """Run stage 1 end-to-end. Returns total row count in the output file.

    Resume semantics: if ``output_path`` exists and ``overwrite`` is False,
    already-processed ids are skipped and new results are appended.
    A partial/corrupt final line (from a killed prior run) is truncated.
    With ``overwrite=True`` the output file is replaced from scratch.
    """
    if input_path is None:
        raise ValueError("stage 1 requires --input")
    if output_path is None:
        raise ValueError("stage 1 requires --output")

    load_dotenv()  # populate NVIDIA_API_KEY / TAVILY_API_KEY from .env if present
    apply_proxy_patches()  # force httpx custom-transport (Data Designer) to honor HTTPS_PROXY
    use_llm_verify = os.environ.get("STAGE1_VALIDATE_LLM", "").lower() in {"1", "true", "yes"}

    out_p = Path(output_path)
    if overwrite and out_p.exists():
        out_p.unlink()

    processed_ids = read_processed_ids(out_p) if not overwrite else set()

    rows = list(read_jsonl(input_path, RawInput))
    rows_to_process = [r for r in rows if _row_id(r) not in processed_ids]

    if not rows_to_process:
        return len(processed_ids)

    decomposed = decompose(rows_to_process)

    stage1_rows: list[Stage1Output] = []
    for row, dr in zip(rows_to_process, decomposed, strict=True):
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

    if processed_ids:
        append_jsonl(out_p, stage1_rows)
    else:
        write_jsonl(out_p, stage1_rows)
    return len(processed_ids) + len(stage1_rows)
