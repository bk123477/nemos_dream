"""Shared stage-2 pipeline mode helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal


PipelineMode = Literal["default", "direct", "naive_persona"]

DEFAULT_PIPELINE_MODE: PipelineMode = "default"
DIRECT_PIPELINE_MODE: PipelineMode = "direct"
NAIVE_PERSONA_PIPELINE_MODE: PipelineMode = "naive_persona"
PIPELINE_MODES: tuple[PipelineMode, ...] = (
    DEFAULT_PIPELINE_MODE,
    DIRECT_PIPELINE_MODE,
    NAIVE_PERSONA_PIPELINE_MODE,
)


def normalize_pipeline_mode(value: str | None) -> PipelineMode:
    mode = str(value or DEFAULT_PIPELINE_MODE).strip().lower()
    if mode not in PIPELINE_MODES:
        raise ValueError(
            f"Unsupported stage2 pipeline mode: {value!r}. "
            f"Expected one of {', '.join(PIPELINE_MODES)}."
        )
    return mode  # type: ignore[return-value]


def uses_persona(mode: str | None) -> bool:
    return normalize_pipeline_mode(mode) != DIRECT_PIPELINE_MODE


def uses_mapped_refs(mode: str | None) -> bool:
    return normalize_pipeline_mode(mode) == DEFAULT_PIPELINE_MODE


def requires_step4(mode: str | None) -> bool:
    return normalize_pipeline_mode(mode) == DEFAULT_PIPELINE_MODE


def default_step3_filename(mode: str | None) -> str:
    normalized = normalize_pipeline_mode(mode)
    if normalized == DEFAULT_PIPELINE_MODE:
        return "stage2_rewrite.jsonl"
    if normalized == DIRECT_PIPELINE_MODE:
        return "stage2_direct.jsonl"
    return "stage2_naive_persona.jsonl"


def default_stage2_output_filename(mode: str | None) -> str:
    normalized = normalize_pipeline_mode(mode)
    if normalized == DEFAULT_PIPELINE_MODE:
        return "out.jsonl"
    if normalized == DIRECT_PIPELINE_MODE:
        return "out.direct.jsonl"
    return "out.naive_persona.jsonl"


def default_step3_output_path(base_dir: str | Path, mode: str | None) -> Path:
    return Path(base_dir) / default_step3_filename(mode)


def default_stage2_output_path(base_dir: str | Path, mode: str | None) -> Path:
    return Path(base_dir) / default_stage2_output_filename(mode)
