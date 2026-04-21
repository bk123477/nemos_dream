"""Shared IO helpers — JSONL read/write + Hugging Face dataset loader.

These are the only "real code" utilities outside of ``schemas.py``. Every
stage ``runner.py`` is expected to read its input via :func:`read_jsonl`
and write its output via :func:`write_jsonl` so the files stay in a
consistent shape across stages.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def read_jsonl(path: str | Path, model: type[ModelT]) -> Iterator[ModelT]:
    """Yield one validated ``model`` instance per non-empty line of ``path``.

    Blank lines are skipped. Any JSON or validation error is re-raised with
    the offending line number so callers can fix their data file.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield model.model_validate(json.loads(line))
            except Exception as exc:  # noqa: BLE001 — re-raise with context
                raise ValueError(f"{p}:{lineno}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[BaseModel]) -> int:
    """Write ``rows`` as one JSON object per line. Returns the row count.

    Creates parent directories if they don't exist. Overwrites any existing
    file at ``path``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
            count += 1
    return count


def load_hf_dataset(
    spec: str,
    *,
    limit: int | None = None,
    text_field: str = "text",
    id_field: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Load a HuggingFace dataset and yield ``{"id", "source_text"}`` dicts.

    ``spec`` is ``"owner/name"`` or ``"owner/name:split"`` (default split:
    ``"train"``). ``id_field=None`` auto-generates ``f"{name}:{idx}"`` ids.

    Requires the optional ``datasets`` dependency (already in
    ``pyproject.toml``). Import is deferred so the module loads cheaply when
    the loader isn't used.
    """
    from datasets import load_dataset  # local import — heavy

    name, _, split = spec.partition(":")
    split = split or "train"
    ds = load_dataset(name, split=split)
    short_name = name.rsplit("/", 1)[-1]

    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        src = row.get(text_field)
        if not src:
            continue
        row_id = str(row[id_field]) if id_field else f"{short_name}:{idx}"
        yield {"id": row_id, "source_text": src}
