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


def append_jsonl(path: str | Path, rows: Iterable[BaseModel]) -> int:
    """Append ``rows`` to ``path`` (one JSON object per line). Returns the count appended."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
            count += 1
    return count


def read_processed_ids(path: str | Path, id_field: str = "id") -> set[str]:
    """Return the set of ``id_field`` values already present in ``path``.

    If the final line is a partial/corrupt JSON (e.g. process was killed mid-write),
    it is truncated from the file in place so a subsequent append starts clean.
    Missing file → empty set.
    """
    p = Path(path)
    if not p.exists():
        return set()

    ids: set[str] = set()
    raw_lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    good_lines: list[str] = []
    truncated = False

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            good_lines.append(line)
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            truncated = True
            break
        rid = obj.get(id_field)
        if rid:
            ids.add(str(rid))
        good_lines.append(line)

    if truncated:
        p.write_text("".join(good_lines), encoding="utf-8")
    return ids


def materialize_hf_to_jsonl(
    spec: str,
    output_path: str | Path,
    *,
    limit: int | None = None,
    split: str = "train",
) -> int:
    """Download ``spec`` from HuggingFace and write ``RawInput`` rows to ``output_path``.

    Writes in the same shape as ``data/stage1/example_input.jsonl`` so the
    stage 1 runner can consume it unchanged. Returns the row count written.
    """
    full_spec = spec if ":" in spec else f"{spec}:{split}"
    rows = load_hf_dataset(full_spec, limit=limit)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_hf_dataset(
    spec: str,
    *,
    limit: int | None = None,
    dialogue_field: str = "dialogue",
    speakers_field: str = "speakers",
    narrative_field: str = "narrative",
    id_field: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Load a HuggingFace dataset and yield ``RawInput``-shaped dicts.

    Output dicts match ``nemos_dream.schemas.RawInput``::

        {"id", "original_index", "dialogue", "speakers", "narrative"}

    ``spec`` is ``"owner/name"`` or ``"owner/name:split"`` (default split:
    ``"train"``). ``id_field=None`` auto-generates ``f"{name}-{idx}"`` ids.
    ``original_index`` is the row's position within the chosen split.

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
        dialogue = row.get(dialogue_field)
        speakers = row.get(speakers_field)
        if not dialogue or not speakers:
            continue
        row_id = str(row[id_field]) if id_field else f"{short_name}-{idx}"
        yield {
            "id": row_id,
            "original_index": idx,
            "dialogue": list(dialogue),
            "speakers": list(speakers),
            "narrative": row.get(narrative_field, "") or "",
        }
