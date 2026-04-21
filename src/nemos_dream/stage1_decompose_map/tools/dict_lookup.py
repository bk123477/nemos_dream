"""Exact/normalized lookup against ``configs/stage1/cultural_map_seed.json``.

The file's flat shape — ``{term: {ko, type, notes, source?, reviewed?}}`` —
matches the team seed convention. Entries with ``source="retrieved"`` and
``reviewed=False`` are served only when ``USE_UNREVIEWED=1`` so the
self-growing retrieved tail never silently leaks into production output.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

SEED_PATH = Path(__file__).resolve().parents[4] / "configs" / "stage1" / "cultural_map_seed.json"


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


@lru_cache(maxsize=1)
def _load_index() -> dict[str, dict]:
    raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    allow_unreviewed = os.environ.get("USE_UNREVIEWED", "").lower() in {"1", "true", "yes"}
    index: dict[str, dict] = {}
    for term, entry in raw.items():
        if term.startswith("_"):
            continue
        if not isinstance(entry, dict):
            continue
        source = entry.get("source", "seed")
        reviewed = bool(entry.get("reviewed", source != "retrieved"))
        if source == "retrieved" and not reviewed and not allow_unreviewed:
            continue
        key = _normalize(term)
        index[key] = {
            "en": term,
            "ko": entry.get("ko", ""),
            "type": entry.get("type", "other"),
            "notes": entry.get("notes", ""),
            "approximate": bool(entry.get("approximate", False)),
            "keep": bool(entry.get("keep", False)),
            "source": source,
            "reviewed": reviewed,
        }
    return index


def lookup(term: str) -> dict | None:
    """Return normalized entry dict if ``term`` matches, else None."""
    return _load_index().get(_normalize(term))


def all_entries() -> list[dict]:
    return list(_load_index().values())


def append_entry(
    term: str,
    ko: str,
    *,
    ref_type: str = "other",
    notes: str = "",
) -> bool:
    """Append a retrieved mapping to the seed file under ``source="retrieved"``.

    Idempotent: if ``term`` already exists, returns False without writing.
    Clears the in-memory cache so subsequent ``lookup`` calls see the entry.
    """
    term = term.strip()
    ko = ko.strip()
    if not term or not ko:
        return False
    if lookup(term) is not None:
        return False

    raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    raw[term] = {
        "ko": ko,
        "type": ref_type,
        "notes": notes,
        "source": "retrieved",
        "reviewed": False,
    }
    SEED_PATH.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _load_index.cache_clear()
    return True
