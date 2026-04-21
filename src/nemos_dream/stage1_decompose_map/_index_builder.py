"""Build the retriever index used by the cultural-map cascade.

Embeds every clean seed entry once and persists the vectors as a numpy
``.npz`` at ``configs/stage1/retriever_index.npz``. Entries failing
validator error-level rules, and unreviewed retrieved entries, are excluded —
the index should reflect mappings we trust, not every string in the file.

Run::

    uv run python -m nemos_dream.stage1_decompose_map._index_builder
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nemos_dream.schemas import MappedRef

from ._validator import _check_rules
from .tools.dict_lookup import all_entries
from .tools.retriever_search import INDEX_PATH, embed_passages


def _passage_text(entry: dict) -> str:
    # Anchor the embedding on the Korean mapping (the side queries should pull
    # toward), not on potentially noisy notes.
    return f"{entry['en']}. A {entry['type']} referring to {entry['ko']}"


def _is_clean(entry: dict) -> bool:
    ref = MappedRef(
        term=entry["en"],
        ko=entry.get("ko", ""),
        type=entry.get("type", "other"),
        source="dict",
        retrieved=False,
        notes=entry.get("notes", ""),
    )
    flags = _check_rules(ref)
    return not any(f["severity"] == "error" for f in flags)


def build() -> Path:
    all_ents = all_entries()
    entries = [e for e in all_ents if _is_clean(e)]
    excluded = len(all_ents) - len(entries)
    if excluded:
        print(f"[index_builder] excluded {excluded} entries failing validator rules")
    texts = [_passage_text(e) for e in entries]
    print(f"[index_builder] embedding {len(texts)} entries via NeMo Retriever...")
    vectors = embed_passages(texts)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(INDEX_PATH, vectors=vectors, entries=np.array(entries, dtype=object))
    print(f"[index_builder] wrote {INDEX_PATH} ({vectors.shape})")
    return INDEX_PATH


if __name__ == "__main__":
    build()
