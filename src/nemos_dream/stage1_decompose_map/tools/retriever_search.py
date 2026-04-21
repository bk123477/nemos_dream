"""Embedding similarity search over cultural-map entries via NeMo Retriever.

Requires a pre-built ``retriever_index.npz`` produced by
``python -m nemos_dream.stage1_decompose_map._index_builder``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from nemos_dream.nvidia_clients import NvidiaSyncClient

INDEX_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "stage1" / "retriever_index.npz"
)


class _RetrieverClient(NvidiaSyncClient):
    model_env = "RETRIEVER_MODEL"

    def call(self, texts: list[str], *, input_type: str) -> np.ndarray:
        model = self.model or "nvidia/llama-3.2-nv-embedqa-1b-v2"
        resp = self.openai.embeddings.create(
            model=model,
            input=texts,
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs


@lru_cache(maxsize=1)
def _client() -> _RetrieverClient:
    base_url = os.environ.get("NVIDIA_API_BASE")
    return _RetrieverClient(base_url=base_url) if base_url else _RetrieverClient()


def embed_passages(texts: list[str]) -> np.ndarray:
    return _client().call(texts, input_type="passage")


def embed_query(text: str) -> np.ndarray:
    return _client().call([text], input_type="query")[0]


@lru_cache(maxsize=1)
def _load_index() -> tuple[np.ndarray, list[dict]]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Retriever index not found at {INDEX_PATH}. "
            "Run `python -m nemos_dream.stage1_decompose_map._index_builder` first."
        )
    data = np.load(INDEX_PATH, allow_pickle=True)
    return data["vectors"], list(data["entries"])


def search(term: str, top_k: int = 3, threshold: float = 0.65) -> list[dict]:
    """Return up to ``top_k`` entries with cosine similarity ≥ ``threshold``."""
    vecs, entries = _load_index()
    q = embed_query(term)
    sims = vecs @ q
    order = np.argsort(-sims)[:top_k]
    hits: list[dict] = []
    for i in order:
        score = float(sims[i])
        if score < threshold:
            break
        hit = dict(entries[i])
        hit["score"] = score
        hits.append(hit)
    return hits
