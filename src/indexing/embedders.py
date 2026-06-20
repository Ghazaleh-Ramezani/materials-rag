"""Embedding backends.

Two implementations behind one interface:

* ``SentenceTransformerEmbedder`` — the real thing (``all-MiniLM-L6-v2`` etc.).
  Used by default; requires a one-time model download.
* ``HashingEmbedder`` — a deterministic, dependency-free hashed bag-of-words
  embedder. No downloads, no network. Used as an automatic fallback so the repo
  (and CI) runs fully offline. It is *not* semantically strong — it exists so the
  pipeline is always runnable end-to-end.

``get_embedder()`` picks the backend from config and silently falls back to
hashing if sentence-transformers is unavailable.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Protocol

import numpy as np

from src.config import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")


class HashingEmbedder:
    """Hashed bag-of-words -> fixed-dim vector. Deterministic and offline."""

    def __init__(self, dim: int | None = None):
        self.dim = dim or config.embedding_dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        tokens = _TOKEN_RE.findall(text.lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 1) % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        mat = np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros(
            (0, self.dim), dtype="float32"
        )
        return _l2_normalize(mat)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name or config.embedding_model)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        emb = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return emb.astype("float32")


def get_embedder() -> Embedder:
    backend = config.embedding_backend.lower()
    if backend in {"hashing", "hash"}:
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:  # ImportError or model download failure
        print(
            f"[embed] sentence-transformers unavailable ({exc!r}); "
            f"falling back to offline HashingEmbedder."
        )
        return HashingEmbedder()
