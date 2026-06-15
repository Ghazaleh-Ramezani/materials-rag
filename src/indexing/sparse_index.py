"""Sparse lexical index (BM25 via rank-bm25)."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from src.config import config
from src.schemas import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseIndex:
    """Thin wrapper around BM25Okapi keeping chunk-id alignment."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: List[str] = []

    def build(self, chunks: List[Chunk]) -> "SparseIndex":
        corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self._chunk_ids = [c.id for c in chunks]
        return self

    def search_sparse(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("SparseIndex not built/loaded.")
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._chunk_ids, scores), key=lambda x: x[1], reverse=True
        )
        return [(cid, float(s)) for cid, s in ranked[:k]]

    # ---- persistence ----
    def save(self, path: Path | None = None) -> None:
        path = path or (config.index_dir / "bm25.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self._chunk_ids}, fh)

    @classmethod
    def load(cls, path: Path | None = None) -> "SparseIndex":
        path = path or (config.index_dir / "bm25.pkl")
        obj = cls()
        with path.open("rb") as fh:
            data = pickle.load(fh)
        obj._bm25 = data["bm25"]
        obj._chunk_ids = data["chunk_ids"]
        return obj
