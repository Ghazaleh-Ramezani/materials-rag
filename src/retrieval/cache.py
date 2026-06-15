"""A small semantic cache to skip redundant retrieval+LLM work.

Two layers:
1. exact match on a normalized query string;
2. semantic match — embed the query and return a cached entry whose query
   embedding has cosine similarity above a threshold.

Entries persist to a JSONL file so the cache survives restarts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import config
from src.indexing.embedders import Embedder, get_embedder

_WS_RE = re.compile(r"\s+")


def _normalize(query: str) -> str:
    return _WS_RE.sub(" ", query.strip().lower())


class SemanticCache:
    def __init__(self, embedder: Optional[Embedder] = None, threshold: float | None = None):
        self.embedder = embedder or get_embedder()
        self.threshold = config.cache_similarity_threshold if threshold is None else threshold
        self._exact: Dict[str, dict] = {}
        self._keys: List[str] = []  # normalized queries, aligned with _matrix rows
        self._matrix: Optional[np.ndarray] = None  # (n, dim) normalized embeddings
        self._load()

    # ---- lookup ----
    def get(self, query: str) -> Optional[dict]:
        norm = _normalize(query)
        if norm in self._exact:
            hit = dict(self._exact[norm])
            hit["_cache"] = "exact"
            return hit
        if self._matrix is None or not len(self._keys):
            return None
        q = self.embedder.encode([query])  # (1, dim), normalized
        sims = (self._matrix @ q[0])  # cosine, since both normalized
        best = int(np.argmax(sims))
        if float(sims[best]) >= self.threshold:
            hit = dict(self._exact[self._keys[best]])
            hit["_cache"] = "semantic"
            hit["_cache_similarity"] = round(float(sims[best]), 4)
            return hit
        return None

    # ---- write ----
    def set(self, query: str, payload: dict) -> None:
        norm = _normalize(query)
        if norm in self._exact:
            return
        self._exact[norm] = payload
        self._keys.append(norm)
        emb = self.embedder.encode([query])
        self._matrix = emb if self._matrix is None else np.vstack([self._matrix, emb])
        self._append_disk(norm, payload)

    # ---- persistence ----
    def _append_disk(self, norm: str, payload: dict) -> None:
        path = config.cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"query": norm, "payload": payload}, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        path = config.cache_path
        if not path.exists():
            return
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            return
        self._keys = [r["query"] for r in rows]
        for r in rows:
            self._exact[r["query"]] = r["payload"]
        self._matrix = self.embedder.encode(self._keys)
