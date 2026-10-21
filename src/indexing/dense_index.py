"""Dense vector index backed by FAISS (inner product on normalized vectors,
i.e. cosine similarity)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import faiss

from src.config import config
from src.indexing.embedders import Embedder, get_embedder
from src.schemas import Chunk


class DenseIndex:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self._index: faiss.Index | None = None
        self._chunk_ids: List[str] = []

    def build(self, chunks: List[Chunk]) -> "DenseIndex":
        embeddings = self.embedder.encode([c.text for c in chunks])
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # vectors are L2-normalized -> IP == cosine
        index.add(embeddings)
        self._index = index
        self._chunk_ids = [c.id for c in chunks]
        return self

    def search_dense(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("DenseIndex not built/loaded.")
        q = self.embedder.encode([query])
        k = min(k, len(self._chunk_ids))
        scores, idxs = self._index.search(q, k)
        out: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((self._chunk_ids[idx], float(score)))
        return out

    # ---- persistence ----
    def save(self, path: Path | None = None) -> None:
        path = path or (config.index_dir / "faiss.index")
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        path.with_suffix(".ids.json").write_text(json.dumps(self._chunk_ids), encoding="utf-8")
        meta = {"embedder": self.embedder.name, "dim": int(self._index.d), "n": len(self._chunk_ids)}
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None, embedder: Embedder | None = None) -> "DenseIndex":
        """Load an index and check it was built with the current embedder.

        Vectors from two embedders live in different spaces even when their
        dimensions agree (the offline hashing embedder and all-MiniLM-L6-v2
        are both 384-d), so a mismatch would return plausible-looking but
        meaningless neighbours instead of an error.
        """
        path = path or (config.index_dir / "faiss.index")
        obj = cls(embedder=embedder)
        obj._index = faiss.read_index(str(path))
        obj._chunk_ids = json.loads(path.with_suffix(".ids.json").read_text(encoding="utf-8"))
        meta_path = path.with_suffix(".meta.json")
        built_with = None
        if meta_path.exists():
            built_with = json.loads(meta_path.read_text(encoding="utf-8"))["embedder"]
        if built_with != obj.embedder.name:
            raise ValueError(
                f"Dense index at {path} was built with embedder {built_with!r}, but the current "
                f"embedder is {obj.embedder.name!r}. Rebuild it with `python -m scripts.build_indexes`."
            )
        return obj
