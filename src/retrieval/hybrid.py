"""Hybrid retrieval = sparse (BM25) + dense (FAISS) fused with Reciprocal Rank
Fusion, with an optional cross-encoder reranking stage on top."""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.config import config
from src.indexing.dense_index import DenseIndex
from src.indexing.sparse_index import SparseIndex
from src.schemas import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, List[Tuple[str, float]]],
    k_constant: int = 60,
) -> List[Tuple[str, float, List[str]]]:
    """Combine several ranked lists of (chunk_id, score) into one.

    RRF score for an item = sum over lists of 1 / (k_constant + rank), where rank
    is 1-based. Returns (chunk_id, fused_score, source_list_names) sorted desc.
    """
    fused: Dict[str, float] = {}
    sources: Dict[str, List[str]] = {}
    for list_name, results in ranked_lists.items():
        for rank, (cid, _score) in enumerate(results, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_constant + rank)
            sources.setdefault(cid, []).append(list_name)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [(cid, score, sources[cid]) for cid, score in ordered]


class CrossEncoderReranker:
    """Reranks (query, passage) pairs. Falls back to lexical overlap offline."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name or config.reranker_model)
        except Exception as exc:
            print(f"[rerank] cross-encoder unavailable ({exc!r}); using lexical overlap.")

    def _lexical_score(self, query: str, text: str) -> float:
        q = set(query.lower().split())
        t = set(text.lower().split())
        if not q:
            return 0.0
        return len(q & t) / len(q)

    def rerank(
        self, query: str, candidates: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []
        if self._model is not None:
            pairs = [(query, c.chunk.text) for c in candidates]
            scores = self._model.predict(pairs)
        else:
            scores = [self._lexical_score(query, c.chunk.text) for c in candidates]
        for c, s in zip(candidates, scores):
            c.score = float(s)
            c.sources = list(dict.fromkeys(c.sources + ["reranker"]))
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]


class HybridRetriever:
    def __init__(
        self,
        sparse: SparseIndex,
        dense: DenseIndex,
        chunks: List[Chunk],
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.sparse = sparse
        self.dense = dense
        self._by_id: Dict[str, Chunk] = {c.id: c for c in chunks}
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        use_reranker: bool | None = None,
    ) -> List[RetrievedChunk]:
        k = k or config.final_k
        use_reranker = config.use_reranker if use_reranker is None else use_reranker

        sparse_hits = self.sparse.search_sparse(query, config.sparse_k)
        dense_hits = self.dense.search_dense(query, config.dense_k)
        fused = reciprocal_rank_fusion(
            {"sparse": sparse_hits, "dense": dense_hits},
            k_constant=config.rrf_k_constant,
        )

        # how many to keep before (optional) reranking
        pool = config.rerank_candidates if use_reranker else k
        candidates: List[RetrievedChunk] = []
        for cid, score, srcs in fused[:pool]:
            chunk = self._by_id.get(cid)
            if chunk is not None:
                candidates.append(RetrievedChunk(chunk=chunk, score=score, sources=srcs))

        if use_reranker and self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_k=k)
        return candidates[:k]
