"""
Materials Science RAG Pipeline
==============================
Hybrid retrieval (BM25 + dense) with RRF fusion, cross-encoder re-ranking,
and an entity-aware semantic cache.

Author: Ghazaleh Ramezani, Ph.D. (Concordia University)
Corpus: Ramezani et al., Micromachines 16(4):393, 2025 (+ related work)

Design references:
  - Robertson & Zaragoza (2009)  -- BM25
  - Reimers & Gurevych (2019)     -- Sentence-BERT (bi-encoder)
  - Nogueira & Cho (2019)         -- Passage re-ranking with BERT (cross-encoder)
  - Cormack et al. (2009)         -- Reciprocal Rank Fusion
  - Lewis et al. (2020)           -- Retrieval-Augmented Generation
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class Chunk:
    chunk_id: str
    section: str
    text: str


@dataclass
class Result:
    chunk: Chunk
    score: float
    method: str


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    """
    Fuse multiple rankings (lists of doc indices, best-first) into one.
    RRF score for doc d = sum over rankings of 1 / (k + rank_d).
    k=60 is the canonical constant (Cormack et al. 2009). RRF needs no score
    normalisation -- it only uses rank position, so BM25 and cosine scales
    never have to be reconciled.
    """
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns zeros if the range is degenerate."""
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)


class HybridRetriever:
    """
    Fuses BM25 (lexical) and dense (semantic) retrieval. Two fusion strategies:

      - "rrf"      Reciprocal Rank Fusion. Rank-only, needs no score
                   normalisation — robust default when the two score scales
                   (unbounded BM25 vs cosine in [-1,1]) are hard to reconcile.
      - "weighted" alpha * dense_norm + (1-alpha) * bm25_norm, after per-query
                   min-max normalisation. `alpha` is a tunable hyperparameter:
                   with a labelled dev set you can optimise the domain-specific
                   balance between lexical and semantic signal.

    Keeping both lets you `compare_configs` on a dev set and pick per corpus.
    """

    def __init__(
        self,
        embed_model: str = "all-MiniLM-L6-v2",
        fusion: str = "rrf",          # "rrf" | "weighted"
        alpha: float = 0.5,           # only used when fusion == "weighted"
    ):
        assert fusion in ("rrf", "weighted")
        self.embedder = SentenceTransformer(embed_model)
        self.fusion = fusion
        self.alpha = alpha
        self.chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._emb: Optional[np.ndarray] = None

    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self._bm25 = BM25Okapi([self._tok(c.text) for c in chunks])
        self._emb = self.embedder.encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        )

    def retrieve(self, query: str, k: int = 50) -> List[Result]:
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        dense_scores = self._emb @ q
        bm25_scores = np.array(self._bm25.get_scores(self._tok(query)))

        if self.fusion == "rrf":
            dense_rank = list(np.argsort(dense_scores)[::-1])
            bm25_rank = list(np.argsort(bm25_scores)[::-1])
            fused = reciprocal_rank_fusion([dense_rank, bm25_rank])
            return [Result(self.chunks[i], s, "hybrid_rrf") for i, s in fused[:k]]

        # weighted
        combined = self.alpha * _minmax(dense_scores) + \
            (1 - self.alpha) * _minmax(bm25_scores)
        top = np.argsort(combined)[::-1][:k]
        return [Result(self.chunks[i], float(combined[i]), "hybrid_weighted") for i in top]


class Reranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model)

    def rerank(self, query: str, candidates: List[Result], k: int = 4) -> List[Result]:
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        order = np.argsort(scores)[::-1][:k]
        return [Result(candidates[i].chunk, float(scores[i]), "reranked") for i in order]


class SemanticCache:
    """
    Caches answers keyed by query embedding; reuses a cached answer if cosine
    similarity > threshold. Entity-aware bypass: queries containing entity
    patterns (sample IDs, numeric values + units) skip the cache, because
    "conductivity of sample 5" and "conductivity of sample 6" embed almost
    identically (cosine > 0.95) but have different answers.
    """

    ENTITY_PATTERNS = [
        r"\bsample\s*\d+",
        r"\d+\.?\d*\s*(s/m|mpa|wt%|nm|c\b|k\b)",
        r"\bfigure\s*\d+", r"\btable\s*\d+",
    ]

    def __init__(self, embedder, threshold: float = 0.95):
        self.embedder = embedder
        self.threshold = threshold
        self._keys: List[np.ndarray] = []
        self._answers: List[str] = []

    def _is_entity_specific(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(p, q) for p in self.ENTITY_PATTERNS)

    def get(self, query: str) -> Optional[str]:
        if self._is_entity_specific(query) or not self._keys:
            return None
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = np.array([k @ q for k in self._keys])
        best = int(np.argmax(sims))
        if sims[best] > self.threshold:
            return self._answers[best]
        return None

    def put(self, query: str, answer: str) -> None:
        if self._is_entity_specific(query):
            return
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        self._keys.append(q)
        self._answers.append(answer)


class RAGPipeline:
    def __init__(
        self,
        corpus: List[Tuple[str, str, str]],
        embed_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fusion: str = "rrf",          # "rrf" | "weighted"
        alpha: float = 0.5,
        use_cache: bool = True,
        llm_fn: Optional[Callable[[str, List[Chunk]], str]] = None,
    ):
        self.chunks = [Chunk(cid, sec, txt) for cid, sec, txt in corpus]
        self.retriever = HybridRetriever(embed_model, fusion=fusion, alpha=alpha)
        self.retriever.index(self.chunks)
        self.reranker = Reranker(rerank_model)
        self.cache = SemanticCache(self.retriever.embedder) if use_cache else None
        self.llm_fn = llm_fn or self._default_llm

    @staticmethod
    def _default_llm(query: str, chunks: List[Chunk]) -> str:
        ctx = "\n".join(f"[{c.section}] {c.text}" for c in chunks)
        return f"(LLM would answer using:)\n{ctx}"

    def retrieve(self, query: str, k: int = 4, fetch_k: int = 50) -> List[Result]:
        candidates = self.retriever.retrieve(query, k=fetch_k)
        return self.reranker.rerank(query, candidates, k=k)

    def query(self, question: str, k: int = 4) -> str:
        if self.cache:
            cached = self.cache.get(question)
            if cached is not None:
                return f"[cache hit] {cached}"
        results = self.retrieve(question, k=k)
        answer = self.llm_fn(question, [r.chunk for r in results])
        if self.cache:
            self.cache.put(question, answer)
        return answer
