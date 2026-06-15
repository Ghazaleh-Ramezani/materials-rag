"""End-to-end pipeline assembly: load indexes, wire cache + retriever + LLM,
expose a single ``answer_question`` entry point used by the CLI and the API."""

from __future__ import annotations

import time
from typing import Optional

from src.config import config
from src.indexing.dense_index import DenseIndex
from src.indexing.sparse_index import SparseIndex
from src.ingestion.ingest_papers import load_chunks
from src.generation.llm import LLMClient
from src.retrieval.cache import SemanticCache
from src.retrieval.hybrid import CrossEncoderReranker, HybridRetriever
from src.schemas import Answer, Chunk, RetrievedChunk


class RAGPipeline:
    def __init__(self) -> None:
        chunks = load_chunks()
        embedder = None  # let DenseIndex build/share its embedder
        self.dense = DenseIndex.load()
        self.sparse = SparseIndex.load()
        reranker = CrossEncoderReranker() if config.use_reranker else None
        self.retriever = HybridRetriever(
            sparse=self.sparse, dense=self.dense, chunks=chunks, reranker=reranker
        )
        # share the dense index's embedder with the cache to avoid loading twice
        self.cache: Optional[SemanticCache] = (
            SemanticCache(embedder=self.dense.embedder) if config.cache_enabled else None
        )
        self.llm = LLMClient()

    def answer_question(
        self, query: str, k: Optional[int] = None, use_reranker: Optional[bool] = None
    ) -> Answer:
        t0 = time.perf_counter()

        if self.cache is not None:
            cached = self.cache.get(query)
            if cached is not None:
                contexts = [
                    RetrievedChunk(
                        chunk=Chunk.from_dict(c["chunk"]),
                        score=c["score"],
                        sources=c.get("sources", []),
                    )
                    for c in cached["contexts"]
                ]
                return Answer(
                    answer_text=cached["answer_text"],
                    used_chunks=cached["used_chunks"],
                    contexts=contexts,
                    metadata={
                        **cached.get("metadata", {}),
                        "cache": cached["_cache"],
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    },
                )

        contexts = self.retriever.retrieve(query, k=k, use_reranker=use_reranker)
        answer_text, used = self.llm.generate(query, contexts)
        answer = Answer(
            answer_text=answer_text,
            used_chunks=used,
            contexts=contexts,
            metadata={
                "cache": "miss",
                "provider": self.llm.provider,
                "n_contexts": len(contexts),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        )

        if self.cache is not None:
            self.cache.set(
                query,
                {
                    "answer_text": answer.answer_text,
                    "used_chunks": answer.used_chunks,
                    "contexts": [
                        {
                            "chunk": c.chunk.to_dict(),
                            "score": c.score,
                            "sources": c.sources,
                        }
                        for c in answer.contexts
                    ],
                    "metadata": {"provider": self.llm.provider},
                },
            )
        return answer
