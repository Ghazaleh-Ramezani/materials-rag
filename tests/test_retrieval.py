"""End-to-end smoke tests over the bundled sample corpus.

These run fully offline (EMBEDDING_BACKEND=hashing, LLM_PROVIDER=mock).
"""

import os

os.environ.setdefault("EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("CACHE_ENABLED", "false")

import pytest

from src.indexing.dense_index import DenseIndex
from src.indexing.sparse_index import SparseIndex
from src.ingestion.ingest_papers import run as ingest_run, load_chunks
from src.retrieval.hybrid import HybridRetriever


@pytest.fixture(scope="module")
def retriever():
    ingest_run()  # falls back to sample_corpus
    chunks = load_chunks()
    sparse = SparseIndex().build(chunks)
    dense = DenseIndex().build(chunks)
    return HybridRetriever(sparse=sparse, dense=dense, chunks=chunks)


def test_ingestion_produces_chunks():
    n = ingest_run()
    assert n > 0


def test_retrieval_returns_k(retriever):
    results = retriever.retrieve("rGO strain sensor gauge factor", k=5, use_reranker=False)
    assert 1 <= len(results) <= 5
    # every result carries provenance
    assert all(r.sources for r in results)


def test_relevant_doc_surfaces(retriever):
    results = retriever.retrieve(
        "How does reduction degree affect gauge factor of rGO?", k=5, use_reranker=False
    )
    doc_ids = {r.chunk.doc_id for r in results}
    assert "rgo_strain_sensors" in doc_ids
