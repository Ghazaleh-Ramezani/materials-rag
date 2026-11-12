"""pgvector backend: selection, safety checks, and parity with FAISS against a live PostgreSQL.

The live tests run when PGVECTOR_TEST_DSN points at a PostgreSQL with the vector
extension (CI starts pgvector/pgvector:pg16); without it they are skipped.
"""

import json
import os

import numpy as np
import pytest

from src.config import config
from src.indexing.dense_index import DenseIndex
from src.indexing.embedders import HashingEmbedder
from src.indexing.pgvector_index import PgVectorIndex, _table, get_dense_index_class
from src.indexing.sparse_index import SparseIndex
from src.ingestion.ingest_papers import load_chunks
from src.ingestion.ingest_papers import run as ingest_run
from src.retrieval.hybrid import HybridRetriever

DSN = os.environ.get("PGVECTOR_TEST_DSN")
live = pytest.mark.skipif(not DSN, reason="set PGVECTOR_TEST_DSN to a PostgreSQL with pgvector")


@pytest.mark.parametrize("backend, cls", [("faiss", DenseIndex), ("pgvector", PgVectorIndex)])
def test_backend_selection(monkeypatch, backend, cls):
    monkeypatch.setattr(config, "vector_backend", backend)
    assert get_dense_index_class() is cls


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "vector_backend", "chroma")
    with pytest.raises(ValueError, match="VECTOR_BACKEND"):
        get_dense_index_class()


def test_collection_names_cannot_inject_sql():
    assert _table("materials_v2") == "rag_vectors_materials_v2"
    with pytest.raises(ValueError):
        _table("x; DROP TABLE rag_collections")


def test_search_before_build_fails_clearly():
    with pytest.raises(RuntimeError, match="not built"):
        PgVectorIndex(embedder=HashingEmbedder(), dsn="postgresql://unused").search_dense("q", 3)


@pytest.fixture(scope="module")
def chunks():
    ingest_run()
    return load_chunks()


@pytest.fixture(scope="module")
def questions():
    with config.qa_path.open(encoding="utf-8") as fh:
        return [json.loads(line)["question"] for line in fh if line.strip()]


@live
def test_pgvector_matches_faiss_on_every_benchmark_question(chunks, questions):
    emb = HashingEmbedder()
    faiss_idx = DenseIndex(embedder=emb).build(chunks)
    pg_idx = PgVectorIndex(embedder=emb, dsn=DSN, collection="parity").build(chunks)
    for q in questions:
        a, b = faiss_idx.search_dense(q, 5), pg_idx.search_dense(q, 5)
        np.testing.assert_allclose([s for _, s in a], [s for _, s in b], atol=1e-5, err_msg=q)
        # identical scores may come back in either order; the sets per score level must agree
        by_score = lambda hits: {round(s, 5): set() for _, s in hits}  # noqa: E731
        sa, sb = by_score(a), by_score(b)
        for cid, s in a:
            sa[round(s, 5)].add(cid)
        for cid, s in b:
            sb[round(s, 5)].add(cid)
        assert sa == sb, q


@live
def test_hybrid_retrieval_is_unchanged_by_the_backend(chunks, questions):
    emb = HashingEmbedder()
    sparse = SparseIndex().build(chunks)
    with_faiss = HybridRetriever(sparse=sparse, dense=DenseIndex(embedder=emb).build(chunks), chunks=chunks)
    with_pg = HybridRetriever(sparse=sparse, chunks=chunks,
                              dense=PgVectorIndex(embedder=emb, dsn=DSN, collection="hybrid").build(chunks))
    for q in questions:
        docs = lambda r: [c.chunk.doc_id for c in r.retrieve(q, k=3, use_reranker=False)]  # noqa: E731
        assert docs(with_faiss) == docs(with_pg), q


@live
def test_load_checks_the_embedder_and_the_collection(chunks):
    PgVectorIndex(embedder=HashingEmbedder(384), dsn=DSN, collection="checked").build(chunks)
    loaded = PgVectorIndex.load(embedder=HashingEmbedder(384), dsn=DSN, collection="checked")
    assert loaded.search_dense("gauge factor of rGO", 2)
    with pytest.raises(ValueError, match="built with embedder"):
        PgVectorIndex.load(embedder=HashingEmbedder(256), dsn=DSN, collection="checked")
    with pytest.raises(ValueError, match="does not exist"):
        PgVectorIndex.load(embedder=HashingEmbedder(384), dsn=DSN, collection="never_built")


@live
def test_rebuild_replaces_the_collection(chunks):
    idx = PgVectorIndex(embedder=HashingEmbedder(), dsn=DSN, collection="rebuilt")
    idx.build(chunks)
    idx.build(chunks[:2])
    assert len(idx.search_dense("graphene", 10)) == 2
