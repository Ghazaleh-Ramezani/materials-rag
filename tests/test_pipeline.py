"""Index persistence, the cache inside the pipeline, and the HTTP API, all on
the bundled sample corpus with every path redirected to a temp directory."""

import pytest
from fastapi.testclient import TestClient

from src.config import config
from src.indexing.dense_index import DenseIndex
from src.indexing.embedders import HashingEmbedder
from src.indexing.sparse_index import SparseIndex
from src.ingestion.ingest_papers import load_chunks
from src.ingestion.ingest_papers import run as ingest_run

QUESTION = "How does reduction degree affect the gauge factor of rGO?"


@pytest.fixture()
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "processed_dir", tmp_path / "processed")
    monkeypatch.setattr(config, "index_dir", tmp_path / "index")
    monkeypatch.setattr(config, "chunks_path", tmp_path / "processed" / "chunks.jsonl")
    monkeypatch.setattr(config, "cache_path", tmp_path / "cache.jsonl")
    monkeypatch.setattr(config, "raw_dir", tmp_path / "raw")  # empty: falls back to sample_corpus
    monkeypatch.setattr(config, "cache_enabled", True)
    ingest_run()
    chunks = load_chunks()
    SparseIndex().build(chunks).save()
    DenseIndex(embedder=HashingEmbedder()).build(chunks).save()
    return tmp_path


class _OtherEmbedder(HashingEmbedder):
    """Same dimension as the default hashing embedder, different vector space."""

    def __init__(self):
        super().__init__()
        self.name = "sentence-transformers:some-model"


def test_index_refuses_a_different_embedder(built):
    DenseIndex.load(embedder=HashingEmbedder())
    with pytest.raises(ValueError, match="was built with embedder"):
        DenseIndex.load(embedder=_OtherEmbedder())


def test_cache_keeps_k_apart(built):
    from src.pipeline import RAGPipeline

    pipe = RAGPipeline()
    one = pipe.answer_question(QUESTION, k=1)
    three = pipe.answer_question(QUESTION, k=3)
    again = pipe.answer_question(QUESTION, k=3)
    assert (len(one.contexts), one.metadata["cache"]) == (1, "miss")
    assert (len(three.contexts), three.metadata["cache"]) == (3, "miss")
    assert (len(again.contexts), again.metadata["cache"]) == (3, "exact")


def test_api_health_and_qa(built):
    from src.api import main

    main.get_pipeline.cache_clear()
    client = TestClient(main.app)
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/qa", json={"query": QUESTION, "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["contexts"]) == 2
    assert "rgo_strain_sensors" in {c["doc_id"] for c in body["contexts"]}
    main.get_pipeline.cache_clear()


def test_evaluation_with_faithfulness_judge(built, monkeypatch):
    from src.evaluation import eval_retrieval
    from src.generation.llm import LLMClient

    # A stand-in provider: answers and verdicts come from this function, so
    # the judged path runs without a key or network.
    def fake_complete(self, system, prompt):
        return "SUPPORTED" if system == eval_retrieval.FAITHFULNESS_SYSTEM else "An answer."

    monkeypatch.setattr(config, "llm_provider", "anthropic")
    monkeypatch.setattr(LLMClient, "complete", fake_complete)
    summary = eval_retrieval.evaluate(k=3, faithfulness=True)
    assert summary["n_questions"] == 15
    assert summary["faithfulness"] == 1.0
    assert summary["faithfulness_invalid_verdicts"] == 0
    assert summary["hit@3"] == 1.0
