"""Entity-aware semantic cache: matching rules, namespaces, persistence."""

import json

import pytest

from src.indexing.embedders import HashingEmbedder
from src.retrieval.cache import SemanticCache, extract_entities

Q_CNF = "What is the gauge factor of rGO/CNF composite films?"


@pytest.fixture()
def cache(tmp_path):
    return SemanticCache(embedder=HashingEmbedder(), threshold=0.8, path=tmp_path / "cache.jsonl")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10 wt% CNC film", {"10wt%", "cnc"}),
        ("annealed at 25 °C", {"25°c"}),
        ("conductivity of 1.5 S/m", {"1.5s/m"}),
        ("rGO/CNF composite", {"rgo", "cnf"}),
        ("a general overview of the field", set()),
    ],
)
def test_entity_extraction(text, expected):
    assert extract_entities(text) == expected


def test_paraphrase_with_the_same_entities_hits(cache):
    cache.set(Q_CNF, {"answer": "cnf"})
    hit = cache.get("what is the gauge factor of rGO/CNF composite films")
    assert hit is not None and hit["answer"] == "cnf"


def test_shared_entity_with_a_different_material_misses(cache):
    # Shares "rgo" but asks about CNC: a different question.
    cache.set(Q_CNF, {"answer": "cnf"})
    assert cache.get("What is the gauge factor of rGO/CNC composite films?") is None


def test_entity_free_queries_need_the_fallback_flag(tmp_path):
    common = dict(embedder=HashingEmbedder(), threshold=0.8)
    off = SemanticCache(path=tmp_path / "off.jsonl", **common)
    off.set("what is a good overview of the field", {"answer": "old"})
    assert off.get("what is a good overview of this field") is None
    on = SemanticCache(path=tmp_path / "on.jsonl", unconstrained_fallback=True, **common)
    on.set("what is a good overview of the field", {"answer": "old"})
    assert on.get("what is a good overview of this field")["_cache"] == "semantic"


def test_namespaces_keep_retrieval_settings_apart(cache):
    cache.set(Q_CNF, {"answer": "k1"}, namespace="k=1|rerank=0")
    assert cache.get(Q_CNF, namespace="k=3|rerank=0") is None
    assert cache.get(Q_CNF, namespace="k=1|rerank=0")["_cache"] == "exact"


def test_reload_rebuilds_identical_embeddings(cache, tmp_path):
    cache.set(Q_CNF, {"answer": "cnf"})
    query = "what is the gauge factor of rGO/CNF composite films"
    before = cache.get(query)
    reloaded = SemanticCache(embedder=HashingEmbedder(), threshold=0.8, path=tmp_path / "cache.jsonl")
    assert len(reloaded) == 1
    assert (reloaded._matrix == cache._matrix).all()
    assert reloaded.get(query) == before


def test_rows_without_query_text_are_skipped(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps({"key": "old##entities=rgo", "entities": ["rgo"], "payload": {}}) + "\n")
    assert len(SemanticCache(embedder=HashingEmbedder(), path=path)) == 0
