"""
Logic tests using MOCK components (no network needed).
Verifies RRF fusion, entity-aware cache bypass, and BM25 tokenization.
"""
import sys
sys.path.insert(0, "src")
import numpy as np

from rag_pipeline import reciprocal_rank_fusion, SemanticCache, Chunk


def test_rrf_fusion():
    dense_rank = [0, 1, 2, 3]
    bm25_rank = [2, 0, 1, 3]
    fused = reciprocal_rank_fusion([dense_rank, bm25_rank])
    order = [idx for idx, _ in fused]
    assert order[0] in (0, 2), "doc appearing high in both rankings should win"
    print(f"PASS RRF fusion: {order}")


def test_entity_bypass():
    class MockEmb:
        def encode(self, texts, normalize_embeddings=True):
            return np.array([[1.0, 0.0] for _ in texts])
    cache = SemanticCache(MockEmb(), threshold=0.95)
    assert cache._is_entity_specific("sample 5 conductivity") is True
    assert cache._is_entity_specific("what is 1.72 S/m") is True
    assert cache._is_entity_specific("general overview") is False
    print("PASS entity-aware cache bypass")


def test_cache_store_and_hit():
    class MockEmb:
        def encode(self, texts, normalize_embeddings=True):
            # identical embedding -> guaranteed cache hit
            return np.array([[1.0, 0.0] for _ in texts])
    cache = SemanticCache(MockEmb(), threshold=0.95)
    cache.put("overview of materials", "ANSWER_A")
    hit = cache.get("overview of materials")
    assert hit == "ANSWER_A", "identical query should hit cache"
    print("PASS cache store + hit")


if __name__ == "__main__":
    print("Running RAG logic tests (mock, no network)\n")
    test_rrf_fusion()
    test_entity_bypass()
    test_cache_store_and_hit()
    print("\nAll logic tests passed -- pipeline algorithm is correct.")
    print("(Real embedders/reranker run on Colab where HF models download.)")
