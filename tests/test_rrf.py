"""Tests for Reciprocal Rank Fusion."""

from src.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    # 'b' appears high in both lists; it should win.
    sparse = [("a", 9.0), ("b", 8.0), ("c", 1.0)]
    dense = [("b", 0.9), ("d", 0.8), ("a", 0.1)]
    fused = reciprocal_rank_fusion({"sparse": sparse, "dense": dense}, k_constant=60)
    ids = [cid for cid, _, _ in fused]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_sources_tracked():
    sparse = [("a", 1.0)]
    dense = [("a", 1.0), ("b", 1.0)]
    fused = reciprocal_rank_fusion({"sparse": sparse, "dense": dense})
    by_id = {cid: srcs for cid, _, srcs in fused}
    assert set(by_id["a"]) == {"sparse", "dense"}
    assert by_id["b"] == ["dense"]


def test_rrf_scores_decreasing():
    sparse = [("a", 1.0), ("b", 1.0), ("c", 1.0)]
    fused = reciprocal_rank_fusion({"sparse": sparse})
    scores = [s for _, s, _ in fused]
    assert scores == sorted(scores, reverse=True)
