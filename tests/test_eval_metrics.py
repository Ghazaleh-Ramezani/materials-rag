"""Retrieval metrics on hand-built rankings."""

import pytest

from src.evaluation.eval_retrieval import (
    hit_at_k,
    judge_faithfulness,
    recall_at_k,
    reciprocal_rank,
)
from src.schemas import Chunk, RetrievedChunk


def test_recall_counts_every_relevant_document():
    ranked = ["a", "x", "y", "b"]
    assert recall_at_k(ranked, {"a", "b"}, k=3) == 0.5
    assert recall_at_k(ranked, {"a", "b"}, k=4) == 1.0


def test_hit_needs_only_one_relevant_document():
    assert hit_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 1.0
    assert hit_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_reciprocal_rank_uses_the_first_relevant_document():
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("SUPPORTED", 1.0),
        ("  unsupported.", 0.0),
        ("Supported, mostly", 1.0),
        ("The answer is supported", None),  # first word is not a verdict
        ("", None),
    ],
)
def test_faithfulness_verdict_parsing(reply, expected):
    ctx = [RetrievedChunk(chunk=Chunk("d::0000", "d", 0, "GO raises strength."), score=1.0)]
    seen = {}

    def fake_complete(system, prompt):
        seen["prompt"] = prompt
        return reply

    assert judge_faithfulness("q?", "GO raises strength.", ctx, fake_complete) == expected
    assert "[d::0000] GO raises strength." in seen["prompt"]
