"""
Retrieval evaluation — Recall@k and MRR.

Evaluating the *retriever itself*, independent of the final LLM answer.
We hand-build a small test set: each question is paired with the chunk_id(s)
that actually contain the answer (ground truth). Then we measure:

  - Recall@k : fraction of questions whose ground-truth chunk appears in top-k
  - MRR      : mean of 1/rank of the first relevant chunk (rank quality, not
               just presence — a relevant chunk at rank 1 is worth more than
               the same chunk at rank 4, because the LLM attends to it more)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Set

from rag_pipeline import Result


@dataclass
class EvalExample:
    question: str
    relevant_chunk_ids: Set[int]


def recall_at_k(
    examples: List[EvalExample],
    retrieve_fn: Callable[[str, int], List[Result]],
    k: int = 4,
) -> float:
    hits = 0
    for ex in examples:
        retrieved_ids = {r.chunk.chunk_id for r in retrieve_fn(ex.question, k)}
        if retrieved_ids & ex.relevant_chunk_ids:
            hits += 1
    return hits / len(examples)


def mrr(
    examples: List[EvalExample],
    retrieve_fn: Callable[[str, int], List[Result]],
    k: int = 10,
) -> float:
    total = 0.0
    for ex in examples:
        results = retrieve_fn(ex.question, k)
        rr = 0.0
        for rank, r in enumerate(results, start=1):
            if r.chunk.chunk_id in ex.relevant_chunk_ids:
                rr = 1.0 / rank
                break
        total += rr
    return total / len(examples)


def evaluate(
    examples: List[EvalExample],
    retrieve_fn: Callable[[str, int], List[Result]],
    ks: List[int] = (1, 3, 4, 10),
) -> dict:
    metrics = {f"recall@{k}": recall_at_k(examples, retrieve_fn, k) for k in ks}
    metrics["mrr"] = mrr(examples, retrieve_fn, k=10)
    return metrics


def compare_configs(examples, configs: dict) -> None:
    """configs: {name: retrieve_fn}. Prints a comparison table."""
    print(f"{'Config':<28} {'R@1':>6} {'R@3':>6} {'R@4':>6} {'MRR':>6}")
    print("-" * 56)
    for name, fn in configs.items():
        m = evaluate(examples, fn)
        print(f"{name:<28} {m['recall@1']:>6.2f} {m['recall@3']:>6.2f} "
              f"{m['recall@4']:>6.2f} {m['mrr']:>6.2f}")
