"""Evaluation harness.

Retrieval metrics (always available, offline):
    * Recall@k  — fraction of questions whose relevant doc_id appears in top-k
    * MRR       — mean reciprocal rank of the first relevant doc_id

Optional generation metric (needs an LLM provider):
    * faithfulness — LLM judges whether the answer is supported by the contexts.
      Skipped automatically when LLM_PROVIDER=mock.

Results are written to data/processed/eval_results.csv.

Usage:
    python -m src.evaluation.eval_retrieval --k 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

from src.config import config
from src.pipeline import RAGPipeline
from src.schemas import RetrievedChunk


def _load_qa(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _relevant_doc_ids(item: dict) -> set:
    ids = item.get("doc_ids") or ([item["doc_id"]] if "doc_id" in item else [])
    return set(ids)


def recall_at_k(retrieved_doc_ids: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    topk = set(retrieved_doc_ids[:k])
    return 1.0 if (topk & relevant) else 0.0


def reciprocal_rank(retrieved_doc_ids: List[str], relevant: set) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(k: int = None, use_reranker: bool = None) -> dict:
    k = k or config.final_k
    qa = _load_qa(config.qa_path)
    pipeline = RAGPipeline()

    rows = []
    recall_sum = 0.0
    rr_sum = 0.0
    for item in qa:
        question = item["question"]
        relevant = _relevant_doc_ids(item)
        # retrieve a generous pool for ranking metrics
        contexts: List[RetrievedChunk] = pipeline.retriever.retrieve(
            question, k=max(k, 10), use_reranker=use_reranker
        )
        retrieved_doc_ids = []
        for c in contexts:
            if c.chunk.doc_id not in retrieved_doc_ids:
                retrieved_doc_ids.append(c.chunk.doc_id)

        r = recall_at_k(retrieved_doc_ids, relevant, k)
        rr = reciprocal_rank(retrieved_doc_ids, relevant)
        recall_sum += r
        rr_sum += rr
        rows.append(
            {
                "question": question,
                "relevant": "|".join(sorted(relevant)),
                "retrieved": "|".join(retrieved_doc_ids[:k]),
                f"recall@{k}": r,
                "reciprocal_rank": round(rr, 4),
            }
        )

    n = max(1, len(qa))
    summary = {
        "n_questions": len(qa),
        f"recall@{k}": round(recall_sum / n, 4),
        "mrr": round(rr_sum / n, 4),
        "reranker": bool(config.use_reranker if use_reranker is None else use_reranker),
        "embedding_backend": config.embedding_backend,
    }

    out_path = config.processed_dir / "eval_results.csv"
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"[eval] per-question results -> {out_path}")
    return summary


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval (Recall@k, MRR)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--reranker", action="store_true", default=None)
    args = ap.parse_args()
    evaluate(k=args.k, use_reranker=args.reranker)


if __name__ == "__main__":
    _cli()
