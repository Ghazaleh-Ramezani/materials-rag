"""Evaluation harness.

Retrieval metrics (always available, offline), over the ranked list of
distinct documents behind the retrieved chunks:
    * Recall@k  — share of a question's relevant documents found in the top k,
                  averaged over questions
    * Hit@k     — share of questions with at least one relevant document in
                  the top k
    * MRR       — mean reciprocal rank of the first relevant document
With one relevant document per question Recall@k and Hit@k coincide; they
separate as soon as a question has several.

Optional generation metric (--faithfulness; needs LLM_PROVIDER=anthropic or
openai, skipped with the offline mock generator):
    * faithfulness — the configured LLM judges whether every claim in the
      generated answer is supported by the retrieved passages. Only an exact
      SUPPORTED / UNSUPPORTED first word counts; anything else is recorded as
      an invalid verdict rather than guessed.

Results are written to data/processed/eval_results.csv.

Usage:
    python -m src.evaluation.eval_retrieval --k 5
    python -m src.evaluation.eval_retrieval --k 5 --faithfulness
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Callable, List, Optional

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
    return len(set(retrieved_doc_ids[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved_doc_ids: List[str], relevant: set, k: int) -> float:
    return 1.0 if set(retrieved_doc_ids[:k]) & relevant else 0.0


FAITHFULNESS_SYSTEM = (
    "You check answers against source passages. Reply with exactly one word: "
    "SUPPORTED if every factual claim in the answer is stated in or directly "
    "implied by the passages, otherwise UNSUPPORTED."
)


def judge_faithfulness(
    question: str, answer: str, contexts: List[RetrievedChunk], complete: Callable[[str, str], str]
) -> Optional[float]:
    """1.0 if the judge says SUPPORTED, 0.0 if UNSUPPORTED, None otherwise."""
    passages = "\n\n".join(f"[{c.chunk.id}] {c.chunk.text}" for c in contexts)
    prompt = f"Passages:\n{passages}\n\nQuestion: {question}\n\nAnswer:\n{answer}"
    match = re.match(r"\s*([A-Za-z]+)", complete(FAITHFULNESS_SYSTEM, prompt) or "")
    verdict = match.group(1).upper() if match else ""
    return {"SUPPORTED": 1.0, "UNSUPPORTED": 0.0}.get(verdict)


def reciprocal_rank(retrieved_doc_ids: List[str], relevant: set) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(k: int = None, use_reranker: bool = None, faithfulness: bool = False) -> dict:
    k = k or config.final_k
    qa = _load_qa(config.qa_path)
    pipeline = RAGPipeline()

    judge = faithfulness and pipeline.llm.provider != "mock"
    if faithfulness and not judge:
        print("[eval] faithfulness needs LLM_PROVIDER=anthropic or openai; skipping it.")

    rows = []
    recall_sum = 0.0
    hit_sum = 0.0
    rr_sum = 0.0
    faith_scores: List[float] = []
    faith_invalid = 0
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
        h = hit_at_k(retrieved_doc_ids, relevant, k)
        rr = reciprocal_rank(retrieved_doc_ids, relevant)
        recall_sum += r
        hit_sum += h
        rr_sum += rr
        row = {
            "question": question,
            "relevant": "|".join(sorted(relevant)),
            "retrieved": "|".join(retrieved_doc_ids[:k]),
            f"recall@{k}": round(r, 4),
            f"hit@{k}": h,
            "reciprocal_rank": round(rr, 4),
        }
        if judge:
            answer = pipeline.answer_question(question, k=k, use_reranker=use_reranker)
            score = judge_faithfulness(
                question, answer.answer_text, answer.contexts, pipeline.llm.complete
            )
            row["faithful"] = "" if score is None else score
            if score is None:
                faith_invalid += 1
            else:
                faith_scores.append(score)
        rows.append(row)

    n = max(1, len(qa))
    summary = {
        "n_questions": len(qa),
        f"recall@{k}": round(recall_sum / n, 4),
        f"hit@{k}": round(hit_sum / n, 4),
        "mrr": round(rr_sum / n, 4),
        "reranker": bool(config.use_reranker if use_reranker is None else use_reranker),
        "embedding_backend": config.embedding_backend,
    }
    if judge:
        summary["faithfulness"] = (
            round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None
        )
        summary["faithfulness_invalid_verdicts"] = faith_invalid

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
    ap = argparse.ArgumentParser(description="Evaluate retrieval (Recall@k, Hit@k, MRR)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--reranker", action="store_true", default=None)
    ap.add_argument("--faithfulness", action="store_true", help="LLM-judged answer faithfulness")
    args = ap.parse_args()
    evaluate(k=args.k, use_reranker=args.reranker, faithfulness=args.faithfulness)


if __name__ == "__main__":
    _cli()
