"""Ask the pipeline a question from the command line.

Usage:
    python -m scripts.run_query "How does rGO improve sensor sensitivity?"
    python -m scripts.run_query "..." --k 5 --reranker
"""

from __future__ import annotations

import argparse
import json

from src.pipeline import RAGPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the Materials-RAG pipeline")
    ap.add_argument("query", type=str)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--reranker", action="store_true", default=None)
    args = ap.parse_args()

    pipeline = RAGPipeline()
    answer = pipeline.answer_question(args.query, k=args.k, use_reranker=args.reranker)

    print("\n=== ANSWER ===")
    print(answer.answer_text)
    print("\n=== CONTEXTS ===")
    for c in answer.contexts:
        print(f"  [{c.chunk.id}] score={c.score:.4f} sources={c.sources}")
    print("\n=== META ===")
    print(json.dumps(answer.metadata, indent=2))


if __name__ == "__main__":
    main()
