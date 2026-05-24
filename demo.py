"""
Scientific RAG — end-to-end demo.

Usage (Colab or local with internet):
    pip install -r requirements.txt
    # put your PDFs in ./data/
    python demo.py

This downloads the embedding + reranker models from HuggingFace on first run,
loads your PDFs, indexes them, and answers a sample question — printing the
retrieved chunks with their source paper and section.
"""

import sys
sys.path.insert(0, "src")

from pdf_loader import load_pdf_directory
from rag_pipeline import ScientificRAG
from evaluation import EvalExample, evaluate, compare_configs


def main():
    print("=" * 60)
    print("Scientific RAG — Hybrid Retrieval + Cross-Encoder Re-ranking")
    print("=" * 60)

    # 1. Load PDFs from ./data/
    print("\n[1] Loading PDFs from ./data/ ...")
    docs = load_pdf_directory("data")
    if not docs:
        print("    No PDFs found in ./data/. Add your papers and re-run.")
        return
    print(f"    Loaded {len(docs)} section-blocks across the corpus.")

    # 2. Build the pipeline
    #    For a SCIENTIFIC corpus, swap embed_model -> 'allenai/scibert_scivocab_uncased'
    print("\n[2] Building RAG pipeline (this downloads models on first run)...")
    rag = ScientificRAG(
        embed_model="all-MiniLM-L6-v2",          # or scibert for materials terms
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alpha=0.5,                                # 0.5 = equal BM25 + dense weight
        use_reranker=True,
    )
    n_chunks = rag.index_documents(docs)
    print(f"    Indexed {n_chunks} chunks.")

    # 3. Ask a question
    question = "Which reducing agent gave the highest electrical conductivity?"
    print(f"\n[3] Query: {question}\n")
    results = rag.query(question, k=4, fetch_k=50)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.3f}] ({r.chunk.source} / {r.chunk.section})")
        print(f"     {r.chunk.text[:120]}...\n")

    # 4. (Optional) evaluate retrieval quality with a hand-built test set
    #    Fill in relevant_chunk_ids after inspecting your indexed chunks.
    print("[4] To evaluate: build EvalExample test cases with known answer chunks,")
    print("    then call evaluate(...) — see src/evaluation.py.")


if __name__ == "__main__":
    main()
