"""Build the sparse (BM25) and dense (FAISS) indexes from chunks.jsonl.

Usage:
    python -m scripts.build_indexes
"""

from __future__ import annotations

from src.config import config
from src.indexing.dense_index import DenseIndex
from src.indexing.sparse_index import SparseIndex
from src.ingestion.ingest_papers import load_chunks


def main() -> None:
    config.ensure_dirs()
    chunks = load_chunks()
    print(f"[build] loaded {len(chunks)} chunks")

    sparse = SparseIndex().build(chunks)
    sparse.save()
    print(f"[build] sparse BM25 index saved -> {config.index_dir / 'bm25.pkl'}")

    dense = DenseIndex().build(chunks)
    dense.save()
    print(
        f"[build] dense FAISS index saved -> {config.index_dir / 'faiss.index'} "
        f"(backend={config.embedding_backend}, dim={dense.embedder.dim})"
    )


if __name__ == "__main__":
    main()
