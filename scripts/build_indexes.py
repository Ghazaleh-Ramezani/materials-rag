"""Build the sparse (BM25) and dense (FAISS or pgvector) indexes from chunks.jsonl.

Usage:
    python -m scripts.build_indexes
"""

from __future__ import annotations

from src.config import config
from src.indexing.pgvector_index import get_dense_index_class
from src.indexing.sparse_index import SparseIndex
from src.ingestion.ingest_papers import load_chunks


def main() -> None:
    config.ensure_dirs()
    chunks = load_chunks()
    print(f"[build] loaded {len(chunks)} chunks")

    sparse = SparseIndex().build(chunks)
    sparse.save()
    print(f"[build] sparse BM25 index saved -> {config.index_dir / 'bm25.pkl'}")

    dense = get_dense_index_class()().build(chunks)
    dense.save()
    where = (f"pgvector collection {config.pgvector_collection!r}" if config.vector_backend == "pgvector"
             else str(config.index_dir / "faiss.index"))
    print(f"[build] dense index saved -> {where} "
          f"(embeddings={config.embedding_backend}, dim={dense.embedder.dim})")


if __name__ == "__main__":
    main()
