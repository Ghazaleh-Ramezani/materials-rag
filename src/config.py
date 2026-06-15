"""Central configuration.

Everything is read from environment variables (see .env.example) with defaults
that let the whole pipeline run offline and without API keys. Swap the defaults
via env to use real sentence-transformer models, a cross-encoder reranker, and a
live LLM provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # ---- paths ----
    raw_dir: Path = ROOT / "data" / "raw"
    sample_corpus_dir: Path = ROOT / "sample_corpus"
    processed_dir: Path = ROOT / "data" / "processed"
    index_dir: Path = ROOT / "data" / "index"
    chunks_path: Path = ROOT / "data" / "processed" / "chunks.jsonl"
    qa_path: Path = ROOT / "data" / "qa_benchmark" / "qa.jsonl"

    # ---- chunking ----
    chunk_size_words: int = _env_int("CHUNK_SIZE_WORDS", 220)  # ~300 tokens
    chunk_overlap_words: int = _env_int("CHUNK_OVERLAP_WORDS", 40)

    # ---- embeddings ----
    # "sentence-transformers" (real, needs download) or "hashing" (offline, deterministic)
    embedding_backend: str = _env("EMBEDDING_BACKEND", "sentence-transformers")
    embedding_model: str = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = _env_int("EMBEDDING_DIM", 384)  # used by hashing fallback

    # ---- retrieval ----
    sparse_k: int = _env_int("SPARSE_K", 20)
    dense_k: int = _env_int("DENSE_K", 20)
    rrf_k_constant: int = _env_int("RRF_K_CONSTANT", 60)
    final_k: int = _env_int("FINAL_K", 5)

    # ---- reranker ----
    use_reranker: bool = _env_bool("USE_RERANKER", False)
    reranker_model: str = _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidates: int = _env_int("RERANK_CANDIDATES", 25)

    # ---- cache ----
    cache_enabled: bool = _env_bool("CACHE_ENABLED", True)
    cache_similarity_threshold: float = _env_float("CACHE_SIM_THRESHOLD", 0.97)
    cache_path: Path = ROOT / "data" / "cache.jsonl"

    # ---- generation ----
    # "anthropic", "openai", or "mock" (extractive, offline)
    llm_provider: str = _env("LLM_PROVIDER", "mock")
    anthropic_model: str = _env("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    openai_model: str = _env("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens: int = _env_int("LLM_MAX_TOKENS", 700)

    def ensure_dirs(self) -> None:
        for p in (self.processed_dir, self.index_dir, self.qa_path.parent):
            p.mkdir(parents=True, exist_ok=True)


config = Config()
