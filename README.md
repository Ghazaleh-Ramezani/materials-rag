# Materials-RAG — Hybrid RAG over Scientific Literature

An end-to-end Retrieval-Augmented Generation pipeline over a corpus of materials-science
papers. It ingests and cleans PDFs into chunked, metadata-rich documents, indexes them in
**both** a BM25 sparse index and a FAISS dense index, fuses the two with **Reciprocal Rank
Fusion**, optionally **reranks** with a cross-encoder, **caches** semantically similar
queries, and generates cited answers behind a **FastAPI** `/qa` endpoint. Ships with an
**evaluation harness** (Recall@k, MRR) and is **containerized with Docker**.

The repo runs fully **offline and key-free out of the box** (a deterministic hashing
embedder + a mock extractive generator), so a reviewer can clone and get answers in one
command — then flip two env vars to use real `sentence-transformers` embeddings and a live
LLM.

---

## Architecture

```
                                  ┌─────────────────────────────┐
  data/raw/*.pdf|*.txt            │        Ingestion            │
  (or bundled sample_corpus) ───▶ │  extract → clean → chunk    │ ──▶ data/processed/chunks.jsonl
                                  │  + metadata (title, year…)  │
                                  └─────────────────────────────┘
                                                │
                  ┌─────────────────────────────┴─────────────────────────────┐
                  ▼                                                             ▼
        ┌───────────────────┐                                       ┌────────────────────┐
        │  Sparse index     │                                       │   Dense index      │
        │  BM25 (rank-bm25) │                                       │  FAISS (cosine)    │
        └───────────────────┘                                       │  sentence-transf.  │
                  │                                                  └────────────────────┘
                  │  top-k                                                 │  top-k
                  └───────────────────────┐         ┌──────────────────────┘
                                          ▼         ▼
                               ┌────────────────────────────┐
                               │  Reciprocal Rank Fusion     │
                               └────────────────────────────┘
                                          │ fused top-N
                                          ▼
                               ┌────────────────────────────┐
                               │  Cross-encoder reranker     │  (optional)
                               └────────────────────────────┘
                                          │ final-k contexts
                                          ▼
   query ──▶ Semantic cache ──(miss)──▶  LLM generation (Anthropic / OpenAI / mock)
              │  (hit)                          │  cited answer
              └──────────────▶ Answer ◀─────────┘
                                   ▲
                          FastAPI  POST /qa
```

## Stack

| Layer        | Tooling                                                          |
|--------------|-----------------------------------------------------------------|
| Ingestion    | `pypdf`, regex cleaning, word-window chunking                   |
| Sparse index | `rank-bm25` (BM25Okapi)                                          |
| Dense index  | `faiss-cpu` (IndexFlatIP / cosine), `sentence-transformers`     |
| Fusion       | Reciprocal Rank Fusion (own implementation)                     |
| Rerank       | `sentence-transformers` CrossEncoder (optional)                 |
| Cache        | exact + embedding nearest-neighbor, JSONL-persisted             |
| Generation   | Anthropic / OpenAI SDK, or offline mock                         |
| Serving      | FastAPI + Uvicorn                                               |
| Eval         | Recall@k, MRR (+ optional LLM faithfulness)                     |
| Packaging    | Docker, `pyproject.toml`                                         |

## Quickstart (offline, no keys)

```bash
pip install -r requirements.txt

python -m src.ingestion.ingest_papers      # → data/processed/chunks.jsonl (uses sample_corpus)
python -m scripts.build_indexes            # → BM25 + FAISS indexes
python -m scripts.run_query "How does rGO reduction degree affect the gauge factor?"
python -m src.evaluation.eval_retrieval --k 3
```

Serve the API:

```bash
uvicorn src.api.main:app --reload --port 8000
curl -X POST localhost:8000/qa -H "Content-Type: application/json" \
     -d '{"query": "What acquisition function is used for multi-objective BO?", "k": 3}'
```

Or with Docker:

```bash
docker build -t materials-rag . && docker run -p 8000:8000 materials-rag
```

A `Makefile` wraps these (`make build`, `make query`, `make api`, `make eval`, `make test`).

## Using it for real

1. Drop your own `*.pdf` (or `*.txt`) into `data/raw/`. Optionally add a sidecar
   `paper.meta.json` (`{"title": ..., "year": ..., "journal": ...}`) for richer metadata.
2. Switch on real models / generation in `.env` (copy from `.env.example`):
   ```
   EMBEDDING_BACKEND=sentence-transformers
   USE_RERANKER=true
   LLM_PROVIDER=anthropic        # or openai
   ANTHROPIC_API_KEY=...
   ```
3. Re-run `make build` and query as above.

## Project layout

```
src/
  ingestion/ingest_papers.py   extract → clean → chunk → chunks.jsonl
  indexing/sparse_index.py     BM25 build / search / persist
  indexing/dense_index.py      FAISS build / search / persist
  indexing/embedders.py        sentence-transformers + offline hashing fallback
  retrieval/hybrid.py          RRF fusion + cross-encoder reranker
  retrieval/cache.py           semantic cache (exact + nearest-neighbor)
  generation/llm.py            RAG prompt + anthropic/openai/mock providers
  pipeline.py                  cache → retrieve → rerank → generate
  api/main.py                  FastAPI /qa and /health
  evaluation/eval_retrieval.py Recall@k, MRR
scripts/                       build_indexes.py, run_query.py
tests/                         RRF + end-to-end retrieval tests
sample_corpus/                 4 synthetic materials-science docs so it runs offline
data/qa_benchmark/qa.jsonl     12-question evaluation set
```

## Design notes & honest limitations

- **Offline-by-default is deliberate.** The hashing embedder and mock generator exist so the
  pipeline is always runnable in CI and demos; they are *not* semantically strong. Real
  retrieval quality comes from `EMBEDDING_BACKEND=sentence-transformers` + a reranker.
- **Tokens are approximated by words** in chunking (~1.3 tokens/word) to avoid a tokenizer
  dependency in the ingestion path.
- **The bundled corpus is tiny (4 docs / 12 questions).** On it, Recall@k and MRR are near
  1.0 — expected and not impressive on its own. The metrics become meaningful on a real
  corpus of 30–50+ papers with overlapping topics; the harness is built to scale to that.
- **Faithfulness / RAGAS-style scoring** is wired as optional and requires an LLM provider;
  retrieval metrics run without any key.
- PDF cleaning is heuristic (drops reference lists, lone page numbers, runaway whitespace);
  messy real-world PDFs will need per-source tuning.

## Tests

```bash
pytest -q        # RRF fusion logic + offline end-to-end retrieval over sample_corpus
```
