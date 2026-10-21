# Materials-RAG — Hybrid RAG over Scientific Literature

An end-to-end Retrieval-Augmented Generation pipeline over a corpus of materials-science
papers. It ingests and cleans PDFs into chunked, metadata-rich documents, indexes them in
**both** a BM25 sparse index and a FAISS dense index, fuses the two with **Reciprocal Rank
Fusion**, optionally **reranks** with a cross-encoder, **caches** semantically similar
queries with entity-aware matching, and generates cited answers behind a **FastAPI** `/qa`
endpoint. Ships with an **evaluation harness** (Recall@k, Hit@k, MRR) and is
**containerized with Docker**.

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
| Cache        | exact + entity-matched nearest-neighbor, JSONL-persisted        |
| Generation   | Anthropic / OpenAI SDK, or offline mock                         |
| Serving      | FastAPI + Uvicorn                                               |
| Eval         | Recall@k, Hit@k, MRR (+ optional LLM faithfulness)              |
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
  retrieval/cache.py           entity-aware semantic cache (exact + nearest-neighbor)
  generation/llm.py            RAG prompt + anthropic/openai/mock providers
  pipeline.py                  cache → retrieve → rerank → generate
  api/main.py                  FastAPI /qa and /health
  evaluation/eval_retrieval.py Recall@k, Hit@k, MRR
scripts/                       build_indexes.py, run_query.py
tests/                         RRF, cache, metrics, index, pipeline and API tests
sample_corpus/                 4 synthetic materials-science docs so it runs offline
data/qa_benchmark/qa.jsonl     15 questions: 12 single-document, 3 cross-document
```

## Retrieval results on the sample corpus

`python -m src.evaluation.eval_retrieval --k K` with the offline hashing embedder,
no reranker, 15 questions over 4 documents (5 chunks):

| k | Recall@k | Hit@k | MRR |
|---|:---:|:---:|:---:|
| 1 | 0.867 | 0.933 | 0.967 |
| 3 | 0.967 | 1.000 | 0.967 |
| 5 | 1.000 | 1.000 | 0.967 |

Recall@k is the share of a question's relevant documents found in the top k; Hit@k is
the share of questions with at least one. They separate on the three cross-document
questions, which need both of their documents: at k=1 Recall@k is capped at 0.5 for
each of them, and at k=3 one still has a single document of its two. On a corpus this
small the numbers confirm the harness end to end; retrieval quality is measured on a
real corpus of 30–50+ papers with overlapping topics, which is the scale the harness
is built for.

## Design notes

- **Offline by default.** The hashing embedder and mock generator keep the whole
  pipeline runnable in CI and demos with no downloads or keys. Retrieval quality comes
  from `EMBEDDING_BACKEND=sentence-transformers` plus the cross-encoder reranker.
- **The dense index records its embedder.** `faiss.meta.json` stores the embedder that
  built the index, and loading checks it. The hashing embedder and all-MiniLM-L6-v2 are
  both 384-dimensional, so a mismatch would otherwise load cleanly and return neighbours
  from the wrong vector space; instead it raises and asks for a rebuild.
- **Entity-aware cache.** A semantic hit requires the same set of materials, formulas
  and number-unit pairs ("10 wt%", "25 °C") as the cached question. "gauge factor of
  rGO/CNF films" and "gauge factor of rGO/CNC films" share rGO and are still different
  questions, so they never share an answer. Queries with no entities get exact matches
  only unless the unconstrained fallback is switched on.
- **Cache entries are scoped to retrieval settings.** The cache key includes `k` and
  whether the reranker ran, so a request for 3 contexts is never served a cached
  1-context answer.
- **Cache persistence is exact.** Each JSONL row stores the normalized query, and the
  same text is embedded when the entry is written and when the file is reloaded, so
  similarities are identical across restarts.
- **Tokens are approximated by words** in chunking (~1.3 tokens/word), keeping the
  ingestion path free of a tokenizer dependency.
- **Faithfulness** (`--faithfulness`): the configured LLM judges whether each generated
  answer is supported by its retrieved passages. Only an exact SUPPORTED / UNSUPPORTED
  verdict counts, and anything else is reported as an invalid verdict. Retrieval metrics
  run without any key.
- PDF cleaning is heuristic (drops reference lists, lone page numbers, runaway
  whitespace) and is the place to add per-source rules for messy PDFs.

## Tests

```bash
pytest -q        # ~5 s, offline: RRF, cache rules, metrics, judge parsing, index check, pipeline, API
```
