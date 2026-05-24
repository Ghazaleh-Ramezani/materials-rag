# Materials Science RAG Pipeline

**Author:** Ghazaleh Ramezani, Ph.D. | Concordia University
**Corpus:** Ramezani et al., *Micromachines* 16(4):393, 2025 (+ related work)

A production-grade Retrieval-Augmented Generation (RAG) pipeline over scientific
publications, implementing advanced retrieval techniques from the ML systems
literature.

## Architecture

```
User query
    │
    ▼
[Semantic Cache] ──cache hit──▶ cached answer
    │ miss
    ▼
[Hybrid Retrieval]
    ├── BM25 (lexical — exact match, numbers, terminology)
    ├── Dense bi-encoder (all-MiniLM-L6-v2, semantic similarity)
    └── RRF fusion → top-50 candidates
    │
    ▼
[Cross-encoder Re-ranking]
    └── cross-encoder/ms-marco-MiniLM-L-6-v2
    └── top-50 → top-4 (query+chunk joint attention)
    │
    ▼
[LLM Generation]  (OpenAI GPT-4 / local model)
    │
    ▼
Answer + cache store (entity-aware bypass)
```

## Key Design Decisions

### Why Hybrid (BM25 + Dense)?
BM25 alone fails on **vocabulary mismatch** — "reducing agent" vs "reductant" scores zero in BM25.
Dense embeddings alone fail on **exact values in tables** — "1.72 S/m" needs exact token match.
RRF fusion captures both signals without tuning separate weights.

### Fusion: RRF (default) and weighted — both implemented
Two strategies, selectable per corpus:

- **RRF** (`fusion="rrf"`, default) — Reciprocal Rank Fusion uses only **rank
  position**, not raw scores, so the incompatible scales of BM25 (unbounded)
  and cosine (`[-1, 1]`) never have to be normalised. `score(d) = Σ 1/(k + rank_d)`, `k=60`. Robust, zero-tuning default.
- **Weighted** (`fusion="weighted"`, `alpha`) — `alpha * dense_norm + (1-alpha) * bm25_norm`
  after per-query min-max normalisation. `alpha` is a **tunable hyperparameter**:
  with a labelled dev set you can optimise the lexical/semantic balance for a
  specific domain.

Both are benchmarked side-by-side with `compare_configs` (Recall@k + MRR) so the
choice is data-driven, not arbitrary.

### Why Cross-encoder Re-ranking?
Bi-encoder embeds query and chunk **independently** — fast but loses inter-token attention.
Cross-encoder sees query+chunk **jointly** — slower but dramatically more accurate for top-k precision.
Two-stage: bi-encoder for recall (top-50), cross-encoder for precision (top-4).

### Semantic Cache with Entity-Aware Bypass
Risk: "conductivity of sample 5" and "conductivity of sample 6" have cosine > 0.95 but different answers.
Solution: regex detects entity patterns (sample IDs, numeric values + units) → bypasses cache entirely.

## Evaluation

Retrieval is evaluated *independently of the final LLM answer* on a hand-built
test set (questions paired with the chunk IDs that contain the answer):

| Metric | Description |
|--------|-------------|
| Recall@4 | Ground-truth chunk in top-4 retrieved |
| MRR | Mean reciprocal rank of first relevant chunk |

Run `python runner.py` to populate these numbers on your corpus.

## Quickstart (Google Colab)

```bash
pip install -r requirements.txt
```

```python
# With your own PDFs:
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("your_paper.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_documents(docs)

# Convert to (id, section, text) format and pass to RAGPipeline
corpus = [(f"c{i}", "paper", c.page_content) for i, c in enumerate(chunks)]

from src.rag_pipeline import RAGPipeline
pipeline = RAGPipeline(corpus)
print(pipeline.query("What reducing agent gave the best conductivity?"))
```

## Files

| File | Description |
|------|-------------|
| `src/rag_pipeline.py` | Full pipeline: BM25, dense retriever, hybrid RRF, cross-encoder, semantic cache |
| `src/pdf_loader.py` | Section-aware PDF loading (Abstract / Methods / Results / ...) |
| `src/evaluation.py` | Recall@k, MRR, config comparison |
| `runner.py` | Demo script with sample corpus + evaluation |
| `tests/test_logic.py` | Unit tests (mock embedder, no network) |

## Resume Bullet

> *"Built a production-grade RAG pipeline (hybrid BM25+dense retrieval with RRF
> fusion, cross-encoder re-ranking, entity-aware semantic cache) over 13
> peer-reviewed publications — evaluated with Recall@4 and MRR metrics."*

## References

- Robertson & Zaragoza (2009) — BM25
- Reimers & Gurevych (2019) — Sentence-BERT
- Nogueira & Cho (2019) — Passage Re-ranking with BERT
- Cormack et al. (2009) — Reciprocal Rank Fusion
- Lewis et al. (2020) — RAG (Facebook AI)

## License

MIT
