# 🚀 Setup Guide — GitHub + Colab

## Part 1 — Run on Google Colab (get real numbers)

1. Go to [colab.research.google.com](https://colab.research.google.com), new notebook.

2. Upload `scientific_rag.zip` (Files panel → Upload), then unzip:
   ```python
   !unzip scientific_rag.zip
   %cd scientific_rag
   !pip install -r requirements.txt
   ```

3. Run the demo on the bundled sample corpus (downloads models first time):
   ```python
   !python runner.py
   ```
   You'll see retrieval results + Recall@k / MRR printed.

4. **With your own papers:** upload your PDFs into `scientific_rag/data/`, then:
   ```python
   import sys; sys.path.insert(0, "src")
   from pdf_loader import load_pdf_directory
   from rag_pipeline import RAGPipeline

   docs = load_pdf_directory("data")
   corpus = [(f"c{i}", d["section"], d["text"]) for i, d in enumerate(docs)]
   rag = RAGPipeline(corpus, fusion="rrf")          # or fusion="weighted", alpha=0.5
   print(rag.query("Which reducing agent gave the best conductivity?"))
   ```

5. **Compare fusion methods** (the interview talking point):
   ```python
   from evaluation import EvalExample, compare_configs
   from rag_pipeline import HybridRetriever, Chunk

   chunks = [Chunk(str(i), d["section"], d["text"]) for i, d in enumerate(docs)]
   # build two retrievers
   rrf = HybridRetriever(fusion="rrf");        rrf.index(chunks)
   wtd = HybridRetriever(fusion="weighted", alpha=0.5); wtd.index(chunks)

   examples = [EvalExample("...", {0})]  # fill with your ground-truth chunk ids
   compare_configs(examples, {
       "RRF":            lambda q, k: rrf.retrieve(q, k),
       "weighted(0.5)":  lambda q, k: wtd.retrieve(q, k),
   })
   ```

6. **Screenshot** the Recall@k / MRR output — that's your evidence.

---

## Part 2 — Push to GitHub

```bash
# on your machine, inside the unzipped folder
cd scientific_rag
git init
git add .
git commit -m "Materials-science RAG: hybrid retrieval, RRF/weighted fusion, cross-encoder rerank, semantic cache"
git branch -M main
git remote add origin https://github.com/Ghazaleh-Ramezani/materials-rag.git
git push -u origin main
```

(Create the empty repo first at github.com/new, name it `materials-rag`.)

---

## Part 3 — Add to résumé (with real numbers)

Once you have the Recall@4 / MRR numbers from step 6, the bullet becomes:

> *"Built a hybrid-retrieval RAG pipeline (BM25 + dense with RRF and weighted
> fusion, cross-encoder re-ranking, entity-aware semantic cache) over my
> published materials-science corpus — achieving Recall@4 = X.XX and MRR = X.XX,
> with section-aware retrieval and source attribution.
> github.com/Ghazaleh-Ramezani/materials-rag"*

---

## Part 4 — Interview talking points (you can defend all of these)

| Question | Your answer |
|---|---|
| Why hybrid? | BM25 nails exact numbers/entities; dense handles synonyms (reducing agent ≈ reductant) |
| RRF vs weighted? | Implemented both. RRF = robust, no normalisation. Weighted = tunable alpha on a dev set. Benchmarked with compare_configs. |
| Why cross-encoder? | Bi-encoder embeds separately (fast, recall); cross-encoder joint attention (precise). Two-stage: top-50 → top-4. |
| Cache risk? | "sample 5" vs "sample 6" embed >0.95 but differ → entity-aware regex bypass. |
| How do you know retrieval is good? | Recall@k + MRR on a hand-built test set, independent of the LLM answer. |
| Production scaling? | FAISS-flat → managed HNSW (Pinecone), semantic cache, async FastAPI, batched LLM calls. |
