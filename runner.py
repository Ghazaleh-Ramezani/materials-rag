"""
Demo runner — sample corpus + evaluation.

Run on Google Colab (or any machine with internet) where the HuggingFace
models can download:

    pip install -r requirements.txt
    python runner.py
"""
import sys
sys.path.insert(0, "src")

from rag_pipeline import RAGPipeline, Chunk, Result
from evaluation import EvalExample, evaluate, compare_configs


# Sample corpus from Ramezani et al. 2025 (replace with your real PDF chunks)
SAMPLE_CORPUS = [
    ("c0", "Results",  "L-ascorbic acid acts as a green reductant for graphene oxide, "
                       "yielding an electrical conductivity of 1.72 S/m."),
    ("c1", "Results",  "Citric acid as a reducing agent produced a lower conductivity of 1.62 S/m."),
    ("c2", "Results",  "The tensile strength of the CNC/CNF/rGO composite reached 46 MPa."),
    ("c3", "Methods",  "Graphene oxide was synthesized via a modified Hummers method."),
    ("c4", "Methods",  "Thin films were prepared by spin coating at 2000 rpm for 30 s."),
    ("c5", "Abstract", "We report a LASSO-based optimization of nanocellulose/rGO composites "
                       "for flexible electronics, balancing conductivity and mechanical strength."),
    ("c6", "Discussion","The dielectric response correlated strongly with filler dispersion uniformity."),
]


def main():
    print("=" * 64)
    print("Materials Science RAG — RRF Hybrid + Cross-Encoder Re-ranking")
    print("=" * 64)

    print("\n[1] Building pipeline (downloads models on first run)...")
    pipeline = RAGPipeline(SAMPLE_CORPUS, use_cache=True)
    print(f"    Indexed {len(pipeline.chunks)} chunks.")

    # --- Demo query
    q = "Which reducing agent gave the highest conductivity?"
    print(f"\n[2] Query: {q}\n")
    for i, r in enumerate(pipeline.retrieve(q, k=3), 1):
        print(f"  {i}. [{r.score:.3f}] ({r.chunk.section}) {r.chunk.text[:70]}...")

    # --- Evaluation: hand-built test set
    print("\n[3] Retrieval evaluation (Recall@k + MRR):")
    examples = [
        EvalExample("which reductant gave best conductivity", {0}),
        EvalExample("what tensile strength was reached", {2}),
        EvalExample("how was graphene oxide synthesized", {3}),
        EvalExample("what deposition method for thin films", {4}),
    ]

    # map chunk_id string -> need int ids for eval; adapt retrieve to ids
    id_to_idx = {c.chunk_id: i for i, c in enumerate(pipeline.chunks)}

    def retrieve_ids(query, k):
        # returns Result objects whose chunk.chunk_id we convert to int index
        results = pipeline.retrieve(query, k=k, fetch_k=50)
        # wrap so evaluation sees integer chunk_id
        out = []
        for r in results:
            rr = Result(
                chunk=type(r.chunk)(chunk_id=id_to_idx[r.chunk.chunk_id],
                                    section=r.chunk.section, text=r.chunk.text),
                score=r.score, method=r.method,
            )
            out.append(rr)
        return out

    examples_int = [EvalExample(e.question, e.relevant_chunk_ids) for e in examples]
    metrics = evaluate(examples_int, retrieve_ids)
    for key, val in metrics.items():
        print(f"    {key}: {val:.3f}")

    # --- Cache demo
    print("\n[4] Semantic cache demo:")
    a1 = pipeline.query("Give me an overview of the composite")
    a2 = pipeline.query("Give me an overview of the composite")  # should hit cache
    print(f"    2nd identical query cached: {'cache hit' in a2}")
    a3 = pipeline.query("What is the conductivity of sample 5?")  # entity -> bypass
    print(f"    entity query bypassed cache: {'cache hit' not in a3}")


if __name__ == "__main__":
    main()
