"""FastAPI service exposing the RAG pipeline.

    POST /qa     {"query": "...", "k": 5, "use_reranker": false}
    GET  /health

Run:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.pipeline import RAGPipeline

app = FastAPI(title="Materials-RAG", version="0.1.0")


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


class QARequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: Optional[int] = None
    use_reranker: Optional[bool] = None


class ContextOut(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    sources: List[str]
    metadata: dict


class QAResponse(BaseModel):
    answer: str
    used_chunks: List[str]
    contexts: List[ContextOut]
    metadata: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest) -> QAResponse:
    pipeline = get_pipeline()
    answer = pipeline.answer_question(req.query, k=req.k, use_reranker=req.use_reranker)
    payload = answer.to_dict()
    return QAResponse(
        answer=payload["answer"],
        used_chunks=payload["used_chunks"],
        contexts=[ContextOut(**c) for c in payload["contexts"]],
        metadata=payload["metadata"],
    )
