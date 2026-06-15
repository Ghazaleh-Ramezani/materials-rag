"""Core data structures passed between pipeline stages.

Kept dependency-light (stdlib dataclasses) so every module can import them
without pulling in FAISS, FastAPI, or any model library.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """A single retrievable unit of text plus its provenance metadata."""

    id: str  # globally unique, e.g. "graphene_review::0007"
    doc_id: str  # source document id, e.g. "graphene_review"
    chunk_id: int  # ordinal position within the document
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(
            id=d["id"],
            doc_id=d["doc_id"],
            chunk_id=int(d["chunk_id"]),
            text=d["text"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class RetrievedChunk:
    """A chunk returned by a retriever, annotated with a relevance score."""

    chunk: Chunk
    score: float
    # which retriever(s) surfaced this chunk, for debugging / explainability
    sources: List[str] = field(default_factory=list)

    def to_context_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "doc_id": self.chunk.doc_id,
            "text": self.chunk.text,
            "score": round(float(self.score), 6),
            "sources": self.sources,
            "metadata": self.chunk.metadata,
        }


@dataclass
class Answer:
    """Final generated answer with the contexts that grounded it."""

    answer_text: str
    used_chunks: List[str]  # chunk ids the generator was given / cited
    contexts: List[RetrievedChunk]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer_text,
            "used_chunks": self.used_chunks,
            "contexts": [c.to_context_dict() for c in self.contexts],
            "metadata": self.metadata,
        }
