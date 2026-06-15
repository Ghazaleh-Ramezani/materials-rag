"""Answer generation over retrieved contexts.

Providers:
* ``mock``      — offline extractive answer; no key needed. Default.
* ``anthropic`` — Claude via the anthropic SDK (needs ANTHROPIC_API_KEY).
* ``openai``    — GPT via the openai SDK (needs OPENAI_API_KEY).

All providers receive the same RAG-style prompt and are asked to cite contexts
by their ``[doc_id::chunk_id]`` tag.
"""

from __future__ import annotations

from typing import List, Tuple

from src.config import config
from src.schemas import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a careful scientific assistant. Answer the question using ONLY the "
    "provided context passages. Cite the passages you rely on inline using their "
    "[doc_id::chunk_id] tags. If the context is insufficient, say so explicitly."
)


def build_prompt(query: str, contexts: List[RetrievedChunk]) -> str:
    blocks = []
    for c in contexts:
        blocks.append(f"[{c.chunk.id}] {c.chunk.text}")
    context_str = "\n\n".join(blocks)
    return (
        f"Context passages:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer (with [doc_id::chunk_id] citations):"
    )


def _mock_answer(query: str, contexts: List[RetrievedChunk]) -> str:
    """Deterministic extractive 'answer' so the pipeline runs without an API key.
    Stitches the top contexts together with citation tags — useful for plumbing,
    demos, and retrieval-only evaluation."""
    if not contexts:
        return "No relevant context was retrieved for this question."
    top = contexts[: min(3, len(contexts))]
    lines = [
        "Based on the retrieved passages (mock generation; set LLM_PROVIDER to "
        "anthropic/openai for real synthesis):"
    ]
    for c in top:
        snippet = c.chunk.text.strip().replace("\n", " ")
        snippet = (snippet[:240] + "…") if len(snippet) > 240 else snippet
        lines.append(f"- [{c.chunk.id}] {snippet}")
    return "\n".join(lines)


def _anthropic_answer(prompt: str) -> str:  # pragma: no cover - needs network/key
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=config.anthropic_model,
        max_tokens=config.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _openai_answer(prompt: str) -> str:  # pragma: no cover - needs network/key
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=config.openai_model,
        max_tokens=config.max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


class LLMClient:
    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or config.llm_provider).lower()

    def generate(self, query: str, contexts: List[RetrievedChunk]) -> Tuple[str, List[str]]:
        used = [c.chunk.id for c in contexts]
        if self.provider == "mock":
            return _mock_answer(query, contexts), used
        prompt = build_prompt(query, contexts)
        if self.provider == "anthropic":
            return _anthropic_answer(prompt), used
        if self.provider == "openai":
            return _openai_answer(prompt), used
        raise ValueError(f"Unknown LLM provider: {self.provider!r}")
