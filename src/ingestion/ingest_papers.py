"""Ingestion: turn raw papers (PDF or TXT) into clean, chunked, metadata-rich
records written to ``data/processed/chunks.jsonl``.

Usage:
    python -m src.ingestion.ingest_papers                 # uses data/raw, falls back to sample_corpus
    python -m src.ingestion.ingest_papers --input-dir X   # explicit source dir
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from src.config import config
from src.schemas import Chunk

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pypdf is required to read PDFs. `pip install pypdf` or drop .txt files in data/raw."
        ) from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return _read_txt(path)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_REFERENCES_RE = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)
_PAGE_NUM_RE = re.compile(r"\n\s*\d{1,4}\s*\n")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Light, heuristic cleaning. Not perfect — that's expected for scraped PDFs.

    - drops everything after a 'References'/'Bibliography' heading
    - removes lone page numbers
    - collapses runaway whitespace
    """
    text = raw.replace("\r", "\n")
    # cut the reference list (keep only the first body before it)
    match = _REFERENCES_RE.search(text)
    if match:
        text = text[: match.start()]
    text = _PAGE_NUM_RE.sub("\n", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, size_words: int, overlap_words: int) -> List[str]:
    """Word-based sliding window. We approximate tokens with words (~1.3 tok/word
    for English prose) to stay dependency-free; documented as such in the README."""
    words = text.split()
    if not words:
        return []
    if size_words <= 0:
        return [text]
    step = max(1, size_words - overlap_words)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + size_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size_words >= len(words):
            break
    return chunks


def _doc_metadata(path: Path) -> Dict[str, str]:
    """Best-effort metadata. If a sidecar ``<name>.meta.json`` exists, merge it."""
    meta: Dict[str, str] = {
        "source_file": path.name,
        "title": path.stem.replace("_", " ").title(),
    }
    sidecar = path.with_suffix(".meta.json")
    if sidecar.exists():
        try:
            meta.update(json.loads(sidecar.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return meta


def iter_chunks(input_dir: Path) -> Iterable[Chunk]:
    files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in {".pdf", ".txt"}
    )
    for path in files:
        doc_id = path.stem
        cleaned = clean_text(extract_text(path))
        metadata = _doc_metadata(path)
        for i, piece in enumerate(
            chunk_text(cleaned, config.chunk_size_words, config.chunk_overlap_words)
        ):
            yield Chunk(
                id=f"{doc_id}::{i:04d}",
                doc_id=doc_id,
                chunk_id=i,
                text=piece,
                metadata=metadata,
            )


def run(input_dir: Path | None = None, output_path: Path | None = None) -> int:
    config.ensure_dirs()
    input_dir = input_dir or config.raw_dir
    # fall back to the bundled sample corpus if data/raw is empty
    has_raw = input_dir.exists() and any(
        p.suffix.lower() in {".pdf", ".txt"} for p in input_dir.iterdir()
    )
    if not has_raw:
        print(f"[ingest] {input_dir} has no PDF/TXT; using sample_corpus instead.")
        input_dir = config.sample_corpus_dir
    output_path = output_path or config.chunks_path

    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in iter_chunks(input_dir):
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    print(f"[ingest] wrote {count} chunks -> {output_path}")
    return count


def load_chunks(path: Path | None = None) -> List[Chunk]:
    path = path or config.chunks_path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.ingestion.ingest_papers` first."
        )
    with path.open(encoding="utf-8") as fh:
        return [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Ingest papers into chunks.jsonl")
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    run(args.input_dir, args.output)


if __name__ == "__main__":
    _cli()
