"""
PDF loader for scientific papers — extracts text and tags sections.

Detects common scientific paper sections (Abstract, Introduction, Methods,
Results, Discussion, Conclusion, References) via heading patterns so that
chunks can later be filtered by section type — e.g. "experimental conditions"
queries can be restricted to Methods chunks for higher precision.
"""

from __future__ import annotations

import re
from typing import List

from pypdf import PdfReader


SECTION_PATTERNS = [
    (r"\babstract\b", "Abstract"),
    (r"\b(introduction|background)\b", "Introduction"),
    (r"\b(materials and methods|methods|methodology|experimental)\b", "Methods"),
    (r"\b(results and discussion|results)\b", "Results"),
    (r"\b(discussion)\b", "Discussion"),
    (r"\b(conclusion|conclusions|summary)\b", "Conclusion"),
    (r"\b(references|bibliography)\b", "References"),
]


def detect_section(line: str) -> str | None:
    """Return a section label if the line looks like a section heading."""
    stripped = line.strip().lower()
    # headings are short and often numbered ("3. Methods", "Results")
    if len(stripped) > 60:
        return None
    for pattern, label in SECTION_PATTERNS:
        if re.search(pattern, stripped):
            # avoid matching mid-sentence — heading-ish lines are mostly the term
            words = re.findall(r"\w+", stripped)
            if len(words) <= 5:
                return label
    return None


def load_pdf(path: str) -> List[dict]:
    """
    Load a PDF and split into section-tagged text blocks.
    Returns a list of {text, source, section}.
    """
    reader = PdfReader(path)
    source = path.split("/")[-1]

    full_text = []
    for page in reader.pages:
        full_text.append(page.extract_text() or "")
    text = "\n".join(full_text)

    blocks: List[dict] = []
    current_section = "Unknown"
    buffer: List[str] = []

    for line in text.split("\n"):
        sec = detect_section(line)
        if sec:
            if buffer:
                blocks.append({
                    "text": "\n".join(buffer).strip(),
                    "source": source,
                    "section": current_section,
                })
                buffer = []
            current_section = sec
        else:
            buffer.append(line)

    if buffer:
        blocks.append({
            "text": "\n".join(buffer).strip(),
            "source": source,
            "section": current_section,
        })

    # drop empty/tiny blocks
    return [b for b in blocks if len(b["text"]) > 50]


def load_pdf_directory(directory: str) -> List[dict]:
    """Load all PDFs in a directory."""
    import os
    docs: List[dict] = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(directory, fname)
            try:
                docs.extend(load_pdf(path))
                print(f"  loaded {fname}")
            except Exception as e:
                print(f"  ! failed {fname}: {e}")
    return docs
