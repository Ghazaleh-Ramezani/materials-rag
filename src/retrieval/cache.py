"""Semantic cache with entity-aware cache key construction.

Three lookup layers:
1. Exact match on a normalized, entity-anchored cache key.
2. Entity-constrained semantic match — embeds the query but only considers
   cache entries that share at least one named entity with the incoming query.
   This prevents a query about "CNC" from hitting a cached answer about "CNF"
   even when their embedding similarity is above the threshold.
3. Unconstrained semantic fallback (optional, off by default) for queries
   where no entities are detected.

Entity extraction
-----------------
Uses a lightweight rule-based extractor tuned for materials-science text:
chemical formulas, material abbreviations (CNC, CNF, rGO, PVDF, ...), and
numeric–unit pairs (e.g. "10 wt%", "50 GPa").  A spaCy NER model can be
plugged in instead via the `entity_extractor` argument.

Why this matters
----------------
Without entity awareness, two semantically similar queries like
  "What is the gauge factor of rGO sensors?"
  "What is the gauge factor of CNC sensors?"
can share a cache entry because their embeddings are close.  Entity-gating
blocks this: rGO ≠ CNC, so the second query misses and triggers fresh retrieval.

Entries persist to a JSONL file so the cache survives restarts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from src.config import config
from src.indexing.embedders import Embedder, get_embedder

_WS_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Rule-based entity extractor for materials science
# ---------------------------------------------------------------------------

# Common materials abbreviations and chemical names
_MATERIALS = re.compile(
    r"\b("
    r"rGO|GO|graphene|"
    r"CNC|CNF|nanocellulose|cellulose|"
    r"PVDF|PDMS|epoxy|"
    r"carbon nanotube|CNT|MWCNT|SWCNT|"
    r"chitosan|alginate|"
    r"TiO2|ZnO|Fe3O4|Al2O3|SiO2"
    r")\b",
    re.IGNORECASE,
)

# Numeric + unit pairs: "10 wt%", "50 GPa", "300 K", "1.5 S/m"
_NUMERIC_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:wt%|vol%|GPa|MPa|kPa|Pa|nm|μm|mm|cm|"
    r"S/m|S/cm|Ω|mΩ|GF|Hz|K|°C|mol|g/mol)\b",
    re.IGNORECASE,
)

# Generic chemical formula: starts with capital letter, has digits
_FORMULA = re.compile(r"\b[A-Z][a-zA-Z0-9]{1,12}\b")


def extract_entities(text: str) -> Set[str]:
    """
    Extract a set of canonical entity strings from a materials-science query.

    Returns lowercase strings for case-insensitive comparison.
    Precedence: known materials > numeric-unit pairs > formulas.
    """
    entities: Set[str] = set()

    for m in _MATERIALS.finditer(text):
        entities.add(m.group(0).lower())

    for m in _NUMERIC_UNIT.finditer(text):
        entities.add(m.group(0).lower().replace(" ", ""))

    for m in _FORMULA.finditer(text):
        tok = m.group(0)
        # Heuristic: skip common English words that match the pattern
        if any(c.isdigit() for c in tok) or tok.isupper():
            entities.add(tok.lower())

    return entities


def _normalize(query: str) -> str:
    return _WS_RE.sub(" ", query.strip().lower())


def _entity_key(norm_query: str, entities: Set[str]) -> str:
    """Build a cache key that embeds sorted entity strings into the query."""
    if not entities:
        return norm_query
    sorted_ents = "|".join(sorted(entities))
    return f"{norm_query}##entities={sorted_ents}"


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

class _Entry:
    __slots__ = ("norm_key", "payload", "entities", "embedding")

    def __init__(
        self,
        norm_key: str,
        payload: dict,
        entities: Set[str],
        embedding: Optional[np.ndarray] = None,
    ):
        self.norm_key = norm_key
        self.payload = payload
        self.entities = entities
        self.embedding = embedding  # (dim,) unit vector, set after first encode


# ---------------------------------------------------------------------------
# Entity-aware semantic cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Three-layer cache: exact-key → entity-constrained semantic → unconstrained semantic.

    Args:
        embedder: Sentence embedding model. Defaults to project embedder.
        threshold: Cosine similarity threshold for semantic layer (default from config).
        entity_extractor: Optional callable (str) -> Set[str]. Defaults to rule-based extractor.
        unconstrained_fallback: If True and no entities detected, fall back to pure cosine
            similarity (original behaviour). Default False — requires explicit entity overlap.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        threshold: Optional[float] = None,
        entity_extractor=None,
        unconstrained_fallback: bool = False,
    ):
        self.embedder = embedder or get_embedder()
        self.threshold = config.cache_similarity_threshold if threshold is None else threshold
        self.extract = entity_extractor or extract_entities
        self.unconstrained_fallback = unconstrained_fallback

        self._by_key: Dict[str, _Entry] = {}   # norm_key → entry
        self._entries: List[_Entry] = []        # ordered list for matrix ops
        self._matrix: Optional[np.ndarray] = None  # (n, dim) normalized embeddings
        self._load()

    # ------------------------------------------------------------------ lookup

    def get(self, query: str) -> Optional[dict]:
        entities = self.extract(query)
        norm = _normalize(query)
        key = _entity_key(norm, entities)

        # Layer 1: exact key match
        if key in self._by_key:
            hit = dict(self._by_key[key].payload)
            hit["_cache"] = "exact"
            return hit

        if not self._entries or self._matrix is None:
            return None

        q_emb = self.embedder.encode([query])  # (1, dim), normalized

        # Layer 2: entity-constrained semantic match
        # Only consider entries whose entity set overlaps with the query's entities
        if entities:
            candidate_indices = [
                i for i, e in enumerate(self._entries)
                if e.entities & entities  # non-empty intersection
            ]
            if candidate_indices:
                sub_matrix = self._matrix[candidate_indices]   # (m, dim)
                sims = sub_matrix @ q_emb[0]                   # (m,)
                best_local = int(np.argmax(sims))
                best_global = candidate_indices[best_local]
                if float(sims[best_local]) >= self.threshold:
                    entry = self._entries[best_global]
                    hit = dict(entry.payload)
                    hit["_cache"] = "entity_semantic"
                    hit["_cache_similarity"] = round(float(sims[best_local]), 4)
                    hit["_cache_matched_entities"] = sorted(
                        entities & entry.entities
                    )
                    return hit

        # Layer 3: unconstrained semantic fallback (opt-in)
        if self.unconstrained_fallback or not entities:
            sims = self._matrix @ q_emb[0]
            best = int(np.argmax(sims))
            if float(sims[best]) >= self.threshold:
                entry = self._entries[best]
                hit = dict(entry.payload)
                hit["_cache"] = "semantic"
                hit["_cache_similarity"] = round(float(sims[best]), 4)
                return hit

        return None

    # ------------------------------------------------------------------ write

    def set(self, query: str, payload: dict) -> None:
        entities = self.extract(query)
        norm = _normalize(query)
        key = _entity_key(norm, entities)

        if key in self._by_key:
            return  # already cached

        emb = self.embedder.encode([query])  # (1, dim)
        entry = _Entry(norm_key=key, payload=payload, entities=entities, embedding=emb[0])

        self._by_key[key] = entry
        self._entries.append(entry)
        self._matrix = (
            emb if self._matrix is None
            else np.vstack([self._matrix, emb])
        )
        self._append_disk(key, payload, entities)

    # ------------------------------------------------------------------ persistence

    def _append_disk(self, key: str, payload: dict, entities: Set[str]) -> None:
        path = config.cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"key": key, "entities": sorted(entities), "payload": payload},
                ensure_ascii=False,
            ) + "\n")

    def _load(self) -> None:
        path = config.cache_path
        if not path.exists():
            return
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if not rows:
            return

        keys = [r.get("key") or _normalize(r.get("query", "")) for r in rows]
        embeddings = self.embedder.encode(keys)  # (n, dim)

        for i, (r, emb) in enumerate(zip(rows, embeddings)):
            key = keys[i]
            entities = set(r.get("entities", []))
            payload = r.get("payload", r)  # backwards-compat: old format had no "payload" wrapper
            entry = _Entry(norm_key=key, payload=payload, entities=entities, embedding=emb)
            self._by_key[key] = entry
            self._entries.append(entry)

        self._matrix = embeddings
