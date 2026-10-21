"""Semantic cache with entity-aware matching.

Lookup order:
1. Exact match on the normalized query, its entity set and its namespace.
2. Entity-matched semantic match: the query is embedded and compared only with
   cached entries that carry exactly the same entity set and namespace. A
   question about an rGO/CNC film and one about an rGO/CNF film share "rgo"
   but are different questions, so their entity sets differ and they never
   share an answer, however close their embeddings are.
3. Unconstrained semantic match (opt-in, ``unconstrained_fallback=True``):
   cosine similarity over every entry in the namespace, entities ignored.

A query with no detected entities gets exact matches only, unless the
unconstrained fallback is switched on.

The namespace carries the request parameters that change the answer, such as
the number of contexts ``k`` and whether the reranker ran, so a request for
k=3 is never served a cached k=1 answer.

Entity extraction
-----------------
A lightweight rule-based extractor tuned for materials-science text: material
names and abbreviations (CNC, CNF, rGO, PVDF, ...), chemical formulas, and
number-unit pairs ("10 wt%", "50 GPa", "25 °C"). A spaCy NER model can be
plugged in through ``entity_extractor``.

Persistence
-----------
Entries are appended to a JSONL file. Each row stores the normalized query
text, and that same text is what gets embedded both when an entry is written
and when the file is reloaded, so similarities are identical before and after
a restart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from src.config import config
from src.indexing.embedders import Embedder, get_embedder

_WS_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Rule-based entity extractor for materials science
# ---------------------------------------------------------------------------

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

# Number-unit pairs: "10 wt%", "50 GPa", "300 K", "1.5 S/m", "25 °C". Units
# such as "wt%" or "°C" end or start in a non-word character, where \b does
# not match, so the boundaries are explicit look-arounds instead.
_NUMERIC_UNIT = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s*"
    r"(?:wt%|vol%|GPa|MPa|kPa|Pa|nm|μm|um|mm|cm|"
    r"S/m|S/cm|Ω|mΩ|GF|Hz|K|°C|mol|g/mol)"
    r"(?![\w%/])",
    re.IGNORECASE,
)

# Chemical formula or upper-case abbreviation: capital letter, then letters or
# digits, containing a digit or written entirely in upper case.
_FORMULA = re.compile(r"\b[A-Z][a-zA-Z0-9]{1,12}\b")


def extract_entities(text: str) -> Set[str]:
    """Return lower-cased entity strings found in `text`."""
    entities: Set[str] = set()
    for m in _MATERIALS.finditer(text):
        entities.add(m.group(0).lower())
    for m in _NUMERIC_UNIT.finditer(text):
        entities.add(m.group(0).lower().replace(" ", ""))
    for m in _FORMULA.finditer(text):
        tok = m.group(0)
        if any(c.isdigit() for c in tok) or tok.isupper():
            entities.add(tok.lower())
    return entities


def _normalize(query: str) -> str:
    return _WS_RE.sub(" ", query.strip().lower())


def _cache_key(norm_query: str, entities: Set[str], namespace: str) -> str:
    return f"{namespace}##{norm_query}##entities={'|'.join(sorted(entities))}"


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


class _Entry:
    __slots__ = ("key", "query", "namespace", "payload", "entities")

    def __init__(self, key: str, query: str, namespace: str, payload: dict, entities: Set[str]):
        self.key = key
        self.query = query  # normalized query text; this is what gets embedded
        self.namespace = namespace
        self.payload = payload
        self.entities = frozenset(entities)


# ---------------------------------------------------------------------------
# Entity-aware semantic cache
# ---------------------------------------------------------------------------


class SemanticCache:
    """
    Args:
        embedder: Sentence embedding model. Defaults to the project embedder.
        threshold: Cosine similarity needed for a semantic hit (default from config).
        entity_extractor: Callable str -> Set[str]. Defaults to the rule-based extractor.
        unconstrained_fallback: Also allow semantic hits that ignore entities.
            Off by default.
        path: JSONL file the cache persists to. Defaults to ``config.cache_path``.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        threshold: Optional[float] = None,
        entity_extractor: Optional[Callable[[str], Set[str]]] = None,
        unconstrained_fallback: bool = False,
        path: Optional[Path] = None,
    ):
        self.embedder = embedder or get_embedder()
        self.threshold = config.cache_similarity_threshold if threshold is None else threshold
        self.extract = entity_extractor or extract_entities
        self.unconstrained_fallback = unconstrained_fallback
        self.path = Path(path) if path is not None else config.cache_path

        self._by_key: Dict[str, _Entry] = {}
        self._entries: List[_Entry] = []
        self._matrix: Optional[np.ndarray] = None  # (n, dim) unit vectors, row i <-> _entries[i]
        self._load()

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ lookup

    def get(self, query: str, namespace: str = "") -> Optional[dict]:
        entities = self.extract(query)
        norm = _normalize(query)
        key = _cache_key(norm, entities, namespace)

        if key in self._by_key:
            return {**self._by_key[key].payload, "_cache": "exact"}

        if self._matrix is None:
            return None

        in_ns = [i for i, e in enumerate(self._entries) if e.namespace == namespace]
        if not in_ns:
            return None
        q_emb = self.embedder.encode([norm])[0]

        if entities:
            same = [i for i in in_ns if self._entries[i].entities == entities]
            hit = self._best(same, q_emb)
            if hit is not None:
                i, sim = hit
                return {
                    **self._entries[i].payload,
                    "_cache": "entity_semantic",
                    "_cache_similarity": round(sim, 4),
                    "_cache_entities": sorted(entities),
                }

        if self.unconstrained_fallback:
            hit = self._best(in_ns, q_emb)
            if hit is not None:
                i, sim = hit
                return {
                    **self._entries[i].payload,
                    "_cache": "semantic",
                    "_cache_similarity": round(sim, 4),
                }
        return None

    def _best(self, indices: List[int], q_emb: np.ndarray):
        if not indices:
            return None
        sims = self._matrix[indices] @ q_emb
        j = int(np.argmax(sims))
        if float(sims[j]) >= self.threshold:
            return indices[j], float(sims[j])
        return None

    # ------------------------------------------------------------------ write

    def set(self, query: str, payload: dict, namespace: str = "") -> None:
        entities = self.extract(query)
        norm = _normalize(query)
        key = _cache_key(norm, entities, namespace)
        if key in self._by_key:
            return
        self._add(_Entry(key, norm, namespace, payload, entities), self.embedder.encode([norm]))
        self._append_disk(self._by_key[key])

    def _add(self, entry: _Entry, emb: np.ndarray) -> None:
        self._by_key[entry.key] = entry
        self._entries.append(entry)
        self._matrix = emb if self._matrix is None else np.vstack([self._matrix, emb])

    # ------------------------------------------------------------------ persistence

    def _append_disk(self, entry: _Entry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "key": entry.key,
            "query": entry.query,
            "namespace": entry.namespace,
            "entities": sorted(entry.entities),
            "payload": entry.payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        if not self.path.exists():
            return
        entries: Dict[str, _Entry] = {}
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "query" not in r:
                    continue  # rows written before the query text was stored
                entities = set(r.get("entities", []))
                namespace = r.get("namespace", "")
                key = _cache_key(r["query"], entities, namespace)
                entries.setdefault(key, _Entry(key, r["query"], namespace, r["payload"], entities))
        if entries:
            loaded = list(entries.values())
            embs = self.embedder.encode([e.query for e in loaded])
            for e in loaded:
                self._by_key[e.key] = e
                self._entries.append(e)
            self._matrix = embs if self._matrix is None else np.vstack([self._matrix, embs])
