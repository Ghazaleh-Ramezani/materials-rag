"""Dense index in PostgreSQL with pgvector: same interface as the FAISS ``DenseIndex``.

Use it when the corpus should live in a database the rest of the system already runs
(backups, access control, filtering by metadata in SQL) instead of an index file:

    VECTOR_BACKEND=pgvector PGVECTOR_DSN=postgresql://rag:rag@localhost:5432/rag \\
        python -m scripts.build_indexes

Vectors are L2-normalized, so ranking by inner product (``<#>``, which returns the
negated inner product) equals cosine ranking, as with ``faiss.IndexFlatIP``. An HNSW
index (``vector_ip_ops``) serves the queries; ``hnsw.ef_search`` is raised per query so
the candidate list is never shorter than ``k``. Each collection records the embedder
that produced its vectors and refuses to load under a different one, as the FAISS index
does: two embedders can share a dimension and still live in unrelated spaces.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.config import config
from src.indexing.embedders import Embedder, get_embedder
from src.schemas import Chunk

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rag_collections (
    name      text PRIMARY KEY,
    embedder  text NOT NULL,
    dim       integer NOT NULL,
    n         integer NOT NULL,
    built_at  timestamptz NOT NULL DEFAULT now()
);
"""


def _connect(dsn: str):
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(dsn, autocommit=False)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def _table(collection: str) -> str:
    if not collection.replace("_", "").isalnum():
        raise ValueError(f"collection name must be alphanumeric/underscore: {collection!r}")
    return f"rag_vectors_{collection}"


class PgVectorIndex:
    backend = "pgvector"

    def __init__(self, embedder: Embedder | None = None, dsn: str | None = None,
                 collection: str | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self.dsn = dsn or config.pgvector_dsn
        self.collection = collection or config.pgvector_collection
        self.table = _table(self.collection)
        self._n = 0

    def build(self, chunks: List[Chunk]) -> "PgVectorIndex":
        """(Re)create the collection: one row per chunk, then the HNSW index, in one transaction."""
        emb = self.embedder.encode([c.text for c in chunks]).astype(np.float32)
        dim = int(emb.shape[1])
        with _connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(SCHEMA)
            cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            cur.execute(f"""CREATE TABLE {self.table} (
                                position  integer PRIMARY KEY,
                                chunk_id  text UNIQUE NOT NULL,
                                doc_id    text NOT NULL,
                                embedding vector({dim}) NOT NULL)""")
            cur.executemany(f"INSERT INTO {self.table} (position, chunk_id, doc_id, embedding) VALUES (%s, %s, %s, %s)",
                            [(i, c.id, c.doc_id, emb[i]) for i, c in enumerate(chunks)])
            cur.execute(f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_ip_ops) "
                        "WITH (m = 16, ef_construction = 64)")
            cur.execute("""INSERT INTO rag_collections (name, embedder, dim, n) VALUES (%s, %s, %s, %s)
                           ON CONFLICT (name) DO UPDATE SET embedder = EXCLUDED.embedder, dim = EXCLUDED.dim,
                                                            n = EXCLUDED.n, built_at = now()""",
                        (self.collection, self.embedder.name, dim, len(chunks)))
            conn.commit()
        self._n = len(chunks)
        return self

    def search_dense(self, query: str, k: int) -> List[Tuple[str, float]]:
        if not self._n:
            raise RuntimeError("PgVectorIndex not built/loaded.")
        q = self.embedder.encode([query])[0].astype(np.float32)
        k = min(k, self._n)
        with _connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"SET LOCAL hnsw.ef_search = {max(40, 2 * k)}")
            cur.execute(f"""SELECT chunk_id, -(embedding <#> %s) AS score FROM {self.table}
                            ORDER BY embedding <#> %s LIMIT %s""", (q, q, k))
            rows = cur.fetchall()
        return [(cid, float(score)) for cid, score in rows]

    def save(self, path=None) -> None:
        """Nothing to write: the database is the index (kept for interface parity)."""

    @classmethod
    def load(cls, path=None, embedder: Embedder | None = None, dsn: str | None = None,
             collection: str | None = None) -> "PgVectorIndex":
        obj = cls(embedder=embedder, dsn=dsn, collection=collection)
        with _connect(obj.dsn) as conn, conn.cursor() as cur:
            cur.execute(SCHEMA)
            cur.execute("SELECT embedder, n FROM rag_collections WHERE name = %s", (obj.collection,))
            row = cur.fetchone()
            conn.commit()
        if row is None:
            raise ValueError(f"pgvector collection {obj.collection!r} does not exist. "
                             "Build it with `VECTOR_BACKEND=pgvector python -m scripts.build_indexes`.")
        built_with, obj._n = row
        if built_with != obj.embedder.name:
            raise ValueError(
                f"pgvector collection {obj.collection!r} was built with embedder {built_with!r}, but the current "
                f"embedder is {obj.embedder.name!r}. Rebuild it with `python -m scripts.build_indexes`."
            )
        return obj


def get_dense_index_class():
    """The dense index implementation selected by ``VECTOR_BACKEND`` (faiss | pgvector)."""
    from src.indexing.dense_index import DenseIndex

    backends = {"faiss": DenseIndex, "pgvector": PgVectorIndex}
    if config.vector_backend not in backends:
        raise ValueError(f"VECTOR_BACKEND must be one of {sorted(backends)}, got {config.vector_backend!r}")
    return backends[config.vector_backend]
