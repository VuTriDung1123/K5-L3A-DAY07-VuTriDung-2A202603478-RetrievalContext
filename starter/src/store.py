"""PostgreSQL + pgvector persistent store with exact cosine search and SQL metadata filtering."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import psycopg

from .models import Chunk

ALLOWED_FILTER_KEYS = frozenset({"chapter", "section", "document_id", "page", "chapter_number"})


def _to_pgvector(vec: list[float], expected_dim: int) -> str:
    """Validate vector length/finiteness and format as pgvector literal."""
    if len(vec) != expected_dim:
        raise ValueError(f"Vector dimension mismatch: expected {expected_dim}, got {len(vec)}.")
    for val in vec:
        if not math.isfinite(val):
            raise ValueError("Vector contains non-finite value (NaN or Inf).")
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


def _hash_chunk(chunk: Chunk) -> str:
    payload = json.dumps(
        {"text": chunk.text, "page": chunk.page, "metadata": chunk.metadata},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PgVectorStore:
    """Persistent PostgreSQL + pgvector store supporting multi-strategy partitioning and SQL filters."""

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url or not database_url.strip():
            raise ValueError("DATABASE_URL must be non-empty.")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        self.database_url = database_url
        self.dimensions = dimensions
        self.metric = "cosine"
        self.search_mode = "exact"
        self.last_upsert_stats: dict[str, int] = {}

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self.database_url, connect_timeout=10)

    def server_info(self) -> dict[str, str]:
        """Return PostgreSQL and pgvector extension version info."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT current_database(), current_setting('server_version'), extversion "
                "FROM pg_extension WHERE extname = 'vector'"
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("pgvector extension is not installed in database.")
            return {
                "database": str(row[0]),
                "postgresql_version": str(row[1]),
                "pgvector_version": str(row[2]),
            }

    def create_schema(self) -> None:
        """Create chunks table with vector column, provenance fields, and metadata indexes."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        chapter TEXT NOT NULL,
                        section TEXT NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        embedding vector({self.dimensions}) NOT NULL,
                        content_hash TEXT NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (strategy, chunk_id)
                    );
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_strat_doc ON chunks (strategy, document_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_strat_chap ON chunks (strategy, chapter);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_strat_sec ON chunks (strategy, section);")
            conn.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Idempotently upsert chunks & vectors for a strategy and prune obsolete chunks per document."""
        if not strategy or not strategy.strip():
            raise ValueError("strategy must be non-empty.")
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunk/vector count mismatch: {len(chunks)} != {len(vectors)}")
        if not chunks:
            self.last_upsert_stats = {"upserted": 0, "deleted_obsolete": 0}
            return

        rows: list[tuple[Any, ...]] = []
        doc_chunk_ids: dict[str, list[str]] = {}

        for chunk, vec in zip(chunks, vectors, strict=True):
            vec_lit = _to_pgvector(vec, self.dimensions)
            chapter = str(chunk.metadata.get("chapter", ""))
            section = str(chunk.metadata.get("section", ""))
            chash = _hash_chunk(chunk)
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chapter,
                    section,
                    chunk.text,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    vec_lit,
                    chash,
                )
            )
            doc_chunk_ids.setdefault(chunk.document_id, []).append(chunk.chunk_id)

        deleted_total = 0
        with self._connect() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO chunks (
                            strategy, chunk_id, document_id, source, page,
                            chapter, section, text, metadata, embedding, content_hash, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, %s, NOW())
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            source = EXCLUDED.source,
                            page = EXCLUDED.page,
                            chapter = EXCLUDED.chapter,
                            section = EXCLUDED.section,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            content_hash = EXCLUDED.content_hash,
                            updated_at = NOW();
                        """,
                        rows,
                    )
                    for doc_id, keep_ids in doc_chunk_ids.items():
                        cur.execute(
                            """
                            DELETE FROM chunks
                            WHERE strategy = %s
                              AND document_id = %s
                              AND NOT (chunk_id = ANY(%s));
                            """,
                            (strategy, doc_id, keep_ids),
                        )
                        deleted_total += cur.rowcount or 0

        self.last_upsert_stats = {
            "upserted": len(rows),
            "deleted_obsolete": deleted_total,
        }

    def count_chunks(self, strategy: str, document_id: str | None = None) -> int:
        """Return persisted chunk count for a strategy (and optional document_id)."""
        with self._connect() as conn, conn.cursor() as cur:
            if document_id is not None:
                cur.execute(
                    "SELECT COUNT(*) FROM chunks WHERE strategy = %s AND document_id = %s;",
                    (strategy, document_id),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE strategy = %s;", (strategy,))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def fetch_chunks_by_page(self, strategy: str, page: int) -> list[dict[str, Any]]:
        """Retrieve stored chunks for a given strategy and page in chunk order."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, document_id, source, page, chapter, section, text, metadata
                FROM chunks
                WHERE strategy = %s AND page = %s
                ORDER BY chunk_id ASC;
                """,
                (strategy, page),
            )
            results: list[dict[str, Any]] = []
            for r in cur.fetchall():
                results.append(
                    {
                        "chunk_id": r[0],
                        "document_id": r[1],
                        "source": r[2],
                        "page": r[3],
                        "chapter": r[4],
                        "section": r[5],
                        "text": r[6],
                        "metadata": r[7],
                    }
                )
            return results

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter allowlisted metadata in SQL, rank by exact cosine distance, return top-k."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        vec_lit = _to_pgvector(query_vector, self.dimensions)

        where_clauses = ["strategy = %s"]
        filter_params: list[Any] = [strategy]

        if filters:
            for key, val in filters.items():
                if key not in ALLOWED_FILTER_KEYS:
                    raise ValueError(f"Unsupported filter key: {key!r}")
                if key == "chapter":
                    where_clauses.append("chapter = %s")
                    filter_params.append(str(val))
                elif key == "section":
                    where_clauses.append("section = %s")
                    filter_params.append(str(val))
                elif key == "document_id":
                    where_clauses.append("document_id = %s")
                    filter_params.append(str(val))
                elif key == "page":
                    where_clauses.append("page = %s")
                    filter_params.append(int(val))
                elif key == "chapter_number":
                    where_clauses.append("(metadata->>'chapter_number')::int = %s")
                    filter_params.append(int(val))

        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT
                chunk_id,
                document_id,
                source,
                page,
                chapter,
                section,
                text,
                metadata,
                (embedding <=> %s::vector) AS cosine_distance
            FROM chunks
            WHERE {where_sql}
            ORDER BY (embedding <=> %s::vector) ASC, chunk_id ASC
            LIMIT %s;
        """
        params: list[Any] = [vec_lit, *filter_params, vec_lit, top_k]

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        hits: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            dist = float(row[8])
            hits.append(
                {
                    "rank": rank,
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "source": row[2],
                    "page": int(row[3]),
                    "chapter": row[4],
                    "section": row[5],
                    "text": row[6],
                    "metadata": row[7],
                    "cosine_distance": round(dist, 6),
                    "score": round(1.0 - dist, 6),
                }
            )
        return hits
