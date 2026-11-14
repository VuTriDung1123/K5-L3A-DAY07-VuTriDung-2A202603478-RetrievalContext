"""Day 6 PostgreSQL + pgvector adapter and OpenRouter query embedder for Day 7."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import psycopg

from .config import REPO_ROOT, AppConfig, get_config

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starter.src.embeddings import OpenRouterEmbedder  # noqa: E402
from starter.src.store import ALLOWED_FILTER_KEYS, _to_pgvector  # noqa: E402


class QueryEmbedderClient:
    """Wrapper over OpenRouterEmbedder with optional deterministic disk cache & cost tracking."""

    def __init__(
        self,
        cfg: AppConfig | None = None,
        cache_path: Path | None = None,
        use_cache: bool = True,
    ) -> None:
        self.cfg = cfg or get_config()
        self.embedder = OpenRouterEmbedder(
            model_name=self.cfg.embedding_model,
            dimensions=self.cfg.embedding_dimensions,
        )
        self.cache_path = cache_path or (REPO_ROOT / "starter" / "data" / "cache" / "query_embeddings.json")
        self.use_cache = use_cache
        self._cache: dict[str, dict[str, Any]] = {}
        if self.use_cache and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

        # Cumulative live call stats across session
        self.cache_hits = 0
        self.live_calls = 0
        self.live_tokens = 0
        self.live_cost_usd = 0.0
        self.live_seconds = 0.0

    def _cache_key(self, text: str) -> str:
        raw = f"{self.cfg.embedding_model}|{self.cfg.embedding_dimensions}|{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def embed_query(self, text: str, bypass_cache: bool = False) -> tuple[list[float], dict[str, Any]]:
        """Return (vector, telemetry) for query text using OpenRouter text-embedding-3-small."""
        clean = text.strip()
        if not clean:
            raise ValueError("Query text must not be empty.")
        key = self._cache_key(clean)
        if self.use_cache and not bypass_cache and key in self._cache:
            self.cache_hits += 1
            entry = self._cache[key]
            return list(entry["vector"]), {
                "cached": True,
                "tokens": entry.get("tokens", 0),
                "cost_usd": 0.0,
                "reference_cost_usd": entry.get("cost_usd", 0.0),
                "seconds": 0.0,
            }

        t0 = time.perf_counter()
        prev_tokens = self.embedder.query_tokens
        prev_cost = self.embedder.query_cost_usd
        vec = self.embedder.embed_query(clean)
        dt = time.perf_counter() - t0
        tokens = self.embedder.query_tokens - prev_tokens
        cost = self.embedder.query_cost_usd - prev_cost

        self.live_calls += 1
        self.live_tokens += tokens
        self.live_cost_usd += cost
        self.live_seconds += dt

        if self.use_cache:
            self._cache[key] = {
                "model": self.cfg.embedding_model,
                "dimensions": self.cfg.embedding_dimensions,
                "text": clean,
                "tokens": tokens,
                "cost_usd": cost,
                "vector": vec,
            }
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False),
                encoding="utf-8",
            )

        return vec, {
            "cached": False,
            "tokens": tokens,
            "cost_usd": round(cost, 8),
            "reference_cost_usd": round(cost, 8),
            "seconds": round(dt, 4),
        }


class Day6VectorDB:
    """Read-only adapter over Day 6 PostgreSQL + pgvector table `chunks`."""

    def __init__(self, cfg: AppConfig | None = None, strategy: str = "B") -> None:
        self.cfg = cfg or get_config()
        self.strategy = strategy
        self.database_url = self.cfg.database_url
        self.dimensions = self.cfg.embedding_dimensions

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self.database_url, connect_timeout=10)

    def info(self) -> dict[str, Any]:
        """Inspect Day 6 database version, strategy chunk counts, and vector dims."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT current_database(), current_setting('server_version'), extversion "
                "FROM pg_extension WHERE extname = 'vector'"
            )
            row = cur.fetchone()
            cur.execute("SELECT strategy, COUNT(*), MIN(page), MAX(page) FROM chunks GROUP BY strategy ORDER BY strategy")
            strategies = [
                {"strategy": r[0], "count": int(r[1]), "min_page": int(r[2]), "max_page": int(r[3])}
                for r in cur.fetchall()
            ]
            cur.execute("SELECT DISTINCT vector_dims(embedding) FROM chunks")
            dims = [int(r[0]) for r in cur.fetchall()]
        return {
            "database": str(row[0]) if row else "",
            "postgresql_version": str(row[1]) if row else "",
            "pgvector_version": str(row[2]) if row else "",
            "vector_dimensions": dims,
            "strategies": strategies,
        }

    def export_chunks(self, strategy: str | None = None) -> list[dict[str, Any]]:
        """Export Day 6 chunks into Day 7 JSON contract dicts."""
        strat = strategy or self.strategy
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, document_id, source, page, chapter, section, text, metadata
                FROM chunks
                WHERE strategy = %s
                ORDER BY page ASC, chunk_id ASC;
                """,
                (strat,),
            )
            rows = cur.fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            meta = dict(r[7] or {})
            meta.setdefault("chapter", r[4])
            meta.setdefault("section", r[5])
            out.append(
                {
                    "id": r[0],
                    "source_id": r[1],
                    "source": r[2],
                    "page": int(r[3]),
                    "text": r[6],
                    "metadata": meta,
                }
            )
        return out

    def search_dense(
        self,
        query_vector: list[float],
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run exact cosine search against Day 6 pgvector chunks with optional metadata filter."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        strat = strategy or self.strategy
        vec_lit = _to_pgvector(query_vector, self.dimensions)

        where_clauses = ["strategy = %s"]
        filter_params: list[Any] = [strat]
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
                chunk_id, document_id, source, page, chapter, section, text, metadata,
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

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            dist = float(row[8])
            meta = dict(row[7] or {})
            meta.setdefault("chapter", row[4])
            meta.setdefault("section", row[5])
            results.append(
                {
                    "rank": rank,
                    "id": row[0],
                    "source_id": row[1],
                    "source": row[2],
                    "page": int(row[3]),
                    "text": row[6],
                    "metadata": meta,
                    "score": round(1.0 - dist, 6),
                    "cosine_distance": round(dist, 6),
                    "strategy": f"dense:{strat}",
                }
            )
        return results

    def fetch_page_chunks(
        self,
        page: int,
        source_id: str = "openstax-pds-2025",
        strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all chunks on a given page in reading order."""
        strat = strategy or self.strategy
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, document_id, source, page, chapter, section, text, metadata
                FROM chunks
                WHERE strategy = %s AND document_id = %s AND page = %s
                ORDER BY chunk_id ASC;
                """,
                (strat, source_id, int(page)),
            )
            rows = cur.fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            meta = dict(r[7] or {})
            meta.setdefault("chapter", r[4])
            meta.setdefault("section", r[5])
            out.append(
                {
                    "id": r[0],
                    "source_id": r[1],
                    "source": r[2],
                    "page": int(r[3]),
                    "text": r[6],
                    "metadata": meta,
                }
            )
        return out
