"""Multimodal Hybrid Retrieval (PostgreSQL pgvector + BM25 + RRF + Rerank) for the Advanced 186-page corpus."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg

from .advanced_ingest import ASSETS_JSON_PATH, CHUNKS_JSON_PATH, _vec_literal
from .config import STARTER_DIR, AppConfig, get_config
from .day6_adapter import QueryEmbedderClient

REPO_ROOT = STARTER_DIR.parent
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokenize_mm(text: str) -> list[str]:
    base = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for tok in base:
        out.append(tok)
        # Also emit sub-parts for dotted/hyphenated tokens (e.g. "spm.1" -> "spm", "1")
        parts = re.split(r"[._-]+", tok)
        if len(parts) > 1:
            out.extend(p for p in parts if p)
    return out


class AdvancedCorpusStore:
    """PostgreSQL + JSON store wrapper for 186-page IPCC AR6 SYR multimodal corpus."""

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()
        self.chunks: list[dict[str, Any]] = json.loads(CHUNKS_JSON_PATH.read_text(encoding="utf-8"))
        self.chunks_by_id: dict[str, dict[str, Any]] = {c["id"]: c for c in self.chunks}
        self.assets: list[dict[str, Any]] = json.loads(ASSETS_JSON_PATH.read_text(encoding="utf-8"))
        self.assets_by_id: dict[str, dict[str, Any]] = {a["asset_id"]: a for a in self.assets}
        self.chunks_by_page: dict[int, list[dict[str, Any]]] = {}
        self.chunks_by_asset: dict[str, list[dict[str, Any]]] = {}
        for c in self.chunks:
            self.chunks_by_page.setdefault(int(c["pdf_page"]), []).append(c)
            if c.get("asset_id"):
                self.chunks_by_asset.setdefault(str(c["asset_id"]), []).append(c)

    def fetch_dense_candidates(
        self,
        query_vec: list[float],
        top_k: int = 20,
        modality: str | None = None,
    ) -> list[dict[str, Any]]:
        q_lit = _vec_literal(query_vec)
        where_clauses = ["1=1"]
        params: list[Any] = [q_lit]
        if modality and modality != "all":
            where_clauses.append("modality = %s")
            params.append(modality)
        params.append(q_lit)
        params.append(top_k)

        sql = f"""
            SELECT id, pdf_page, printed_page, section, modality, asset_id, title, content,
                   crop_path, page_render_path, metadata,
                   1.0 - (embedding <=> %s::vector) AS score
            FROM advanced_chunks
            WHERE {' AND '.join(where_clauses)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        with psycopg.connect(self.cfg.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            results.append(
                {
                    "id": r[0],
                    "pdf_page": int(r[1]),
                    "printed_page": r[2],
                    "section": r[3],
                    "modality": r[4],
                    "asset_id": r[5],
                    "title": r[6],
                    "content": r[7],
                    "crop_path": r[8],
                    "page_render_path": r[9],
                    "metadata": r[10] if isinstance(r[10], dict) else {},
                    "score": float(r[11]),
                }
            )
        return results

    def read_page(self, pdf_page: int, include_neighbor_captions: bool = True) -> dict[str, Any]:
        pages = [pdf_page]
        if include_neighbor_captions:
            if pdf_page > 1:
                pages.append(pdf_page - 1)
            pages.append(pdf_page + 1)

        primary_chunks = self.chunks_by_page.get(pdf_page, [])
        # Also locate assets whose caption_pages or graphic_pages include pdf_page
        linked_assets: list[dict[str, Any]] = []
        seen_a: set[str] = set()
        for a in self.assets:
            if pdf_page in a["graphic_pages"] or pdf_page in a["caption_pages"]:
                if a["asset_id"] not in seen_a:
                    seen_a.add(a["asset_id"])
                    linked_assets.append(
                        {
                            "asset_id": a["asset_id"],
                            "title": a["title"],
                            "modality": a["modality"],
                            "graphic_pages": a["graphic_pages"],
                            "caption_pages": a["caption_pages"],
                            "crop_paths": a["crop_paths"],
                            "chunk_ids": [c["id"] for c in self.chunks_by_asset.get(a["asset_id"], [])],
                        }
                    )

        return {
            "pdf_page": pdf_page,
            "chunks": primary_chunks,
            "linked_assets": linked_assets,
        }


class AdvancedBM25Index:
    """In-memory Okapi BM25 over all multimodal chunks (text + table + chart + diagram_map)."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.docs_tf: list[Counter[str]] = []
        self.docs_len: list[int] = []
        df: Counter[str] = Counter()

        for c in chunks:
            full = f"{c.get('asset_id') or ''} {c.get('title') or ''} {c.get('section') or ''} {c['content']}"
            toks = tokenize_mm(full)
            tf = Counter(toks)
            self.docs_tf.append(tf)
            self.docs_len.append(len(toks))
            df.update(tf.keys())

        n_docs = len(chunks)
        self.avg_len = (sum(self.docs_len) / n_docs) if n_docs else 1.0
        self.idf = {
            tok: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }

    def search(self, query: str, top_k: int = 20, modality: str | None = None) -> list[dict[str, Any]]:
        q_toks = tokenize_mm(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, c in enumerate(self.chunks):
            if modality and modality != "all" and c["modality"] != modality:
                continue
            tf = self.docs_tf[idx]
            dl = self.docs_len[idx] or 1
            denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avg_len))
            s = 0.0
            for t in q_toks:
                f = tf.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * (f * (self.k1 + 1.0)) / (f + denom_norm)
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [{**c, "score": float(s)} for s, c in scored[:top_k]]


class AdvancedHybridRetriever:
    """Dense + BM25 RRF fusion across text, tables, charts, and diagrams."""

    def __init__(
        self,
        store: AdvancedCorpusStore,
        embedder: QueryEmbedderClient,
        bm25: AdvancedBM25Index | None = None,
        candidate_k: int = 25,
        rrf_k: int = 60,
        mode: str = "hybrid",  # 'dense' | 'bm25' | 'hybrid'
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25 or AdvancedBM25Index(store.chunks)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.mode = mode

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        modality: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self.mode == "bm25":
            hits = self.bm25.search(query, top_k=top_k, modality=modality)
            return hits, {"embed_tokens": 0, "embed_cost_usd": 0.0, "reference_embed_cost_usd": 0.0}

        q_vec, q_meta = self.embedder.embed_query(query)
        meta = {
            "embed_tokens": int(q_meta.get("tokens", 0)),
            "embed_cost_usd": float(q_meta.get("cost_usd", 0.0)),
            "reference_embed_cost_usd": float(q_meta.get("reference_cost_usd", 0.0)),
        }
        if self.mode == "dense":
            hits = self.store.fetch_dense_candidates(q_vec, top_k=top_k, modality=modality)
            return hits, meta

        dense_hits = self.store.fetch_dense_candidates(q_vec, top_k=self.candidate_k, modality=modality)
        bm25_hits = self.bm25.search(query, top_k=self.candidate_k, modality=modality)

        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(dense_hits, 1):
            cid = item["id"]
            by_id[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, item in enumerate(bm25_hits, 1):
            cid = item["id"]
            if cid not in by_id:
                by_id[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        results = [{**by_id[cid], "score": round(score, 6)} for cid, score in ordered]
        return results, meta


def pack_multimodal_context(
    candidates: list[dict[str, Any]],
    budget_chars: int = 6500,
) -> tuple[str, list[str]]:
    """Pack retrieved multimodal chunks preserving asset_id, pdf_page, crop_path provenance."""
    parts: list[str] = []
    used_ids: list[str] = []
    seen: set[str] = set()
    used_chars = 0

    for c in candidates:
        cid = c["id"]
        if cid in seen:
            continue
        asset_tag = f" | asset={c['asset_id']}" if c.get("asset_id") else ""
        crop_tag = f" | crop={c['crop_path']}" if c.get("crop_path") else ""
        header = f"[{cid} | {c['modality']} | {c['section']} | PDF p.{c['pdf_page']}{asset_tag}{crop_tag}]"
        block = f"{header}\n{c['content']}"
        if len(parts) > 0 and used_chars + len(block) + 2 > budget_chars:
            continue
        if len(parts) == 0 and len(block) > budget_chars:
            block = block[:budget_chars]
        parts.append(block)
        used_ids.append(cid)
        seen.add(cid)
        used_chars += len(block) + 2

    return "\n\n".join(parts), used_ids


def evaluate_advanced_retrieval(
    split: str = "dev",
    top_k: int = 6,
    out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Benchmark Dense, BM25, and Hybrid RRF retrieval on answerable Advanced split queries."""
    import statistics
    import time

    from .config import DATA_DIR, RESULTS_DIR

    cfg = get_config()
    store = AdvancedCorpusStore(cfg=cfg)
    bm25 = AdvancedBM25Index(store.chunks)
    embedder = QueryEmbedderClient(cfg=cfg, use_cache=True)

    all_cases = json.loads((DATA_DIR / "advanced_benchmark.json").read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c["split"] == split and c["answerable"]]
    out_target = out_dir or (RESULTS_DIR / "advanced" / "retrieval")
    out_target.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for mode in ("dense", "bm25", "hybrid"):
        ret = AdvancedHybridRetriever(store=store, embedder=embedder, bm25=bm25, mode=mode)
        latencies: list[float] = []
        page_hits = 0
        asset_hits = 0
        asset_cases = 0
        total_embed_tokens = 0
        t_all = time.perf_counter()

        for c in cases:
            t0 = time.perf_counter()
            hits, meta = ret.retrieve(c["question"], top_k=top_k, modality="all")
            latencies.append((time.perf_counter() - t0) * 1000.0)
            total_embed_tokens += int(meta.get("embed_tokens", 0))

            gold_pages = set(int(p) for p in c.get("gold_pages") or [])
            ret_pages = {int(h["pdf_page"]) for h in hits}
            if gold_pages & ret_pages:
                page_hits += 1
            else:
                # Also match asset caption/graphic pages stored in metadata
                ret_asset_pages = set()
                for h in hits:
                    md = h.get("metadata") or {}
                    ret_asset_pages.update(int(p) for p in md.get("graphic_pages", []))
                    ret_asset_pages.update(int(p) for p in md.get("caption_pages", []))
                if gold_pages & ret_asset_pages:
                    page_hits += 1

            gold_assets = set(c.get("gold_assets") or [])
            if gold_assets:
                asset_cases += 1
                ret_assets = {str(h.get("asset_id")) for h in hits if h.get("asset_id")}
                if gold_assets & ret_assets:
                    asset_hits += 1

        summaries.append(
            {
                "mode": mode,
                "split": split,
                "top_k": top_k,
                "n_answerable": len(cases),
                "page_hit_n": page_hits,
                "page_hit_rate": round(page_hits / len(cases), 4) if cases else 0.0,
                "asset_cases": asset_cases,
                "asset_hit_n": asset_hits,
                "asset_hit_rate": round(asset_hits / asset_cases, 4) if asset_cases else 0.0,
                "median_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
                "total_seconds": round(time.perf_counter() - t_all, 3),
                "embed_tokens": total_embed_tokens,
            }
        )

    (out_target / f"summary_{split}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summaries


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Benchmark Advanced Multimodal Retrieval")
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--top-k", type=int, default=6)
    args = p.parse_args()
    res = evaluate_advanced_retrieval(split=args.split, top_k=args.top_k)
    print(json.dumps(res, ensure_ascii=False, indent=2))

