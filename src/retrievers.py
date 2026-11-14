"""Retrieval methods and context packing strategies for Day 7."""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import Counter
from typing import Any

import requests

from .config import AppConfig, estimate_llm_cost_usd, get_config
from .day6_adapter import Day6VectorDB, QueryEmbedderClient


def tokenize(text: str) -> list[str]:
    """Normalize Unicode NFC and extract lowercase alphanumeric word tokens."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text or "").casefold())


def matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Apply allowlisted metadata filter checks uniformly across in-memory retrievers."""
    if not filters:
        return True
    meta = chunk.get("metadata") or {}
    for key, val in filters.items():
        if key == "chapter":
            if str(meta.get("chapter", "")) != str(val):
                return False
        elif key == "section":
            if str(meta.get("section", "")) != str(val):
                return False
        elif key == "document_id":
            if str(chunk.get("source_id", "")) != str(val):
                return False
        elif key == "page":
            if int(chunk.get("page", -1)) != int(val):
                return False
        elif key == "chapter_number":
            if int(meta.get("chapter_number", -1)) != int(val):
                return False
        else:
            raise ValueError(f"Unsupported filter key: {key!r}")
    return True


def format_chunk_block(chunk: dict[str, Any]) -> str:
    """Format a chunk with verifiable source, page, and section metadata."""
    meta = chunk.get("metadata") or {}
    sec = meta.get("section", "")
    sec_part = f" | section {sec}" if sec else ""
    return f"[{chunk['id']} | {chunk['source']} | page {chunk['page']}{sec_part}]\n{chunk['text']}"


def pack_context_greedy(results: list[dict[str, Any]], budget: int = 800) -> tuple[str, list[str]]:
    """Baseline greedy deduplicating context packer matching starter/baseline.py header budget."""
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for c in results:
        cid = c["id"]
        if cid in seen:
            continue
        seen.add(cid)
        block = f"[{cid} | {c['source']} | page {c['page']}]\n{c['text']}"
        candidate = "\n\n".join(blocks + [block])
        if len(candidate) <= budget:
            blocks.append(block)
            ids.append(cid)
    return "\n\n".join(blocks), ids


def pack_context_expanded(
    results: list[dict[str, Any]],
    all_chunks_by_page: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    budget: int = 2400,
    expand_neighbors: bool = True,
    max_primary: int = 3,
    neighbor_radius: int = 1,
) -> tuple[str, list[str]]:
    """Pack top results plus adjacent same-page chunks in reading order within char budget.

    Preserves primary hit priority so top-ranked pages stay within budget, then adds
    adjacent same-page chunks (`+-neighbor_radius`) to prevent split-list/split-sentence truncation.
    """
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def try_add(chunk: dict[str, Any]) -> bool:
        cid = chunk["id"]
        if cid in seen_ids:
            return True
        test_list = selected + [chunk]
        test_list_sorted = sorted(
            test_list,
            key=lambda x: (x["source_id"], int(x["page"]), str(x["id"])),
        )
        rendered = "\n\n".join(format_chunk_block(x) for x in test_list_sorted)
        if len(rendered) <= budget:
            selected.append(chunk)
            seen_ids.add(cid)
            return True
        return False

    # First ensure top primary results fit
    primaries = results[:max_primary]
    for r in primaries:
        try_add(r)

    # Expand immediate same-page neighbors around top primary hits
    if expand_neighbors:
        for r in primaries:
            page_list = all_chunks_by_page.get((r["source_id"], int(r["page"])), [])
            pos = next((i for i, item in enumerate(page_list) if item["id"] == r["id"]), -1)
            if pos != -1:
                for offset in range(1, neighbor_radius + 1):
                    if pos + offset < len(page_list):
                        try_add(page_list[pos + offset])
                    if pos - offset >= 0:
                        try_add(page_list[pos - offset])

    # Fill any remaining budget with remaining ranked hits
    for r in results[max_primary:]:
        try_add(r)

    ordered = sorted(selected, key=lambda x: (x["source_id"], int(x["page"]), str(x["id"])))
    context = "\n\n".join(format_chunk_block(x) for x in ordered)
    ids = [x["id"] for x in ordered]
    return context, ids


class DenseRetriever:
    """Method 1: Dense cosine retrieval using Day 6 pgvector + OpenRouter embeddings."""

    name = "dense"

    def __init__(
        self,
        db: Day6VectorDB,
        embedder: QueryEmbedderClient,
        strategy: str = "B",
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.strategy = strategy

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        t0 = time.perf_counter()
        qvec, emb_info = self.embedder.embed_query(query)
        hits = self.db.search_dense(qvec, top_k=top_k, filters=filters, strategy=self.strategy)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return hits, {
            "method": self.name,
            "elapsed_ms": round(elapsed_ms, 2),
            "embed_tokens": emb_info["tokens"],
            "embed_cost_usd": emb_info["cost_usd"],
            "embed_reference_cost_usd": emb_info["reference_cost_usd"],
            "embed_cached": emb_info["cached"],
            "rerank_tokens": 0,
            "rerank_cost_usd": 0.0,
        }


class BM25Retriever:
    """Method 2: Okapi BM25 lexical retrieval with metadata filter support."""

    name = "bm25"

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        include_section_title: bool = True,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.include_section_title = include_section_title
        self._doc_tf: list[Counter[str]] = []
        self._doc_len: list[int] = []
        df: Counter[str] = Counter()

        for c in chunks:
            text = c["text"]
            if include_section_title:
                sec_title = str((c.get("metadata") or {}).get("section_title", ""))
                if sec_title:
                    text = f"{sec_title}\n{text}"
            toks = tokenize(text)
            tf = Counter(toks)
            self._doc_tf.append(tf)
            self._doc_len.append(len(toks))
            df.update(tf.keys())

        n_docs = len(chunks)
        self._avgdl = (sum(self._doc_len) / n_docs) if n_docs else 1.0
        self._idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        t0 = time.perf_counter()
        q_terms = tokenize(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, chunk in enumerate(self.chunks):
            if not matches_filters(chunk, filters):
                continue
            tf = self._doc_tf[idx]
            dl = self._doc_len[idx]
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                f = tf[term]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl))
                score += idf * (f * (self.k1 + 1.0)) / (denom or 1.0)
            if score > 0.0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        top = scored[:top_k]
        hits = [
            {
                **c,
                "rank": rank,
                "score": round(sc, 6),
                "strategy": self.name,
            }
            for rank, (sc, c) in enumerate(top, start=1)
        ]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return hits, {
            "method": self.name,
            "elapsed_ms": round(elapsed_ms, 2),
            "embed_tokens": 0,
            "embed_cost_usd": 0.0,
            "embed_reference_cost_usd": 0.0,
            "embed_cached": False,
            "rerank_tokens": 0,
            "rerank_cost_usd": 0.0,
        }


class HybridRRFRetriever:
    """Method 3: Hybrid Dense + BM25 combined with Reciprocal Rank Fusion (RRF)."""

    name = "hybrid_rrf"

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        *,
        candidate_k: int = 20,
        rrf_k: int = 60,
        w_dense: float = 1.0,
        w_bm25: float = 1.0,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.w_dense = w_dense
        self.w_bm25 = w_bm25

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        t0 = time.perf_counter()
        cand_k = max(self.candidate_k, top_k)
        dense_hits, dense_meta = self.dense.search(query, filters=filters, top_k=cand_k)
        bm25_hits, _ = self.bm25.search(query, filters=filters, top_k=cand_k)

        fused: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}

        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            fused[cid] = dict(item)
            fused[cid]["dense_rank"] = rank
            fused[cid]["dense_score"] = item["score"]
            scores[cid] = scores.get(cid, 0.0) + self.w_dense / (self.rrf_k + rank)

        for rank, item in enumerate(bm25_hits, start=1):
            cid = item["id"]
            if cid not in fused:
                fused[cid] = dict(item)
            fused[cid]["bm25_rank"] = rank
            fused[cid]["bm25_score"] = item["score"]
            scores[cid] = scores.get(cid, 0.0) + self.w_bm25 / (self.rrf_k + rank)

        ordered_ids = sorted(scores.keys(), key=lambda cid: (-scores[cid], cid))[:top_k]
        results: list[dict[str, Any]] = []
        for rank, cid in enumerate(ordered_ids, start=1):
            row = fused[cid]
            row["rank"] = rank
            row["score"] = round(scores[cid], 6)
            row["strategy"] = self.name
            results.append(row)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return results, {
            **dense_meta,
            "method": self.name,
            "elapsed_ms": round(elapsed_ms, 2),
        }


class GeminiReranker:
    """Listwise LLM reranker over candidate chunks using Gemini Flash-Lite."""

    def __init__(self, cfg: AppConfig | None = None, model: str | None = None) -> None:
        self.cfg = cfg or get_config()
        if not self.cfg.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY missing for GeminiReranker.")
        self.model = model or self.cfg.rerank_model
        self._session = requests.Session()

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int = 3,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not candidates:
            return [], {"rerank_tokens": 0, "rerank_cost_usd": 0.0, "rerank_ms": 0.0}

        t0 = time.perf_counter()
        cand_lines = []
        by_id = {c["id"]: c for c in candidates}
        for idx, c in enumerate(candidates, start=1):
            snippet = c["text"].replace("\n", " ")[:750]
            cand_lines.append(f"{idx}. ID={c['id']} (page {c['page']}, sec {c.get('metadata', {}).get('section', '')}): {snippet}")

        prompt = (
            "You are a precision passage reranker for retrieval-augmented QA.\n"
            "Given the user question and candidate textbook chunks, rank the candidate IDs from MOST relevant "
            "and complete for answering the question to LEAST relevant.\n"
            "Prefer chunks containing specific definitions, rules, numbered items, numbers, or direct answers "
            "over vague mentions or table/code noise.\n"
            "Return strictly valid JSON: {\"ranked_ids\": [\"id1\", \"id2\", ...]} containing all candidate IDs.\n\n"
            f"Question: {query}\n\n"
            "Candidates:\n" + "\n".join(cand_lines)
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.cfg.gemini_api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }
        last_exc: Exception | None = None
        data: dict[str, Any] | None = None
        for attempt in range(4):
            try:
                resp = self._session.post(url, json=payload, timeout=60)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(f"Gemini rerank HTTP {resp.status_code}: {resp.text[:300]}")
                if resp.status_code >= 400:
                    raise RuntimeError(f"Gemini rerank HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 3:
                    time.sleep(min(2.0 ** attempt, 8.0))
        if data is None:
            raise RuntimeError(f"Gemini rerank failed after retries: {last_exc}") from last_exc
        usage = data.get("usageMetadata") or {}
        in_tok = int(usage.get("promptTokenCount") or 0)
        out_tok = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
        total_tok = int(usage.get("totalTokenCount") or (in_tok + out_tok))
        cost_usd = estimate_llm_cost_usd(self.model, in_tok, out_tok)

        raw_text = ""
        for part in data["candidates"][0]["content"]["parts"]:
            if "text" in part:
                raw_text += part["text"]
        parsed = json.loads(raw_text)
        ranked_ids = [cid for cid in parsed.get("ranked_ids", []) if cid in by_id]

        # Preserve any missing candidate IDs in original RRF order
        seen = set(ranked_ids)
        for c in candidates:
            if c["id"] not in seen:
                ranked_ids.append(c["id"])
                seen.add(c["id"])

        reranked: list[dict[str, Any]] = []
        for rank, cid in enumerate(ranked_ids[:top_k], start=1):
            row = dict(by_id[cid])
            row["rrf_score"] = row.get("score")
            row["rank"] = rank
            row["score"] = round(1.0 / rank, 6)
            row["strategy"] = "hybrid_rrf_rerank"
            reranked.append(row)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return reranked, {
            "rerank_model": self.model,
            "rerank_prompt_tokens": in_tok,
            "rerank_completion_tokens": out_tok,
            "rerank_tokens": total_tok,
            "rerank_cost_usd": cost_usd,
            "rerank_ms": round(dt_ms, 2),
        }


class HybridRerankRetriever:
    """Two-stage retrieval: Hybrid RRF candidate pool -> Gemini listwise reranker."""

    name = "hybrid_rrf_rerank"

    def __init__(
        self,
        hybrid: HybridRRFRetriever,
        reranker: GeminiReranker,
        *,
        rerank_pool: int = 12,
    ) -> None:
        self.hybrid = hybrid
        self.reranker = reranker
        self.rerank_pool = rerank_pool

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        t0 = time.perf_counter()
        pool_k = max(self.rerank_pool, top_k)
        cands, hybrid_meta = self.hybrid.search(query, filters=filters, top_k=pool_k)
        reranked, rr_meta = self.reranker.rerank(query, cands, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return reranked, {
            **hybrid_meta,
            **rr_meta,
            "method": self.name,
            "elapsed_ms": round(elapsed_ms, 2),
            "pool_size": len(cands),
        }
