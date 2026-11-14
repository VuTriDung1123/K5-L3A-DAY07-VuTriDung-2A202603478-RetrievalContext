"""Benchmark retrieval methods and context strategies on a locked dev or test split."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any

from .config import DATA_DIR, RESULTS_DIR, STARTER_DIR, get_config
from .day6_adapter import Day6VectorDB, QueryEmbedderClient
from .retrievers import (
    BM25Retriever,
    DenseRetriever,
    GeminiReranker,
    HybridRerankRetriever,
    HybridRRFRetriever,
    matches_filters,
    pack_context_expanded,
    pack_context_greedy,
)


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip()


def strict_chunk_hit(chunk: dict[str, Any], q: dict[str, Any]) -> bool:
    """Return True iff single chunk matches gold source_id, page, and ALL required clauses."""
    if chunk["source_id"] != q["gold_source_id"] or int(chunk["page"]) != int(q["gold_page"]):
        return False
    norm_txt = _norm_ws(chunk["text"])
    return all(_norm_ws(cl) in norm_txt for cl in q["required_clauses"])


def clauses_covered(chunks: list[dict[str, Any]], q: dict[str, Any]) -> bool:
    """Return True iff all required clauses appear on gold source_id + page across `chunks`."""
    matching = [
        _norm_ws(c["text"])
        for c in chunks
        if c["source_id"] == q["gold_source_id"] and int(c["page"]) == int(q["gold_page"])
    ]
    if not matching:
        return False
    return all(any(_norm_ws(cl) in txt for txt in matching) for cl in q["required_clauses"])


def evaluate_config(
    config_name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    all_chunks_by_page: dict[tuple[str, int], list[dict[str, Any]]],
    chunks_by_id: dict[str, dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 800,
    expand_neighbors: bool = False,
    greedy_only: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one retrieval + context config over all answerable questions in split."""
    traces: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for q in questions:
        filters = q.get("filter")
        t0 = time.perf_counter()
        results, rmeta = retriever.search(q["question"], filters=filters, top_k=top_k)

        if greedy_only:
            context, context_ids = pack_context_greedy(results, budget=budget)
        else:
            context, context_ids = pack_context_expanded(
                results,
                all_chunks_by_page,
                budget=budget,
                expand_neighbors=expand_neighbors,
                max_primary=top_k,
                neighbor_radius=1,
            )
        total_ms = (time.perf_counter() - t0) * 1000.0

        packed_chunks = [chunks_by_id[cid] for cid in context_ids if cid in chunks_by_id]
        hit1 = bool(results and strict_chunk_hit(results[0], q))
        hit3 = any(strict_chunk_hit(r, q) for r in results[:3])
        union3 = clauses_covered(results[:3], q)
        context_hit = clauses_covered(packed_chunks, q)

        first_strict_rank = next(
            (idx for idx, r in enumerate(results[:3], start=1) if strict_chunk_hit(r, q)),
            None,
        )
        mrr3 = (1.0 / first_strict_rank) if first_strict_rank else 0.0
        filter_ok = True if not filters else all(matches_filters(r, filters) for r in results)

        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "config": config_name,
                "question": q["question"],
                "filter": filters,
                "gold_source_id": q["gold_source_id"],
                "gold_page": q["gold_page"],
                "required_clauses": q["required_clauses"],
                "results": [
                    {
                        "rank": r.get("rank", i + 1),
                        "id": r["id"],
                        "page": r["page"],
                        "chapter": (r.get("metadata") or {}).get("chapter"),
                        "section": (r.get("metadata") or {}).get("section"),
                        "score": r.get("score"),
                        "strict_hit": strict_chunk_hit(r, q),
                    }
                    for i, r in enumerate(results)
                ],
                "context_ids": context_ids,
                "context_chars": len(context),
                "context": context,
                "hit1": hit1,
                "hit3": hit3,
                "union3": union3,
                "context_hit": context_hit,
                "mrr3": mrr3,
                "filter_ok": filter_ok,
                "elapsed_ms": round(total_ms, 2),
                "meta": rmeta,
            }
        )

    total_seconds = time.perf_counter() - t_all
    n = len(traces)
    filter_traces = [t for t in traces if t["filter"]]

    embed_ref_cost = sum(float(t["meta"].get("embed_reference_cost_usd", 0.0)) for t in traces)
    embed_live_cost = sum(float(t["meta"].get("embed_cost_usd", 0.0)) for t in traces)
    rerank_cost = sum(float(t["meta"].get("rerank_cost_usd", 0.0)) for t in traces)
    embed_tokens = sum(int(t["meta"].get("embed_tokens", 0)) for t in traces)
    rerank_tokens = sum(int(t["meta"].get("rerank_tokens", 0)) for t in traces)

    summary = {
        "config": config_name,
        "split": questions[0]["split"] if questions else "",
        "method": getattr(retriever, "name", config_name),
        "top_k": top_k,
        "budget_chars": budget,
        "expand_neighbors": expand_neighbors,
        "n": n,
        "hit1_n": sum(t["hit1"] for t in traces),
        "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 4) if n else 0.0,
        "hit3_n": sum(t["hit3"] for t in traces),
        "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 4) if n else 0.0,
        "union3_n": sum(t["union3"] for t in traces),
        "union3_rate": round(sum(t["union3"] for t in traces) / n, 4) if n else 0.0,
        "context_hit_n": sum(t["context_hit"] for t in traces),
        "context_hit_rate": round(sum(t["context_hit"] for t in traces) / n, 4) if n else 0.0,
        "mrr3": round(sum(t["mrr3"] for t in traces) / n, 4) if n else 0.0,
        "filter_compliance": f"{sum(t['filter_ok'] for t in filter_traces)}/{len(filter_traces)}",
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if n else 0.0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2) if n else 0.0,
        "total_seconds": round(total_seconds, 3),
        "embed_tokens": embed_tokens,
        "rerank_tokens": rerank_tokens,
        "embed_reference_cost_usd": round(embed_ref_cost, 8),
        "embed_live_cost_usd": round(embed_live_cost, 8),
        "rerank_cost_usd": round(rerank_cost, 8),
        "total_reference_cost_usd": round(embed_ref_cost + rerank_cost, 8),
    }
    return summary, traces


def run_split(split: str, configs: list[str], out_dir: Path) -> None:
    cfg = get_config()
    chunks = json.loads((STARTER_DIR / "data" / "day6_exported" / "chunks_b.json").read_text(encoding="utf-8"))
    all_questions = json.loads((DATA_DIR / "basic_retrieval_questions.json").read_text(encoding="utf-8"))
    # Strict split isolation: never touch the other split!
    questions = [q for q in all_questions if q["split"] == split]
    if not questions:
        raise ValueError(f"No questions found for split={split}")

    chunks_by_id = {c["id"]: c for c in chunks}
    chunks_by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for c in chunks:
        chunks_by_page.setdefault((c["source_id"], int(c["page"])), []).append(c)
    for k in chunks_by_page:
        chunks_by_page[k].sort(key=lambda x: str(x["id"]))

    db = Day6VectorDB(cfg=cfg, strategy="B")
    embedder = QueryEmbedderClient(cfg=cfg, use_cache=True)
    dense = DenseRetriever(db=db, embedder=embedder, strategy="B")
    bm25 = BM25Retriever(chunks=chunks, k1=1.5, b=0.75, include_section_title=True)
    hybrid = HybridRRFRetriever(dense=dense, bm25=bm25, candidate_k=20, rrf_k=60)
    reranker = GeminiReranker(cfg=cfg)
    hybrid_rerank = HybridRerankRetriever(hybrid=hybrid, reranker=reranker, rerank_pool=12)

    registry: dict[str, tuple[Any, dict[str, Any]]] = {
        "baseline_dense": (dense, {"top_k": 3, "budget": 800, "expand_neighbors": False, "greedy_only": True}),
        "method2_bm25": (bm25, {"top_k": 3, "budget": 800, "expand_neighbors": False, "greedy_only": True}),
        "method3_hybrid_rrf": (hybrid, {"top_k": 3, "budget": 800, "expand_neighbors": False, "greedy_only": True}),
        "imp1_hybrid_expanded": (hybrid, {"top_k": 3, "budget": 2400, "expand_neighbors": True, "greedy_only": False}),
        "imp2_hybrid_rerank_expanded": (hybrid_rerank, {"top_k": 3, "budget": 2400, "expand_neighbors": True, "greedy_only": False}),
    }

    summaries: list[dict[str, Any]] = []
    all_traces: dict[str, list[dict[str, Any]]] = {}
    for cname in configs:
        if cname not in registry:
            raise ValueError(f"Unknown config {cname!r}")
        ret, params = registry[cname]
        summary, traces = evaluate_config(
            cname,
            ret,
            questions,
            chunks_by_page,
            chunks_by_id,
            **params,
        )
        summaries.append(summary)
        all_traces[cname] = traces
        print(
            f"[{split}] {cname:28s} | Hit@1={summary['hit1_n']}/{summary['n']} ({summary['hit1_rate']*100:.1f}%) "
            f"| Hit@3={summary['hit3_n']}/{summary['n']} ({summary['hit3_rate']*100:.1f}%) "
            f"| Union@3={summary['union3_n']}/{summary['n']} "
            f"| CtxEv={summary['context_hit_n']}/{summary['n']} ({summary['context_hit_rate']*100:.1f}%) "
            f"| MRR@3={summary['mrr3']:.3f} | AvgChars={summary['avg_context_chars']} "
            f"| {summary['total_seconds']}s | ${summary['total_reference_cost_usd']:.6f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"summary_{split}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / f"traces_{split}.json").write_text(
        json.dumps(all_traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / f"summary_{split}.csv").open("w", newline="", encoding="utf-8") as fh:
        if summaries:
            writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Day 7 retrieval configurations")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "baseline_dense",
            "method2_bm25",
            "method3_hybrid_rrf",
            "imp1_hybrid_expanded",
            "imp2_hybrid_rerank_expanded",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "basic" / "retrieval",
    )
    args = parser.parse_args()
    run_split(args.split, args.configs, args.output)


if __name__ == "__main__":
    main()

