"""Run real Agentic RAG evaluation (Baseline vs Improved) on dev or held-out test split."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .agent import GeminiAgentRunner
from .config import DATA_DIR, RESULTS_DIR, STARTER_DIR, get_config
from .day6_adapter import Day6VectorDB, QueryEmbedderClient
from .retrievers import (
    BM25Retriever,
    DenseRetriever,
    GeminiReranker,
    HybridRerankRetriever,
    HybridRRFRetriever,
)


def run_agent_eval(split: str, config_name: str, out_dir: Path) -> dict[str, Any]:
    cfg = get_config()
    chunks = json.loads((STARTER_DIR / "data" / "day6_exported" / "chunks_b.json").read_text(encoding="utf-8"))
    all_cases = json.loads((DATA_DIR / "basic_agent_cases.json").read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c["split"] == split]
    if not cases:
        raise ValueError(f"No agent cases for split={split}")

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

    if config_name == "baseline_agent":
        runner = GeminiAgentRunner(
            retriever=dense,
            db=db,
            all_chunks_by_page=chunks_by_page,
            cfg=cfg,
            top_k=3,
            budget=800,
            max_searches=3,
            allow_page_tool=False,
            expand_neighbors=False,
            greedy_only=True,
            config_name="baseline_agent",
        )
    elif config_name == "hybrid_expanded_agent":
        runner = GeminiAgentRunner(
            retriever=hybrid,
            db=db,
            all_chunks_by_page=chunks_by_page,
            cfg=cfg,
            top_k=3,
            budget=2400,
            max_searches=4,
            allow_page_tool=True,
            expand_neighbors=True,
            greedy_only=False,
            config_name="hybrid_expanded_agent",
        )
    elif config_name == "hybrid_rerank_agent":
        reranker = GeminiReranker(cfg=cfg)
        hybrid_rerank = HybridRerankRetriever(hybrid=hybrid, reranker=reranker, rerank_pool=12)
        runner = GeminiAgentRunner(
            retriever=hybrid_rerank,
            db=db,
            all_chunks_by_page=chunks_by_page,
            cfg=cfg,
            top_k=3,
            budget=2400,
            max_searches=4,
            allow_page_tool=True,
            expand_neighbors=True,
            greedy_only=False,
            config_name="hybrid_rerank_agent",
        )
    else:
        raise ValueError(f"Unknown agent config: {config_name!r}")

    traces: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for c in cases:
        tr = runner.run_query(c["id"], c["question"], filters=c.get("filter"))
        tr["split"] = split
        tr["answerable"] = c["answerable"]
        tr["type"] = c.get("type")
        tr["question"] = c["question"]
        traces.append(tr)
        print(
            f"[{split}][{config_name}] {c['id']} (ans={c['answerable']}): "
            f"status={tr['status']}, searches={tr['search_calls']}, "
            f"cites={len(tr['citations'])}, {tr['elapsed_seconds']}s, ${tr.get('estimated_cost_usd', 0):.6f}",
            flush=True,
        )

    total_seconds = round(time.perf_counter() - t_all, 3)
    run_meta = {
        "config": config_name,
        "split": split,
        "model": runner.model,
        "cases": len(cases),
        "answered": sum(1 for t in traces if t["status"] == "answered"),
        "abstain": sum(1 for t in traces if t["status"] == "abstain"),
        "errors": sum(1 for t in traces if t["status"] == "error"),
        "total_seconds": total_seconds,
        "total_search_calls": sum(t.get("search_calls", 0) for t in traces),
        "total_llm_prompt_tokens": sum(t.get("llm_prompt_tokens", 0) for t in traces),
        "total_llm_completion_tokens": sum(t.get("llm_completion_tokens", 0) for t in traces),
        "total_rerank_tokens": sum(t.get("rerank_tokens", 0) for t in traces),
        "total_embed_tokens": sum(t.get("embed_tokens", 0) for t in traces),
        "total_estimated_cost_usd": round(sum(t.get("estimated_cost_usd", 0.0) for t in traces), 8),
    }

    target_dir = out_dir / f"{config_name}_{split}"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target_dir / "run_summary.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real Day 7 Agent evaluation")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--config",
        choices=("baseline_agent", "hybrid_expanded_agent", "hybrid_rerank_agent"),
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "basic" / "agent",
    )
    args = parser.parse_args()
    meta = run_agent_eval(args.split, args.config, args.output)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

