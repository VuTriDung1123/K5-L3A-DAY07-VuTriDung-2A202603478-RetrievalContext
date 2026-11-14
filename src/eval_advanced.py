"""Evaluate Advanced Multimodal Agentic RAG on dev or held-out test split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .advanced_agent import AdvancedMultimodalAgent
from .advanced_retrieval import AdvancedBM25Index, AdvancedCorpusStore, AdvancedHybridRetriever
from .config import DATA_DIR, RESULTS_DIR, get_config
from .day6_adapter import QueryEmbedderClient


def run_advanced_eval(
    split: str,
    config_name: str,
    out_dir: Path,
    model_override: str | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    store = AdvancedCorpusStore(cfg=cfg)
    bm25 = AdvancedBM25Index(store.chunks)
    embedder = QueryEmbedderClient(cfg=cfg, use_cache=True)

    all_cases = json.loads((DATA_DIR / "advanced_benchmark.json").read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c["split"] == split]
    if not cases:
        raise ValueError(f"No advanced benchmark cases for split={split!r}")

    if config_name == "adv_baseline_dense":
        retriever = AdvancedHybridRetriever(store=store, embedder=embedder, bm25=bm25, mode="dense")
        runner = AdvancedMultimodalAgent(
            retriever=retriever,
            store=store,
            cfg=cfg,
            config_name=config_name,
            top_k=4,
            context_budget=2400,
            max_searches=3,
            allow_page_tool=False,
            allow_vision_tool=False,
            model=model_override,
            vision_model=model_override,
        )
    elif config_name == "adv_hybrid_multimodal":
        retriever = AdvancedHybridRetriever(store=store, embedder=embedder, bm25=bm25, mode="hybrid")
        runner = AdvancedMultimodalAgent(
            retriever=retriever,
            store=store,
            cfg=cfg,
            config_name=config_name,
            top_k=6,
            context_budget=9500,
            max_searches=5,
            allow_page_tool=True,
            allow_vision_tool=False,
            model=model_override,
            vision_model=model_override,
        )
    elif config_name == "adv_hybrid_vision_agent":
        retriever = AdvancedHybridRetriever(store=store, embedder=embedder, bm25=bm25, mode="hybrid")
        runner = AdvancedMultimodalAgent(
            retriever=retriever,
            store=store,
            cfg=cfg,
            config_name=config_name,
            top_k=6,
            context_budget=9500,
            max_searches=5,
            allow_page_tool=True,
            allow_vision_tool=True,
            model=model_override,
            vision_model=model_override,
        )
    else:
        raise ValueError(f"Unknown advanced config: {config_name!r}")

    traces: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for c in cases:
        tr = runner.run_query(c["id"], c["question"])
        tr["split"] = split
        tr["answerable"] = c["answerable"]
        tr["group"] = c["group"]
        tr["question"] = c["question"]
        traces.append(tr)
        print(
            f"[{split}][{config_name}] {c['id']} ({c['group']}, ans={c['answerable']}): "
            f"status={tr['status']}, searches={tr['search_calls']}, vis={tr['vision_calls']}, "
            f"cites={len(tr['citations'])}, {tr['elapsed_seconds']}s, ${tr.get('estimated_cost_usd', 0):.6f}",
            flush=True,
        )

    total_seconds = round(time.perf_counter() - t_all, 3)
    meta = {
        "config": config_name,
        "split": split,
        "model": runner.model,
        "cases": len(cases),
        "answered": sum(1 for t in traces if t["status"] == "answered"),
        "abstain": sum(1 for t in traces if t["status"] == "abstain"),
        "errors": sum(1 for t in traces if t["status"] == "error"),
        "total_seconds": total_seconds,
        "total_search_calls": sum(t.get("search_calls", 0) for t in traces),
        "total_vision_calls": sum(t.get("vision_calls", 0) for t in traces),
        "total_llm_prompt_tokens": sum(t.get("llm_prompt_tokens", 0) for t in traces),
        "total_llm_completion_tokens": sum(t.get("llm_completion_tokens", 0) for t in traces),
        "total_vision_prompt_tokens": sum(t.get("vision_prompt_tokens", 0) for t in traces),
        "total_vision_completion_tokens": sum(t.get("vision_completion_tokens", 0) for t in traces),
        "total_embed_tokens": sum(t.get("embed_tokens", 0) for t in traces),
        "total_estimated_cost_usd": round(sum(t.get("estimated_cost_usd", 0.0) for t in traces), 8),
    }
    target = out_dir / f"{config_name}_{split}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "run_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Advanced Multimodal Agentic RAG")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--config",
        choices=("adv_baseline_dense", "adv_hybrid_multimodal", "adv_hybrid_vision_agent"),
        required=True,
    )
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "advanced" / "agent")
    parser.add_argument("--model", type=str, default=None, help="Optional Gemini model override for agent + vision.")
    args = parser.parse_args()
    meta = run_advanced_eval(args.split, args.config, args.output, model_override=args.model)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

