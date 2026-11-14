"""Offline unit tests for Day 7 Advanced Multimodal Ingestion, Retrieval, and Agent loops."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.advanced_agent import AdvancedMultimodalAgent
from src.advanced_ingest import ADVANCED_SOURCE_ID, classify_section, normalize_page_text, split_text_chunks
from src.advanced_retrieval import (
    AdvancedBM25Index,
    AdvancedCorpusStore,
    AdvancedHybridRetriever,
    pack_multimodal_context,
    tokenize_mm,
)
from src.config import DATA_DIR, AppConfig


class _FakeEmbedder:
    def embed_query(self, text: str):
        return [0.1] * 1536, {
            "cached": True,
            "tokens": 5,
            "cost_usd": 0.0,
            "reference_cost_usd": 1e-7,
            "seconds": 0.0,
        }


class _FakeStore:
    def __init__(self, chunks, assets):
        self.chunks = chunks
        self.chunks_by_id = {c["id"]: c for c in chunks}
        self.assets = assets
        self.assets_by_id = {a["asset_id"]: a for a in assets}
        self.chunks_by_page = {}
        self.chunks_by_asset = {}
        for c in chunks:
            self.chunks_by_page.setdefault(c["pdf_page"], []).append(c)
            if c.get("asset_id"):
                self.chunks_by_asset.setdefault(c["asset_id"], []).append(c)

    def fetch_dense_candidates(self, query_vec, top_k=20, modality=None):
        rows = [c for c in self.chunks if not modality or modality == "all" or c["modality"] == modality]
        return [{**c, "score": 0.9 - idx * 0.05} for idx, c in enumerate(rows[:top_k])]

    def read_page(self, pdf_page: int, include_neighbor_captions: bool = True):
        return {
            "pdf_page": pdf_page,
            "chunks": self.chunks_by_page.get(pdf_page, []),
            "linked_assets": [],
        }


class Day7AdvancedTests(unittest.TestCase):
    def test_section_classification_and_chunk_overlap(self) -> None:
        self.assertEqual(classify_section(1), "Front Matter & TOC")
        self.assertEqual(classify_section(37), "Summary for Policymakers")
        self.assertEqual(classify_section(100), "Section 3: Long-Term Climate and Development Futures")
        self.assertEqual(classify_section(143), "Annex I: Glossary")
        self.assertEqual(classify_section(185), "Index & Back Cover")

        para = normalize_page_text("   Sentence one.   " * 90)
        pieces = split_text_chunks(para, target_chars=300, max_chars=400, overlap_chars=60)
        self.assertGreater(len(pieces), 2)
        for p in pieces:
            self.assertLessEqual(len(p), 400)
        # Adjacent pieces share trailing overlap characters
        self.assertTrue(any(pieces[0][-25:] in pieces[1] or pieces[1][:25] in pieces[0] for _ in [0]))

    def test_tokenize_mm_and_hybrid_modality_filter(self) -> None:
        toks = tokenize_mm("Table SPM.1 C1-C8 CO2-eq")
        self.assertIn("spm.1", toks)
        self.assertIn("spm", toks)
        self.assertIn("c1-c8", toks)
        self.assertIn("c8", toks)

        chunks = [
            {
                "id": f"{ADVANCED_SOURCE_ID}:p037:table:table_spm1:c01",
                "source_id": ADVANCED_SOURCE_ID,
                "pdf_page": 37,
                "printed_page": "",
                "section": "Summary for Policymakers",
                "modality": "table",
                "asset_id": "table_spm1",
                "title": "Table SPM.1 GHG reductions",
                "content": "2030 GHG reduction 43 [34-60]% under C1 1.5C pathways.",
                "crop_path": "starter/data/advanced/crops/table_spm1_p037.png",
                "page_render_path": "starter/data/advanced/pages/page_037.png",
                "metadata": {},
            },
            {
                "id": f"{ADVANCED_SOURCE_ID}:p094:text:c001",
                "source_id": ADVANCED_SOURCE_ID,
                "pdf_page": 94,
                "printed_page": "78",
                "section": "Section 3",
                "modality": "text",
                "asset_id": None,
                "title": "Section 3 (PDF p.94)",
                "content": "Atlantic Meridional Overturning Circulation weakens over 21st century.",
                "crop_path": None,
                "page_render_path": None,
                "metadata": {},
            },
        ]
        store = _FakeStore(chunks, [])
        bm25 = AdvancedBM25Index(chunks)
        ret = AdvancedHybridRetriever(store=store, embedder=_FakeEmbedder(), bm25=bm25, mode="hybrid")

        table_hits, _ = ret.retrieve("Table SPM.1 43% reduction", top_k=5, modality="table")
        self.assertEqual(len(table_hits), 1)
        self.assertEqual(table_hits[0]["modality"], "table")

        packed, cids = pack_multimodal_context(table_hits, budget_chars=1000)
        self.assertEqual(cids, [chunks[0]["id"]])
        self.assertIn("asset=table_spm1", packed)
        self.assertIn("crop=starter/data/advanced/crops/table_spm1_p037.png", packed)

    def test_advanced_agent_vision_and_citation_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            crop_file = Path(tmpdir) / "dummy_crop.png"
            crop_file.write_bytes(b"\x89PNG\r\n\x1a\n")
            cid = f"{ADVANCED_SOURCE_ID}:p037:table:table_spm1:c01"
            chunks = [
                {
                    "id": cid,
                    "source_id": ADVANCED_SOURCE_ID,
                    "pdf_page": 37,
                    "printed_page": "",
                    "section": "Summary for Policymakers",
                    "modality": "table",
                    "asset_id": "table_spm1",
                    "title": "Table SPM.1",
                    "content": "Table SPM.1 emission reductions summary.",
                    "crop_path": str(crop_file),
                    "page_render_path": str(crop_file),
                    "metadata": {},
                }
            ]
            assets = [
                {
                    "asset_id": "table_spm1",
                    "title": "Table SPM.1",
                    "modality": "table",
                    "graphic_pages": [37],
                    "caption_pages": [37],
                    "crop_paths": ["dummy_crop.png"],
                    "page_render_paths": ["dummy_crop.png"],
                    "caption": "Table SPM.1 caption",
                }
            ]
            store = _FakeStore(chunks, assets)
            bm25 = AdvancedBM25Index(chunks)
            ret = AdvancedHybridRetriever(store=store, embedder=_FakeEmbedder(), bm25=bm25, mode="bm25")
            cfg = AppConfig(
                database_url="postgresql://unused",
                openrouter_api_key="unused",
                gemini_api_key="dummy-key",
                embedding_model="openai/text-embedding-3-small",
                embedding_dimensions=1536,
                agent_model="gemini-3.5-flash-lite",
                rerank_model="gemini-3.1-flash-lite-preview",
                vision_model="gemini-3.5-flash-lite",
            )
            agent = AdvancedMultimodalAgent(
                retriever=ret,
                store=store,
                cfg=cfg,
                allow_page_tool=True,
                allow_vision_tool=True,
            )

            turns = iter(
                [
                    {
                        "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 15},
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"functionCall": {"name": "search_corpus", "args": {"query": "Table SPM.1", "modality": "table"}}}],
                                }
                            }
                        ],
                    },
                    {
                        "usageMetadata": {"promptTokenCount": 70, "candidatesTokenCount": 18},
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"functionCall": {"name": "inspect_visual_asset", "args": {"asset_id": "table_spm1", "focus_question": "2030 GHG reduction for 1.5C"}}}],
                                }
                            }
                        ],
                    },
                    {
                        # First submit cites invalid chunk ID -> triggers citation_repair_requested
                        "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 20},
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"functionCall": {"name": "submit_answer", "args": {"decision": "answer", "answer": "43% [bad:id]", "citations": ["bad:id"]}}}],
                                }
                            }
                        ],
                    },
                    {
                        "usageMetadata": {"promptTokenCount": 110, "candidatesTokenCount": 22},
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"functionCall": {"name": "submit_answer", "args": {"decision": "answer", "answer": f"Median GHG reduction is 43% [34-60] [{cid}].", "citations": [cid]}}}],
                                }
                            }
                        ],
                    },
                ]
            )

            with (
                patch("src.advanced_agent.REPO_ROOT", Path(tmpdir)),
                patch.object(agent, "_call_gemini", side_effect=lambda *a, **kw: next(turns)),
                patch("src.advanced_agent._call_gemini_vision", return_value=("Row 1.5C GHG 2030 = 43 [34-60]%.", 30, 12)),
            ):
                tr = agent.run_query("t_vis", "What is 2030 GHG reduction in Table SPM.1?")

            self.assertEqual(tr["status"], "answered")
            self.assertEqual(tr["vision_calls"], 1)
            self.assertEqual(tr["citations"], [cid])
            self.assertTrue(any(e.get("event") == "citation_repair_requested" for e in tr["events"]))

    def test_advanced_benchmark_schema_counts(self) -> None:
        bm = json.loads((DATA_DIR / "advanced_benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual(len({c["id"] for c in bm}), len(bm))

        dev = [c for c in bm if c["split"] == "dev"]
        test = [c for c in bm if c["split"] == "test"]
        self.assertGreaterEqual(sum(1 for c in dev if c["answerable"]), 15)
        self.assertGreaterEqual(sum(1 for c in dev if not c["answerable"]), 3)
        self.assertGreaterEqual(len(test), 30)

        counts = {}
        for c in test:
            counts[c["group"]] = counts.get(c["group"], 0) + 1
        self.assertGreaterEqual(counts.get("table", 0), 6)
        self.assertGreaterEqual(counts.get("diagram_image", 0), 5)
        self.assertGreaterEqual(counts.get("chart", 0), 5)
        self.assertGreaterEqual(counts.get("multihop", 0), 4)
        self.assertGreaterEqual(counts.get("text", 0), 4)
        self.assertGreaterEqual(counts.get("unanswerable", 0), 6)

    def test_disabled_tool_and_text_fallback_repair(self) -> None:
        cid = f"{ADVANCED_SOURCE_ID}:p037:table:table_spm1:c01"
        chunks = [
            {
                "id": cid,
                "source_id": ADVANCED_SOURCE_ID,
                "pdf_page": 37,
                "printed_page": "",
                "section": "Summary for Policymakers",
                "modality": "table",
                "asset_id": "table_spm1",
                "title": "Table SPM.1",
                "content": "Table SPM.1 reports 43% reduction by 2030.",
                "crop_path": None,
                "page_render_path": None,
                "metadata": {},
            }
        ]
        store = _FakeStore(chunks, [])
        bm25 = AdvancedBM25Index(chunks)
        ret = AdvancedHybridRetriever(store=store, embedder=_FakeEmbedder(), bm25=bm25, mode="bm25")
        cfg = AppConfig(
            database_url="postgresql://unused",
            openrouter_api_key="unused",
            gemini_api_key="dummy-key",
            embedding_model="openai/text-embedding-3-small",
            embedding_dimensions=1536,
            agent_model="gemini-3.1-flash-lite",
            rerank_model="gemini-3.1-flash-lite-preview",
            vision_model="gemini-3.1-flash-lite",
        )
        agent = AdvancedMultimodalAgent(
            retriever=ret,
            store=store,
            cfg=cfg,
            allow_page_tool=False,
            allow_vision_tool=False,
        )

        captured_contents = []
        turns = iter(
            [
                {
                    "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 10},
                    "candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "search_corpus", "args": {"query": "Table SPM.1"}}}]}}],
                },
                {
                    # Disabled tool call must produce a user functionResponse turn rather than leaving contents ending on model
                    "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 10},
                    "candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "read_pdf_page", "args": {"pdf_page": 37}}}]}}],
                },
                {
                    # Plain-text response without submit_answer triggers submit_repair_requested
                    "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10},
                    "candidates": [{"content": {"role": "model", "parts": [{"text": "Based on Table SPM.1, GHG drops 43%."}]}}],
                },
                {
                    "usageMetadata": {"promptTokenCount": 60, "candidatesTokenCount": 12},
                    "candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "submit_answer", "args": {"decision": "answer", "answer": f"43% [{cid}].", "citations": [cid]}}}]}}],
                },
            ]
        )

        def _fake_call(contents, system_instruction, tools):
            self.assertEqual(contents[-1]["role"], "user")
            captured_contents.append(len(contents))
            return next(turns)

        with patch.object(agent, "_call_gemini", side_effect=_fake_call):
            tr = agent.run_query("t_repair", "2030 reduction?")

        self.assertEqual(tr["status"], "answered")
        self.assertEqual(tr["citations"], [cid])
        self.assertTrue(any(e.get("event") == "submit_repair_requested" for e in tr["events"]))
        self.assertEqual(captured_contents, [1, 3, 5, 7])


if __name__ == "__main__":
    unittest.main()
