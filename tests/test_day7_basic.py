"""Unit tests for Day 7 retrievers, filters, context packing, and evaluation metrics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import _pack_with_recency_priority
from src.eval_retrieval import clauses_covered, strict_chunk_hit
from src.retrievers import (
    BM25Retriever,
    HybridRRFRetriever,
    pack_context_expanded,
    pack_context_greedy,
)


def _chunk(cid: str, page: int, text: str, chapter: str = "Chapter 1", section: str = "1.1") -> dict:
    return {
        "id": cid,
        "source_id": "doc",
        "source": "doc.pdf",
        "page": page,
        "text": text,
        "metadata": {"chapter": chapter, "section": section, "section_title": "Intro"},
    }


class _FakeDense:
    name = "dense"

    def __init__(self, rows):
        self.rows = rows

    def search(self, query, *, filters=None, top_k=3):
        out = [dict(r, rank=i + 1, score=1.0 - 0.1 * i) for i, r in enumerate(self.rows)]
        return out[:top_k], {"embed_tokens": 0, "embed_cost_usd": 0.0, "embed_reference_cost_usd": 0.0, "embed_cached": True}


class Day7BasicTests(unittest.TestCase):
    def setUp(self):
        self.c1 = _chunk("doc:p010:para:c000", 10, "First part: lossy compression removes redundant data.", "Chapter 2", "2.5")
        self.c2 = _chunk("doc:p010:para:c001", 10, "Second part: lossless compression includes ZIP and PNG.", "Chapter 2", "2.5")
        self.c3 = _chunk("doc:p050:para:c000", 50, "Outliers can bias algorithmic decisions in Chapter 8.", "Chapter 8", "8.2")
        self.chunks = [self.c1, self.c2, self.c3]
        self.by_page = {
            ("doc", 10): [self.c1, self.c2],
            ("doc", 50): [self.c3],
        }

    def test_bm25_and_metadata_filter(self):
        bm25 = BM25Retriever(self.chunks)
        hits, _ = bm25.search("outliers bias", filters={"chapter": "Chapter 8"}, top_k=3)
        self.assertEqual([h["id"] for h in hits], ["doc:p050:para:c000"])
        empty_hits, _ = bm25.search("outliers bias", filters={"chapter": "Chapter 1"}, top_k=3)
        self.assertEqual(empty_hits, [])

    def test_hybrid_rrf_combines_ranks(self):
        bm25 = BM25Retriever(self.chunks)
        dense = _FakeDense([self.c2, self.c1])
        hybrid = HybridRRFRetriever(dense=dense, bm25=bm25, candidate_k=5)
        hits, _ = hybrid.search("lossless compression ZIP", top_k=2)
        self.assertEqual(hits[0]["id"], "doc:p010:para:c001")

    def test_context_expansion_packs_adjacent_same_page_chunk(self):
        q = {
            "gold_source_id": "doc",
            "gold_page": 10,
            "required_clauses": ["lossy compression removes", "lossless compression includes ZIP"],
        }
        # Top-1 alone is not strict single-chunk hit and greedy budget=150 excludes c2
        self.assertFalse(strict_chunk_hit(self.c1, q))
        ctx_greedy, ids_greedy = pack_context_greedy([self.c1], budget=150)
        self.assertEqual(ids_greedy, ["doc:p010:para:c000"])
        self.assertFalse(clauses_covered([self.c1], q))

        # Expanded context with larger budget includes adjacent c2 on page 10
        ctx_exp, ids_exp = pack_context_expanded([self.c1], self.by_page, budget=500, expand_neighbors=True)
        self.assertEqual(ids_exp, ["doc:p010:para:c000", "doc:p010:para:c001"])
        self.assertTrue(clauses_covered([self.c1, self.c2], q))
        self.assertIn("First part", ctx_exp)
        self.assertIn("Second part", ctx_exp)

    def test_recency_priority_prevents_search_starvation(self):
        cands = {self.c1["id"]: {**self.c1, "score": 0.9}, self.c3["id"]: {**self.c3, "score": 0.4}}
        # Budget fits only 1 chunk: prioritizing latest search result c3 keeps c3 instead of starving on c1
        _, ids = _pack_with_recency_priority(
            cands,
            latest_ids=[self.c3["id"]],
            all_chunks_by_page=self.by_page,
            budget=160,
            expand_neighbors=False,
            greedy_only=False,
        )
        self.assertEqual(ids, [self.c3["id"]])


if __name__ == "__main__":
    unittest.main()

