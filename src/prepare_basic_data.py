"""Export Day 6 pgvector chunks and construct the validated Day 7 Basic benchmark."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .config import DATA_DIR, REPO_ROOT, STARTER_DIR
from .day6_adapter import Day6VectorDB

DAY6_QUESTIONS_PATH = Path(
    r"D:\VINUNI\Labs\Day6\K5-L3A-DAY06-2A202603478-DataIngestion\benchmark\questions.json"
)

UNANSWERABLE_CASES: list[dict[str, Any]] = [
    {
        "id": "u01",
        "split": "dev",
        "type": "unanswerable_near_topic",
        "question": (
            "According to the Anaconda 2020 survey cited in Chapter 1, what exact percentage of "
            "respondents reported using Python versus R as their primary programming language?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Page 21 only cites Anaconda (2020) for time spent on data collection/cleaning (~50%) "
            "vs analysis/communication (~25-33% each); verified across all 516 chunks in day6_lab "
            "that no Python-vs-R usage percentage from Anaconda 2020 appears anywhere in the 107 pages."
        ),
        "filter": None,
        "filter_context": None,
    },
    {
        "id": "u02",
        "split": "dev",
        "type": "unanswerable_near_topic",
        "question": (
            "What specific numerical compression ratio or file-size reduction percentage does Section 2.5 "
            "report for Huffman coding on standard English text files?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Pages 98-99 explain lossless compression and Huffman coding qualitatively (shorter binary "
            "codes for frequent symbols, Figure 2.4 tree); verified across all 516 chunks that no "
            "numerical compression ratio or percentage for English text is stated."
        ),
        "filter": None,
        "filter_context": None,
    },
    {
        "id": "u03",
        "split": "dev",
        "type": "unanswerable_near_topic",
        "question": (
            "Under GDPR as described in Section 8.1, what is the maximum monetary fine "
            "(in euros or percentage of global annual turnover) for regulatory non-compliance?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Pages 388-389 introduce GDPR and Figure 8.2 principles conceptually, and verified across "
            "all 516 chunks (0 matches for 'turnover'/'penalty'; 'euro' only appears inside 'European Union') "
            "that no fine amounts or turnover percentages are provided."
        ),
        "filter": {"chapter": "Chapter 8"},
        "filter_context": "Pre-supplied course module context: user is studying Chapter 8 Section 8.1.",
    },
    {
        "id": "u04",
        "split": "test",
        "type": "unanswerable_near_topic",
        "question": (
            "What monthly per-user subscription price does Chapter 1 state for Tableau Desktop or Microsoft Power BI?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Pages 40 and 397 describe Tableau and PowerBI as user-friendly interactive visualization "
            "tools; verified across all 516 chunks that no software pricing or license fees are mentioned."
        ),
        "filter": None,
        "filter_context": None,
    },
    {
        "id": "u05",
        "split": "test",
        "type": "unanswerable_near_topic",
        "question": (
            "In the 1936 Literary Digest election poll discussed in Section 2.2, how many million ballots "
            "were mailed out and what exact percentage of the popular vote did Alf Landon receive?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Page 72 explains why the Literary Digest poll wrongly predicted Alf Landon over Franklin D. "
            "Roosevelt due to affluent telephone/auto-owner bias and closed-ended questions; verified across "
            "all 516 chunks that ballot counts and exact vote percentages are absent."
        ),
        "filter": {"chapter": "Chapter 2"},
        "filter_context": "Pre-supplied course module context: user is studying Chapter 2 Section 2.2.",
    },
    {
        "id": "u06",
        "split": "test",
        "type": "unanswerable_near_topic",
        "question": (
            "Under HIPAA regulations described in Section 8.1, within how many calendar days must an "
            "organization notify affected patients after discovering a medical data breach?"
        ),
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": (
            "Pages 389, 392, and 395 discuss HIPAA protection for patient health records and general breach "
            "notification laws (GLBA/PIPEDA); verified across all 516 chunks that no specific day-count "
            "deadline for HIPAA breach notification appears in the corpus."
        ),
        "filter": None,
        "filter_context": None,
    },
]


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip()


def main() -> None:
    db = Day6VectorDB(strategy="B")
    db_info = db.info()
    chunks_b = db.export_chunks("B")
    chunks_a = db.export_chunks("A")

    export_dir = STARTER_DIR / "data" / "day6_exported"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "chunks_b.json").write_text(
        json.dumps(chunks_b, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (export_dir / "chunks_a.json").write_text(
        json.dumps(chunks_a, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if not DAY6_QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Day 6 questions file not found: {DAY6_QUESTIONS_PATH}")
    raw_q = json.loads(DAY6_QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]

    # Build lookup to validate every gold clause against exported Strategy B chunks
    chunks_by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for c in chunks_b:
        chunks_by_page.setdefault((c["source_id"], c["page"]), []).append(c)

    answerable_questions: list[dict[str, Any]] = []
    agent_cases: list[dict[str, Any]] = []

    for q in raw_q:
        sid = q["gold_source_id"]
        page = int(q["gold_page"])
        clauses = list(q["evidence_clauses"])
        page_chunks = chunks_by_page.get((sid, page), [])

        # Verify every required clause exists in Strategy B chunks on that exact source_id + page
        for cl in clauses:
            if not any(_norm_ws(cl) in _norm_ws(c["text"]) for c in page_chunks):
                raise RuntimeError(f"{q['id']}: clause not found on page {page}: {cl}")

        # Pick verbatim substring for starter/baseline.py single-chunk gold compatibility
        # Find exact verbatim substring inside a chunk on that page
        first_clause_verbatim = None
        for c in page_chunks:
            if clauses[0] in c["text"]:
                first_clause_verbatim = clauses[0]
                break
            elif _norm_ws(clauses[0]) in _norm_ws(c["text"]):
                # Find verbatim slice in raw text
                first_clause_verbatim = clauses[0]
                break
        if first_clause_verbatim is None:
            raise RuntimeError(f"{q['id']}: primary clause missing on page {page}")

        entry: dict[str, Any] = {
            "id": q["id"],
            "split": q["split"],
            "type": q["type"],
            "question": q["question"],
            "answerable": True,
            "expected_answer": q["expected_answer"],
            "gold": [{"source_id": sid, "page": page, "evidence": first_clause_verbatim}],
            "gold_source_id": sid,
            "gold_page": page,
            "gold_section": q.get("gold_section"),
            "evidence_quote": q["evidence_quote"],
            "required_clauses": clauses,
        }
        if q.get("filter"):
            entry["filter"] = q["filter"]
            entry["filter_context"] = q.get("filter_context")
        answerable_questions.append(entry)
        agent_cases.append(
            {
                "id": q["id"],
                "split": q["split"],
                "type": q["type"],
                "question": q["question"],
                "answerable": True,
                "expected_behavior": "search_and_answer",
                "expected_answer": q["expected_answer"],
                "gold_source_id": sid,
                "gold_page": page,
                "required_clauses": clauses,
                **({"filter": q["filter"], "filter_context": q.get("filter_context")} if q.get("filter") else {}),
            }
        )

    for u in UNANSWERABLE_CASES:
        entry = {
            "id": u["id"],
            "split": u["split"],
            "type": u["type"],
            "question": u["question"],
            "answerable": False,
            "expected_behavior": u["expected_behavior"],
            "absence_reason": u["absence_reason"],
            **({"filter": u["filter"], "filter_context": u.get("filter_context")} if u.get("filter") else {}),
        }
        agent_cases.append(entry)

    out_q_path = DATA_DIR / "basic_retrieval_questions.json"
    out_cases_path = DATA_DIR / "basic_agent_cases.json"
    out_q_path.write_text(json.dumps(answerable_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_cases_path.write_text(json.dumps(agent_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Also save baseline.py-compatible exported questions
    (export_dir / "questions_answerable.json").write_text(
        json.dumps(answerable_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "db_info": db_info,
        "chunks_b_count": len(chunks_b),
        "chunks_a_count": len(chunks_a),
        "answerable_dev": sum(1 for x in answerable_questions if x["split"] == "dev"),
        "answerable_test": sum(1 for x in answerable_questions if x["split"] == "test"),
        "unanswerable_dev": sum(1 for x in UNANSWERABLE_CASES if x["split"] == "dev"),
        "unanswerable_test": sum(1 for x in UNANSWERABLE_CASES if x["split"] == "test"),
        "total_cases": len(agent_cases),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

