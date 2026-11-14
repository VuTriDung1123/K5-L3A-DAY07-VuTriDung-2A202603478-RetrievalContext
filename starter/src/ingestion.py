"""Read and normalize source PDF data with page provenance and diagnostics."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pypdf

from .models import DocumentPage

# OpenStax Principles of Data Science (PDF) core instructional page ranges (1-based physical pages)
DEFAULT_CORPUS_RANGES: list[tuple[int, int]] = [
    (19, 61),   # Chapter 1: Sections 1.1 - 1.5
    (67, 103),  # Chapter 2: Sections 2.1 - 2.5
    (387, 413), # Chapter 8: Sections 8.1 - 8.3
]

CHAPTER_TITLES: dict[int, str] = {
    1: "What Are Data and Data Science?",
    2: "Collecting and Preparing Data",
    8: "Ethics Throughout the Data Science Cycle",
}

# Section start physical PDF pages from the PDF outline / headings
SECTION_STARTS: list[tuple[int, str, str]] = [
    (19, "1.1", "What Is Data Science?"),
    (22, "1.2", "Data Science in Practice"),
    (26, "1.3", "Data and Datasets"),
    (38, "1.4", "Using Technology for Data Science"),
    (40, "1.5", "Data Science with Python"),
    (67, "2.0", "Introduction"),
    (68, "2.1", "Overview of Data Collection Methods"),
    (71, "2.2", "Survey Design and Implementation"),
    (76, "2.3", "Web Scraping and Social Media Data Collection"),
    (86, "2.4", "Data Cleaning and Preprocessing"),
    (98, "2.5", "Handling Large Datasets"),
    (387, "8.0", "Introduction"),
    (388, "8.1", "Ethics in Data Collection"),
    (398, "8.2", "Ethics in Data Analysis and Modeling"),
    (405, "8.3", "Ethics in Visualization and Reporting"),
]

_FOOTER_RE = re.compile(r"^\s*Access for free at openstax\.org\s*$", re.IGNORECASE)
_EVEN_HEADER_RE = re.compile(r"^\s*(\d+)\s+\d+\s+•\s+.+$")
_ODD_HEADER_RE = re.compile(r"^\s*\d+\.\d+\s+•\s+.+?\s+(\d+)\s*$")
_CH_OPENER_TITLE_RE = re.compile(
    r"^\s*(What Are Data and Data Science\?|Collecting and Preparing Data|Ethics Throughout the Data Science Cycle)\s*$"
)
_CH_OPENER_NUM_RE = re.compile(r"^\s*(1|2|8)\s*$")
_MINOR_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "to", "for",
    "with", "on", "by", "as", "at", "from", "vs.", "versus",
}


def _is_subheading_line(line: str) -> bool:
    """Detect standalone subheadings embedded inside a layout block."""
    if len(line) > 90:
        return False
    if line.endswith((".", "?", "!", ":", ";", ",")):
        return False
    if line.startswith(("•", "#", "(", "[", '"', "“", "Table ", "Figure ")):
        return False
    if re.match(r"^\d+\.\d+\s+[A-Z]", line):
        return True
    if line in (
        "Introduction",
        "Learning Outcomes",
        "Problem",
        "Solution",
        "PYTHON CODE",
        "EXPLORING FURTHER",
        "Chapter Outline",
    ):
        return True
    if re.match(r"^EXAMPLE\s+\d+\.\d+$", line):
        return True
    words = line.split()
    if 1 <= len(words) <= 10 and words[0][0].isupper():
        alpha_words = [w for w in words if w.isalpha()]
        if alpha_words and all(
            (w[0].isupper() or w.lower() in _MINOR_WORDS) for w in alpha_words
        ):
            return True
    return False


def _extract_book_page_and_footer_section(raw_text: str) -> tuple[int | None, str | None]:
    """Read printed book page number and running section label from OpenStax headers/footers."""
    book_page: int | None = None
    footer_sec: str | None = None
    for raw_line in raw_text.splitlines():
        line = unicodedata.normalize("NFKC", raw_line).strip()
        if not line:
            continue
        m_even = _EVEN_HEADER_RE.match(line)
        if m_even:
            book_page = int(m_even.group(1))
        m_odd = _ODD_HEADER_RE.match(line)
        if m_odd:
            book_page = int(m_odd.group(1))
            sec_m = re.match(r"^(\d+\.\d+)\s+•", line)
            if sec_m:
                footer_sec = sec_m.group(1)
    return book_page, footer_sec


def _resolve_page_section(pdf_page: int) -> tuple[int, str, str, str]:
    """Map physical PDF page to chapter number, chapter title, initial section, and section title."""
    if 19 <= pdf_page <= 66:
        chap_num = 1
    elif 67 <= pdf_page <= 112:
        chap_num = 2
    elif 387 <= pdf_page <= 420:
        chap_num = 8
    else:
        raise ValueError(f"Page {pdf_page} is outside the configured corpus chapters (1, 2, 8).")

    sec_code, sec_title = "0.0", "Unknown"
    for start_page, code, title in SECTION_STARTS:
        if pdf_page >= start_page and int(code.split(".")[0]) == chap_num:
            sec_code, sec_title = code, title
    return chap_num, CHAPTER_TITLES[chap_num], sec_code, sec_title


def normalize_text(text: str) -> str:
    """Normalize Unicode, strip OpenStax running headers/footers, and preserve paragraph breaks."""
    txt = unicodedata.normalize("NFKC", text or "")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = re.split(r"\n\s*\n+", txt)
    kept_blocks: list[str] = []

    for blk in raw_blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        filtered: list[str] = []
        for ln in lines:
            if (
                _FOOTER_RE.match(ln)
                or _EVEN_HEADER_RE.match(ln)
                or _ODD_HEADER_RE.match(ln)
                or _CH_OPENER_TITLE_RE.match(ln)
                or _CH_OPENER_NUM_RE.match(ln)
            ):
                continue
            filtered.append(re.sub(r"[ \t]+", " ", ln))

        if not filtered:
            continue

        # Split out inline subheadings embedded right after a completed paragraph
        sub_groups: list[list[str]] = []
        cur: list[str] = []
        for ln in filtered:
            if cur and _is_subheading_line(ln) and cur[-1].endswith((".", "?", "!", ")", '"', "”")):
                sub_groups.append(cur)
                sub_groups.append([ln])
                cur = []
            elif len(cur) == 1 and _is_subheading_line(cur[0]) and not _is_subheading_line(ln):
                sub_groups.append(cur)
                cur = [ln]
            else:
                cur.append(ln)
        if cur:
            sub_groups.append(cur)

        for grp in sub_groups:
            if any(ln.startswith("#") for ln in grp) or "PYTHON CODE" in grp:
                kept_blocks.append("\n".join(grp))
            elif any(ln.startswith("•") or re.match(r"^\d+\s*\.", ln) for ln in grp):
                list_lines: list[str] = []
                for ln in grp:
                    if ln.startswith("•") or re.match(r"^\d+\s*\.", ln) or not list_lines:
                        list_lines.append(ln)
                    else:
                        if list_lines[-1].endswith("/") and "http" in list_lines[-1].split()[-1]:
                            list_lines[-1] += ln
                        else:
                            list_lines[-1] += " " + ln
                kept_blocks.append("\n".join(list_lines))
            else:
                out = grp[0]
                for nxt in grp[1:]:
                    # Rejoin URLs wrapped right after '/'
                    if out.endswith("/") and "http" in out.split()[-1]:
                        out += nxt
                    else:
                        out += " " + nxt
                kept_blocks.append(out)

    return "\n\n".join(kept_blocks).strip()


def expand_page_ranges(ranges: list[tuple[int, int]] | None = None) -> list[int]:
    """Expand inclusive (start, end) 1-based physical page ranges into a sorted list."""
    selected = ranges or DEFAULT_CORPUS_RANGES
    pages: list[int] = []
    for start_p, end_p in selected:
        pages.extend(range(start_p, end_p + 1))
    return pages


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    page_numbers: list[int] | None = None,
    ingestion_log: list[dict[str, Any]] | None = None,
) -> list[DocumentPage]:
    """Extract and normalize selected physical PDF pages with provenance and error logs."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Corpus PDF not found: {pdf_path}")

    reader = pypdf.PdfReader(str(pdf_path))
    total_pdf_pages = len(reader.pages)
    target_pages = page_numbers if page_numbers is not None else expand_page_ranges()
    documents: list[DocumentPage] = []

    for pdf_page in target_pages:
        if pdf_page < 1 or pdf_page > total_pdf_pages:
            msg = f"Requested physical page {pdf_page} out of range (1..{total_pdf_pages})."
            if ingestion_log is not None:
                ingestion_log.append({"page": pdf_page, "status": "error", "reason": msg})
            raise ValueError(msg)

        try:
            raw_text = reader.pages[pdf_page - 1].extract_text(extraction_mode="layout") or ""
        except Exception as exc:
            if ingestion_log is not None:
                ingestion_log.append({"page": pdf_page, "status": "error", "reason": str(exc)})
            continue

        book_page, footer_sec = _extract_book_page_and_footer_section(raw_text)
        normalized = normalize_text(raw_text)
        chap_num, chap_title, sec_code, sec_title = _resolve_page_section(pdf_page)

        if not normalized:
            if ingestion_log is not None:
                ingestion_log.append(
                    {
                        "page": pdf_page,
                        "status": "empty_or_image_only",
                        "raw_chars": len(raw_text),
                        "normalized_chars": 0,
                    }
                )
            continue

        status = "ok"
        notes: list[str] = []
        if len(normalized) < 500:
            status = "low_text_warning"
            notes.append(
                "Short extracted text (<500 chars); may contain image-embedded screenshots, figures, or trailing lines."
            )
        if re.search(r"\b(is|are|be)\s+\.", normalized) or "least other records" in normalized:
            notes.append("Potential missing math-font symbol detected during pypdf extraction.")
            if status == "ok":
                status = "math_glyph_warning"

        if ingestion_log is not None:
            ingestion_log.append(
                {
                    "page": pdf_page,
                    "book_page": book_page,
                    "chapter": f"Chapter {chap_num}",
                    "section": sec_code,
                    "footer_section": footer_sec,
                    "status": status,
                    "raw_chars": len(raw_text),
                    "normalized_chars": len(normalized),
                    "notes": notes,
                }
            )

        documents.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=pdf_page,
                text=normalized,
                metadata={
                    "page_id": f"{document_id}:p{pdf_page:03d}",
                    "chapter": f"Chapter {chap_num}",
                    "chapter_number": chap_num,
                    "chapter_title": chap_title,
                    "section": sec_code,
                    "section_title": sec_title,
                    "book_page": book_page,
                    "raw_chars": len(raw_text),
                    "normalized_chars": len(normalized),
                },
            )
        )

    return documents
