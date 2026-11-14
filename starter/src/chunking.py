"""Fixed-size and paragraph/structure-aware chunking with provenance tracking."""
from __future__ import annotations

import re
from typing import Any

from .ingestion import SECTION_STARTS
from .models import Chunk, DocumentPage

_HEADING_CODES: dict[str, str] = {f"{code} {title}": code for _, code, title in SECTION_STARTS}


def _find_section_transitions(text: str, default_section: str) -> list[tuple[int, str]]:
    """Return sorted (char_offset, section_code) transitions detected on a page."""
    transitions: list[tuple[int, str]] = [(0, default_section)]
    for heading_text, code in _HEADING_CODES.items():
        # Skip Chapter Outline listings on chapter opener pages (where 1.1..1.5 appear together)
        if "Chapter Outline" in text[:600] and text.find(heading_text) < 600:
            # Find second occurrence if section starts on the same page (e.g., 1.1 on p.19)
            first_idx = text.find(heading_text)
            second_idx = text.find(heading_text, first_idx + len(heading_text))
            if second_idx != -1:
                transitions.append((second_idx, code))
            continue
        idx = text.find(heading_text)
        if idx != -1:
            transitions.append((idx, code))
    transitions.sort(key=lambda item: item[0])
    return transitions


def _section_at(transitions: list[tuple[int, str]], offset: int) -> str:
    """Resolve active section code at a given character offset."""
    active = transitions[0][1]
    for pos, code in transitions:
        if pos <= offset:
            active = code
        else:
            break
    return active


def _is_heading_block(block: str) -> bool:
    """Check whether a paragraph block is a heading or short label."""
    if "\n" in block or len(block) > 100:
        return False
    if block.endswith((".", "?", "!", ":", ";", ",")):
        return False
    if block.startswith(("•", "#", "Figure ", "Table ")):
        return False
    if re.match(r"^\d+\.\d+\s+[A-Z]", block):
        return True
    if block in ("Introduction", "Learning Outcomes", "Problem", "Solution", "PYTHON CODE", "EXPLORING FURTHER", "Chapter Outline"):
        return True
    if re.match(r"^EXAMPLE\s+\d+\.\d+$", block):
        return True
    words = block.split()
    return 1 <= len(words) <= 10 and words[0][0].isupper()


def _split_oversized_block(block: str, max_size: int) -> list[str]:
    """Split a single block exceeding max_size along paragraphs, lines, or sentences."""
    if len(block) <= max_size:
        return [block]

    # If heading + body were joined with '\n\n', keep heading attached to body's first part
    if "\n\n" in block:
        parts = [p.strip() for p in block.split("\n\n") if p.strip()]
        if len(parts) >= 2 and _is_heading_block(parts[0]):
            heading_prefix = "\n\n".join(parts[:-1])
            body = parts[-1]
            # Split body while reserving room for heading_prefix on the first piece
            body_units = _split_oversized_block(body, max_size)
            if len(heading_prefix) + 2 + len(body_units[0]) <= max_size:
                body_units[0] = f"{heading_prefix}\n\n{body_units[0]}"
                return body_units
            # If first unit was slightly too tight, re-split body with first slice budget reduced
            first_budget = max(max_size - len(heading_prefix) - 2, max_size // 2)
            first_pieces = _split_oversized_block(body, first_budget)
            first_chunk = f"{heading_prefix}\n\n{first_pieces[0]}"
            remainder_text = " ".join(first_pieces[1:]) if "\n" not in body else "\n".join(first_pieces[1:])
            if remainder_text.strip():
                return [first_chunk] + _split_oversized_block(remainder_text.strip(), max_size)
            return [first_chunk]

    # Prefer line splits for lists/code blocks, sentence splits for prose
    if "\n" in block:
        units = [u for u in block.split("\n") if u.strip()]
        sep = "\n"
    else:
        units = [u.strip() for u in re.split(r"(?<=[.!?])\s+", block) if u.strip()]
        sep = " "

    pieces: list[str] = []
    cur = ""
    for unit in units:
        if len(unit) > max_size:
            if cur:
                pieces.append(cur)
                cur = ""
            # Word-boundary fallback for exceptionally long single sentences
            words = unit.split(" ")
            buf = ""
            for w in words:
                cand = f"{buf} {w}".strip() if buf else w
                if len(cand) <= max_size:
                    buf = cand
                else:
                    if buf:
                        pieces.append(buf)
                    buf = w
            if buf:
                pieces.append(buf)
            continue

        cand = f"{cur}{sep}{unit}" if cur else unit
        if len(cand) <= max_size:
            cur = cand
        else:
            if cur:
                pieces.append(cur)
            cur = unit
    if cur:
        pieces.append(cur)
    return pieces


def _base_metadata(page: DocumentPage, section: str, chunking: str, chunk_index: int, char_start: int, char_end: int) -> dict[str, Any]:
    """Copy verifiable page metadata into chunk metadata."""
    meta = dict(page.metadata)
    meta["section"] = section
    meta["chunking"] = chunking
    meta["chunk_index"] = chunk_index
    meta["char_start"] = char_start
    meta["char_end"] = char_end
    return meta


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split page text into overlapping fixed-size character windows and retain metadata."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    text = page.text
    if not text.strip():
        return []

    default_sec = str(page.metadata.get("section", "0.0"))
    transitions = _find_section_transitions(text, default_sec)
    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + size, len(text))
        raw_slice = text[start:end]
        stripped = raw_slice.strip()
        if stripped:
            leading_trim = len(raw_slice) - len(raw_slice.lstrip())
            char_start = start + leading_trim
            char_end = char_start + len(stripped)
            sec = _section_at(transitions, char_start)
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}:p{page.page:03d}:fixed:c{idx:03d}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=stripped,
                    metadata=_base_metadata(page, sec, "fixed_size", idx, char_start, char_end),
                )
            )
            idx += 1
        if end >= len(text):
            break
        start += step

    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    """Group structural paragraphs up to max_size, splitting oversized blocks cleanly."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    text = page.text
    if not text.strip():
        return []

    default_sec = str(page.metadata.get("section", "0.0"))
    transitions = _find_section_transitions(text, default_sec)

    # Track blocks with exact offsets inside page.text
    raw_spans: list[tuple[int, int, str]] = []
    cursor = 0
    for blk in text.split("\n\n"):
        pos = text.find(blk, cursor)
        clean_blk = blk.strip()
        if clean_blk:
            raw_spans.append((pos, pos + len(blk), clean_blk))
        cursor = pos + len(blk) + 2

    # Merge short headings with the following content block before packing
    merged_units: list[tuple[int, int, str, bool]] = []
    i = 0
    while i < len(raw_spans):
        s_off, e_off, blk_txt = raw_spans[i]
        if _is_heading_block(blk_txt) and i + 1 < len(raw_spans):
            # Chain consecutive short headings (e.g., "EXAMPLE 8.1" + "Problem")
            combined_parts = [blk_txt]
            j = i + 1
            while j < len(raw_spans) - 1 and _is_heading_block(raw_spans[j][2]) and len("\n\n".join(combined_parts)) < max_size // 2:
                combined_parts.append(raw_spans[j][2])
                j += 1
            next_s, next_e, next_txt = raw_spans[j]
            combined_parts.append(next_txt)
            combined_txt = "\n\n".join(combined_parts)
            is_sec_start = bool(re.match(r"^\d+\.\d+\s+[A-Z]", blk_txt)) or _is_heading_block(blk_txt)
            for piece_idx, piece in enumerate(_split_oversized_block(combined_txt, max_size)):
                merged_units.append((s_off, next_e, piece, is_sec_start and piece_idx == 0))
            i = j + 1
        else:
            for piece_idx, piece in enumerate(_split_oversized_block(blk_txt, max_size)):
                merged_units.append((s_off, e_off, piece, False))
            i += 1

    # Pack adjacent units up to max_size, breaking cleanly at new heading/section starts
    chunks: list[Chunk] = []
    cur_text = ""
    cur_start = 0
    cur_end = 0
    idx = 0

    for u_start, u_end, u_text, starts_heading in merged_units:
        if not cur_text:
            cur_text = u_text
            cur_start = u_start
            cur_end = u_end
            continue

        # Start a new chunk when a heading boundary is reached and buffer has real context
        if starts_heading and len(cur_text) >= 250:
            sec = _section_at(transitions, cur_start)
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}:p{page.page:03d}:para:c{idx:03d}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=cur_text,
                    metadata=_base_metadata(page, sec, "by_paragraph", idx, cur_start, cur_end),
                )
            )
            idx += 1
            cur_text = u_text
            cur_start = u_start
            cur_end = u_end
            continue

        cand = f"{cur_text}\n\n{u_text}"
        if len(cand) <= max_size:
            cur_text = cand
            cur_end = u_end
        else:
            sec = _section_at(transitions, cur_start)
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}:p{page.page:03d}:para:c{idx:03d}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=cur_text,
                    metadata=_base_metadata(page, sec, "by_paragraph", idx, cur_start, cur_end),
                )
            )
            idx += 1
            cur_text = u_text
            cur_start = u_start
            cur_end = u_end

    if cur_text:
        sec = _section_at(transitions, cur_start)
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}:p{page.page:03d}:para:c{idx:03d}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=cur_text,
                metadata=_base_metadata(page, sec, "by_paragraph", idx, cur_start, cur_end),
            )
        )

    return chunks
