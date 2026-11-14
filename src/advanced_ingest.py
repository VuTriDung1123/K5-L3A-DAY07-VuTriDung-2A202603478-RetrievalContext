"""Whole-document multimodal ingestion for the 186-page IPCC AR6 SYR Full Volume PDF."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
import psycopg
import pymupdf
import requests

from .config import STARTER_DIR, AppConfig, estimate_llm_cost_usd, get_config

ADVANCED_SOURCE_ID = "ipcc-ar6-syr-2023"
ADVANCED_DIR = STARTER_DIR / "data" / "advanced"
ADVANCED_PDF = ADVANCED_DIR / "IPCC_AR6_SYR_FullVolume.pdf"
CROPS_DIR = ADVANCED_DIR / "crops"
PAGES_DIR = ADVANCED_DIR / "pages"
VISION_CACHE_PATH = ADVANCED_DIR / "vision_cache.json"
EMBED_CACHE_PATH = ADVANCED_DIR / "chunk_embeddings_cache.json"
ASSETS_JSON_PATH = ADVANCED_DIR / "visual_assets.json"
CHUNKS_JSON_PATH = ADVANCED_DIR / "advanced_chunks.json"
MANIFEST_PATH = ADVANCED_DIR / "ingestion_manifest.json"


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    title: str
    modality: str  # 'table' | 'chart' | 'diagram_map'
    graphic_pages: tuple[int, ...]
    caption_pages: tuple[int, ...]
    crop_hint: tuple[float, float, float, float] | None = None  # optional override rect


ASSET_CATALOG: tuple[AssetSpec, ...] = (
    AssetSpec("box_spm1_t1", "Box SPM.1, Table 1: Description and relationship of scenarios and modelled pathways (C1-C8, SSPs)", "table", (26,), (26,), crop_hint=(55, 45, 555, 345)),
    AssetSpec("table_spm1", "Table SPM.1: Greenhouse gas and CO2 emission reductions from 2019 (median and 5-95 percentiles)", "table", (37,), (37,), crop_hint=(55, 45, 555, 175)),
    AssetSpec("table_2_1", "Table 2.1: Assessment of observed changes in large-scale climate indicators and human influence", "table", (63,), (63,)),
    AssetSpec("table_2_2", "Table 2.2: Projected global emissions in 2030 under implemented policies, NDCs, and 2030 emissions gaps", "table", (74,), (74,)),
    AssetSpec("table_3_1", "Table 3.1: Key characteristics of modelled global emissions pathways (C1-C6)", "table", (100,), (100,), crop_hint=(28, 45, 575, 745)),
    AssetSpec("fig_spm1", "Figure SPM.1: Observed impacts, attribution of warming, and regional changes", "chart", (23,), (23, 24)),
    AssetSpec("fig_spm2", "Figure SPM.2: Projected regional temperature, soil moisture, and precipitation maps at 1.5C-4C", "diagram_map", (30,), (30,)),
    AssetSpec("fig_spm3", "Figure SPM.3: Projected risks and impacts on ecosystems, human health, maize and fisheries yields", "diagram_map", (32,), (32, 33)),
    AssetSpec("fig_spm4", "Figure SPM.4: Assessed temperature outcomes (SSP1-2.6/SSP3-7.0) and AR5 vs AR6 Reasons for Concern burning embers", "chart", (33,), (33, 34)),
    AssetSpec("fig_spm5", "Figure SPM.5: Global emissions pathways consistent with policies, NDCs, and C1-C3 pathways", "chart", (38,), (39,)),
    AssetSpec("fig_spm6", "Figure SPM.6: Illustrative Climate Resilient Development pathways diagram", "diagram_map", (41,), (41,)),
    AssetSpec("fig_spm7", "Figure SPM.7: Potential GHG mitigation options, net emission reductions, and cost tiers by 2030", "chart", (43,), (43,)),
    AssetSpec("fig_1_1", "Figure 1.1: The Synthesis Report figures key and icons legend", "diagram_map", (55,), (55,)),
    AssetSpec("fig_2_1", "Figure 2.1: Causal chain from emissions to atmospheric concentrations and global temperature change", "diagram_map", (59,), (59,)),
    AssetSpec("fig_2_2", "Figure 2.2: Regional GHG emissions and historical cumulative CO2 proportions (1850-2019)", "chart", (61,), (62,)),
    AssetSpec("fig_2_3", "Figure 2.3: Vulnerability to climate extremes and historical regional contribution disparities", "diagram_map", (64, 65), (65, 66)),
    AssetSpec("fig_2_4", "Figure 2.4: Unit cost reductions and global adoption of PV, wind, CSP, LEDs, EVs and batteries", "chart", (70,), (70,), crop_hint=(20, 30, 585, 535)),
    AssetSpec("fig_2_5", "Figure 2.5: Global GHG emissions funnels and 2030 policy/NDC implementation gap", "chart", (75,), (75,)),
    AssetSpec("fig_csb2_1", "Cross-Section Box.2 Figure 1: Schematic of AR6 framework assessing emissions, GWLs, and risks", "diagram_map", (81,), (82,)),
    AssetSpec("fig_3_1", "Figure 3.1: Section 3 spatial maps of projected temperature, soil moisture, and precipitation changes", "diagram_map", (86,), (86,)),
    AssetSpec("fig_3_2", "Figure 3.2: Section 3 spatial risk maps (species loss, heat-humidity, maize & fisheries yields)", "diagram_map", (89,), (90,)),
    AssetSpec("fig_3_3", "Figure 3.3: Synthetic risk diagrams (RFC embers, ecosystem risks, regional hexagonal key risk bars)", "chart", (91, 92), (93,)),
    AssetSpec("fig_3_4", "Figure 3.4: Observed and projected global mean sea level rise (2100/2300) and extreme sea level frequency", "chart", (96,), (97,)),
    AssetSpec("fig_3_5", "Figure 3.5: Cumulative past, projected, and committed CO2 emissions vs remaining carbon budgets", "chart", (99,), (99,)),
    AssetSpec("fig_3_6", "Figure 3.6: Total GHG, CO2, and CH4 emissions trajectories and timing of reaching net zero", "chart", (102,), (102,)),
    AssetSpec("fig_4_1", "Figure 4.1: Sectoral CO2 and non-CO2 emissions in pathways limiting warming to 1.5C", "chart", (110,), (111,)),
    AssetSpec("fig_4_2", "Figure 4.2: Section 4 illustrative Climate Resilient Development pathways and near-term choices", "diagram_map", (113,), (113,)),
    AssetSpec("fig_4_3", "Figure 4.3: Compound and cascading climate risks across sectors and regions diagram", "diagram_map", (116,), (117,)),
    AssetSpec("fig_4_4", "Figure 4.4: Near-term mitigation potential and cost breakdown across sectors (Energy, AFOLU, Transport, Buildings, Industry)", "chart", (119,), (120,)),
    AssetSpec("fig_4_5", "Figure 4.5: Potential synergies and trade-offs matrix between mitigation/adaptation options and SDGs", "diagram_map", (125,), (125,)),
    AssetSpec("fig_4_6", "Figure 4.6: Breakdown of average annual mitigation investment flows vs investment needs until 2030 by sector and region", "chart", (129,), (129,)),
)


def classify_section(pdf_page: int) -> str:
    if pdf_page <= 18:
        return "Front Matter & TOC"
    if pdf_page <= 52:
        return "Summary for Policymakers"
    if pdf_page <= 56:
        return "Section 1: Introduction"
    if pdf_page <= 82:
        return "Section 2: Current Status and Trends"
    if pdf_page <= 106:
        return "Section 3: Long-Term Climate and Development Futures"
    if pdf_page <= 132:
        return "Section 4: Near-Term Responses in a Changing Climate"
    if pdf_page <= 146:
        return "Annex I: Glossary"
    if pdf_page <= 150:
        return "Annex II: Acronyms and Units"
    if pdf_page <= 178:
        return "Annex III-V: Contributors, Reviewers, Publications"
    return "Index & Back Cover"


def normalize_page_text(raw: str) -> str:
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def split_text_chunks(
    text: str,
    target_chars: int = 900,
    max_chars: int = 1150,
    overlap_chars: int = 150,
) -> list[str]:
    """Chunk page text on line/sentence boundaries with controlled overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            cut = text.rfind("\n", start + int(target_chars * 0.55), end)
            if cut == -1:
                cut = text.rfind(". ", start + int(target_chars * 0.55), end)
                if cut != -1:
                    cut += 1
            if cut == -1:
                cut = text.rfind(" ", start + int(target_chars * 0.55), end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def _extract_page_text(page: pymupdf.Page) -> str:
    raw = page.get_text("text")
    return raw if isinstance(raw, str) else str(raw)


def _extract_caption_text(doc: pymupdf.Document, caption_pages: tuple[int, ...], title: str) -> str:
    prefix = title.split(":", 1)[0].strip()
    pieces: list[str] = []
    for pno in caption_pages:
        raw = _extract_page_text(doc[pno - 1])
        idx = raw.find(prefix)
        if idx != -1:
            pieces.append(re.sub(r"\s+", " ", raw[idx : idx + 2200]).strip())
        else:
            pieces.append(re.sub(r"\s+", " ", raw[:1400]).strip())
    return "\n".join(pieces).strip()


def _compute_graphic_rect(page: pymupdf.Page, fallback_full: bool = False) -> pymupdf.Rect:
    pr = page.rect
    if fallback_full:
        return pymupdf.Rect(15, 25, pr.width - 15, pr.height - 20)
    rects: list[pymupdf.Rect] = []
    for d in page.get_drawings():
        r = d["rect"]
        # Exclude vertical side-margin section tabs (x0 < 28 or x1 > 584) and page headers/footers
        if r.x1 > 32 and r.x0 < pr.width - 32 and r.y1 > 48 and r.y0 < pr.height - 42:
            if r.width > 4 or r.height > 4:
                rects.append(r)
    for img in page.get_image_info(xrefs=True):
        r = pymupdf.Rect(img["bbox"])
        if r.x1 > 32 and r.x0 < pr.width - 32 and r.y1 > 48 and r.y0 < pr.height - 42:
            rects.append(r)
    if not rects:
        return pymupdf.Rect(20, 35, pr.width - 20, pr.height - 25)
    return pymupdf.Rect(
        max(15.0, min(r.x0 for r in rects) - 20.0),
        max(28.0, min(r.y0 for r in rects) - 24.0),
        min(pr.width - 15.0, max(r.x1 for r in rects) + 20.0),
        min(pr.height - 18.0, max(r.y1 for r in rects) + 28.0),
    )


def _extract_pdfplumber_tables(pdf_path: Path, page_numbers: set[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    with pdfplumber.open(pdf_path) as pdoc:
        for pno in sorted(page_numbers):
            page = pdoc.pages[pno - 1]
            pw = float(page.width)
            tables = page.find_tables() or []
            md_parts: list[str] = []
            for t in tables:
                x0, y0, x1, y1 = t.bbox
                # Skip side-margin section strip
                if x1 < 40 or x0 > pw - 40 or (x1 - x0) < 60:
                    continue
                rows = t.extract() or []
                cleaned = [
                    [re.sub(r"\s+", " ", (cell or "")).strip() for cell in row]
                    for row in rows
                    if any((cell or "").strip() for cell in row)
                ]
                if not cleaned:
                    continue
                for r in cleaned:
                    md_parts.append("| " + " | ".join(r) + " |")
                md_parts.append("")
            out[pno] = "\n".join(md_parts).strip()
    return out


def _call_gemini_vision(
    cfg: AppConfig,
    image_paths: list[Path],
    prompt_text: str,
    model: str | None = None,
) -> tuple[str, int, int]:
    """Call Gemini Vision API with rate-limit retry logic."""
    used_model = model or cfg.vision_model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{used_model}:generateContent?key={cfg.gemini_api_key}"
    parts: list[dict[str, Any]] = []
    for p in image_paths:
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        parts.append({"inlineData": {"mimeType": "image/png", "data": b64}})
    parts.append({"text": prompt_text})
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
    }
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            resp = requests.post(url, json=payload, timeout=75)
            if resp.status_code == 429:
                body = resp.text[:600]
                if "per day" in body.lower() or "perday" in body.lower():
                    raise RuntimeError(f"Gemini Vision daily quota exceeded: {body}")
                m = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", body, re.I)
                wait_s = min(float(m.group(1)) + 1.5, 65.0) if m else min(4.0 * (2**attempt), 45.0)
                last_err = RuntimeError(f"Gemini HTTP 429: {body}")
                if attempt < 4:
                    time.sleep(wait_s)
                    continue
                raise last_err
            if resp.status_code >= 500:
                raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
            data = resp.json()
            cands = data.get("candidates") or [{}]
            out_parts = (cands[0].get("content") or {}).get("parts") or []
            text_out = "\n".join(p.get("text", "") for p in out_parts if "text" in p).strip()
            usage = data.get("usageMetadata") or {}
            in_tok = int(usage.get("promptTokenCount") or 0)
            out_tok = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
            return text_out, in_tok, out_tok
        except Exception as exc:
            last_err = exc
            if "daily quota" in str(exc):
                break
            if attempt < 4:
                time.sleep(min(2.0 ** attempt, 12.0))
    raise RuntimeError(f"Gemini Vision failed: {last_err}") from last_err


def _embed_texts_cached(cfg: AppConfig, texts: list[str], batch_size: int = 40) -> tuple[list[list[float]], int, float]:
    """Embed chunks with SHA-256 disk cache via OpenRouter embeddings endpoint."""
    cache: dict[str, list[float]] = {}
    if EMBED_CACHE_PATH.exists():
        cache = json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))

    hashes = [hashlib.sha256(f"{cfg.embedding_model}::{t}".encode("utf-8")).hexdigest() for t in texts]
    missing_indices = [i for i, h in enumerate(hashes) if h not in cache]
    total_tokens = 0

    for start in range(0, len(missing_indices), batch_size):
        batch_idx = missing_indices[start : start + batch_size]
        batch_inputs = [texts[i] for i in batch_idx]
        for attempt in range(4):
            resp = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {cfg.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": cfg.embedding_model, "input": batch_inputs},
                timeout=60,
            )
            if resp.status_code in (429, 502, 503, 504) and attempt < 3:
                time.sleep(2.0 ** attempt)
                continue
            resp.raise_for_status()
            payload = resp.json()
            data_sorted = sorted(payload["data"], key=lambda x: x["index"])
            for idx_in_batch, row in enumerate(data_sorted):
                h = hashes[batch_idx[idx_in_batch]]
                cache[h] = row["embedding"]
            total_tokens += int((payload.get("usage") or {}).get("total_tokens") or 0)
            break

    EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBED_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    cost_usd = estimate_llm_cost_usd(cfg.embedding_model, total_tokens, 0)
    return [cache[h] for h in hashes], total_tokens, cost_usd


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"


def ensure_advanced_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS advanced_chunks (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                pdf_page INT NOT NULL,
                printed_page TEXT,
                section TEXT NOT NULL,
                modality TEXT NOT NULL,
                asset_id TEXT,
                title TEXT,
                content TEXT NOT NULL,
                bbox_json JSONB,
                crop_path TEXT,
                page_render_path TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                embedding vector(1536)
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_advanced_chunks_embedding_hnsw
            ON advanced_chunks USING hnsw (embedding vector_cosine_ops);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_advanced_chunks_page_modality
            ON advanced_chunks (pdf_page, modality);
            """
        )
    conn.commit()


def run_ingest(skip_vision: bool = False) -> dict[str, Any]:
    cfg = get_config()
    if not ADVANCED_PDF.exists():
        raise FileNotFoundError(f"Missing Advanced PDF: {ADVANCED_PDF}")

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(ADVANCED_PDF)
    n_pages = len(doc)

    # 1. Extract all non-blank pages text across the full 186 pages
    t_parse0 = time.perf_counter()
    page_assets_map: dict[int, list[str]] = {}
    for spec in ASSET_CATALOG:
        for pno in set(spec.graphic_pages) | set(spec.caption_pages):
            page_assets_map.setdefault(pno, []).append(spec.asset_id)

    all_chunks: list[dict[str, Any]] = []
    skipped_pages: list[dict[str, Any]] = []
    indexed_text_pages = 0

    for pno in range(1, n_pages + 1):
        page = doc[pno - 1]
        norm = normalize_page_text(_extract_page_text(page))
        first_tok = norm.splitlines()[0].strip() if norm else ""
        printed_page = first_tok if re.fullmatch(r"(?:\d{1,3}|[ivxlcdm]+)", first_tok, re.I) else ""
        section = classify_section(pno)
        # Log genuinely blank / section-divider-only pages
        if len(norm) < 60 and pno not in page_assets_map:
            skipped_pages.append({"pdf_page": pno, "chars": len(norm), "reason": "blank_or_divider"})
            continue

        indexed_text_pages += 1
        # Render page PNG at 140 DPI if it is a figure/table/box page (or caption page)
        page_png_rel: str | None = None
        if pno in page_assets_map or len(page.get_drawings()) > 40 or len(page.get_image_info(xrefs=True)) > 0:
            page_png = PAGES_DIR / f"page_{pno:03d}.png"
            if not page_png.exists():
                pix = page.get_pixmap(dpi=150)
                pix.save(str(page_png))
            page_png_rel = str(page_png.relative_to(STARTER_DIR.parent)).replace("\\", "/")

        linked = page_assets_map.get(pno, [])
        pieces = split_text_chunks(norm, target_chars=900, max_chars=1150, overlap_chars=150)
        for idx, piece in enumerate(pieces, 1):
            cid = f"{ADVANCED_SOURCE_ID}:p{pno:03d}:text:c{idx:03d}"
            all_chunks.append(
                {
                    "id": cid,
                    "source_id": ADVANCED_SOURCE_ID,
                    "pdf_page": pno,
                    "printed_page": printed_page,
                    "section": section,
                    "modality": "text",
                    "asset_id": None,
                    "title": f"{section} (PDF p.{pno})",
                    "content": piece,
                    "bbox_json": None,
                    "crop_path": None,
                    "page_render_path": page_png_rel,
                    "metadata": {
                        "pdf_page": pno,
                        "printed_page": printed_page,
                        "section": section,
                        "linked_assets": linked,
                    },
                }
            )

    # 2. Extract structured tables with pdfplumber and render 220-DPI graphic/table crops
    table_pages = {p for spec in ASSET_CATALOG if spec.modality == "table" for p in spec.graphic_pages}
    plumber_md_by_page = _extract_pdfplumber_tables(ADVANCED_PDF, table_pages)
    parse_seconds = round(time.perf_counter() - t_parse0, 3)

    vision_cache: dict[str, Any] = {}
    if VISION_CACHE_PATH.exists():
        vision_cache = json.loads(VISION_CACHE_PATH.read_text(encoding="utf-8"))

    t_vis0 = time.perf_counter()
    vis_in_tokens = 0
    vis_out_tokens = 0
    vis_api_calls = 0
    assets_registry: list[dict[str, Any]] = []

    for spec in ASSET_CATALOG:
        crop_paths: list[Path] = []
        render_paths: list[Path] = []
        bboxes: list[list[float]] = []

        for g_idx, gpage in enumerate(spec.graphic_pages, 1):
            page = doc[gpage - 1]
            # Render full graphic page at 150 DPI for provenance
            page_png = PAGES_DIR / f"page_{gpage:03d}.png"
            if not page_png.exists():
                page.get_pixmap(dpi=150).save(str(page_png))
            render_paths.append(page_png)

            rect = (
                pymupdf.Rect(*spec.crop_hint)
                if spec.crop_hint
                else _compute_graphic_rect(page, fallback_full=(spec.asset_id == "fig_1_1"))
            )
            bboxes.append([round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1)])
            suffix = f"_p{g_idx}" if len(spec.graphic_pages) > 1 else ""
            crop_png = CROPS_DIR / f"{spec.asset_id}{suffix}_p{gpage:03d}.png"
            if not crop_png.exists():
                # 220 DPI gives razor-sharp small labels/axis values/table cells
                page.get_pixmap(dpi=220, clip=rect).save(str(crop_png))
            crop_paths.append(crop_png)

        # Also ensure caption pages have full renders
        for cpage in spec.caption_pages:
            cp = PAGES_DIR / f"page_{cpage:03d}.png"
            if not cp.exists():
                doc[cpage - 1].get_pixmap(dpi=150).save(str(cp))

        caption_txt = _extract_caption_text(doc, spec.caption_pages, spec.title)
        plumber_tbl = "\n\n".join(
            plumber_md_by_page[p] for p in spec.graphic_pages if plumber_md_by_page.get(p)
        ).strip()

        cache_key = f"{spec.asset_id}::{cfg.vision_model}"
        if cache_key in vision_cache:
            vis_desc = vision_cache[cache_key]["description"]
        elif skip_vision:
            vis_desc = "(Vision extraction skipped)"
        else:
            if spec.modality == "table":
                vprompt = (
                    f"You are indexing an IPCC AR6 Synthesis Report table crop ({spec.title}).\n"
                    f"Caption context:\n{caption_txt[:1500]}\n\n"
                    f"Automated raw pdfplumber text hint (may have column alignment noise — verify visually against the PNG):\n"
                    f"{plumber_tbl[:2500]}\n\n"
                    "Task: Transcribe the COMPLETE table faithfully into clean GitHub Markdown preserving EVERY row, column header, subheader, units, medians, ranges/percentiles, and key footnotes exactly as printed in the image. Do not omit or round numbers."
                )
            elif spec.modality == "chart":
                vprompt = (
                    f"You are indexing an IPCC AR6 Synthesis Report chart/figure ({spec.title}).\n"
                    f"Caption context:\n{caption_txt[:1500]}\n\n"
                    "Task: Provide a detailed, factual structured extraction of EVERYTHING visible in the chart(s): "
                    "(1) all panels (a, b, c...), titles, and chart types; "
                    "(2) x-axis and y-axis labels, scales, and units; "
                    "(3) legend categories/color codes (e.g. C1-C8, SSP1-1.9..SSP5-8.5, risk levels); "
                    "(4) exact readable values, ticks, percentages, years, error bars/ranges, and threshold intersections; "
                    "(5) key annotations and uncertainty qualifiers."
                )
            else:
                vprompt = (
                    f"You are indexing an IPCC AR6 Synthesis Report diagram/map/matrix ({spec.title}).\n"
                    f"Caption context:\n{caption_txt[:1500]}\n\n"
                    "Task: Extract a thorough structured description of EVERYTHING shown: "
                    "(1) all panels, maps, legends, color scales, or icon keys; "
                    "(2) spatial patterns across regions/warming levels (1.5C, 2C, 3C, 4C) OR diagram flow steps/arrows/decisions; "
                    "(3) every labelled item, category, confidence symbol, and relationship visible in the crop."
                )
            vis_desc, in_t, out_t = _call_gemini_vision(cfg, crop_paths, vprompt)
            vis_in_tokens += in_t
            vis_out_tokens += out_t
            vis_api_calls += 1
            vision_cache[cache_key] = {
                "asset_id": spec.asset_id,
                "model": cfg.vision_model,
                "description": vis_desc,
                "prompt_tokens": in_t,
                "completion_tokens": out_t,
            }
            VISION_CACHE_PATH.write_text(
                json.dumps(vision_cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            time.sleep(0.5)  # gentle pacing on RPM

        primary_page = spec.graphic_pages[0]
        rel_crops = [str(p.relative_to(STARTER_DIR.parent)).replace("\\", "/") for p in crop_paths]
        rel_renders = [str(p.relative_to(STARTER_DIR.parent)).replace("\\", "/") for p in render_paths]

        combined_content = f"TITLE: {spec.title}\nASSET_ID: {spec.asset_id}\nMODALITY: {spec.modality}\nPDF_PAGES: {list(spec.graphic_pages)} (caption pages: {list(spec.caption_pages)})\n"
        if caption_txt:
            combined_content += f"\nCAPTION:\n{caption_txt}\n"
        if plumber_tbl:
            combined_content += f"\nEXTRACTED_TABLE_RAW:\n{plumber_tbl}\n"
        combined_content += f"\nVISUAL_STRUCTURED_EXTRACTION:\n{vis_desc}\n"

        assets_registry.append(
            {
                "asset_id": spec.asset_id,
                "title": spec.title,
                "modality": spec.modality,
                "graphic_pages": list(spec.graphic_pages),
                "caption_pages": list(spec.caption_pages),
                "bboxes": bboxes,
                "crop_paths": rel_crops,
                "page_render_paths": rel_renders,
                "caption": caption_txt,
                "vision_description": vis_desc,
            }
        )

        # Split visual structured extraction into indexed chunks while sharing crop/render provenance
        asset_pieces = split_text_chunks(combined_content, target_chars=1400, max_chars=1800, overlap_chars=200)
        for aidx, apiece in enumerate(asset_pieces, 1):
            cid = f"{ADVANCED_SOURCE_ID}:p{primary_page:03d}:{spec.modality}:{spec.asset_id}:c{aidx:02d}"
            all_chunks.append(
                {
                    "id": cid,
                    "source_id": ADVANCED_SOURCE_ID,
                    "pdf_page": primary_page,
                    "printed_page": "",
                    "section": classify_section(primary_page),
                    "modality": spec.modality,
                    "asset_id": spec.asset_id,
                    "title": spec.title,
                    "content": apiece,
                    "bbox_json": bboxes[0] if bboxes else None,
                    "crop_path": rel_crops[0] if rel_crops else None,
                    "page_render_path": rel_renders[0] if rel_renders else None,
                    "metadata": {
                        "graphic_pages": list(spec.graphic_pages),
                        "caption_pages": list(spec.caption_pages),
                        "all_crop_paths": rel_crops,
                        "all_render_paths": rel_renders,
                        "asset_id": spec.asset_id,
                    },
                }
            )

    vision_seconds = round(time.perf_counter() - t_vis0, 3)

    # 3. Embed all chunks with SHA-256 disk cache
    t_emb0 = time.perf_counter()
    embed_inputs = [
        f"[{c['modality'].upper()} | {c['section']} | PDF p.{c['pdf_page']}] {c['title']}\n{c['content']}"
        for c in all_chunks
    ]
    embeddings, emb_tokens, emb_cost = _embed_texts_cached(cfg, embed_inputs)
    embed_seconds = round(time.perf_counter() - t_emb0, 3)

    # 4. Persist to PostgreSQL + local JSON artifacts
    t_db0 = time.perf_counter()
    with psycopg.connect(cfg.database_url) as conn:
        ensure_advanced_schema(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM advanced_chunks WHERE source_id = %s;", (ADVANCED_SOURCE_ID,))
            for c, vec in zip(all_chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO advanced_chunks
                    (id, source_id, pdf_page, printed_page, section, modality, asset_id, title, content, bbox_json, crop_path, page_render_path, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb, %s::vector)
                    """,
                    (
                        c["id"],
                        c["source_id"],
                        c["pdf_page"],
                        c["printed_page"],
                        c["section"],
                        c["modality"],
                        c["asset_id"],
                        c["title"],
                        c["content"],
                        json.dumps(c["bbox_json"]) if c["bbox_json"] is not None else None,
                        c["crop_path"],
                        c["page_render_path"],
                        json.dumps(c["metadata"], ensure_ascii=False),
                        _vec_literal(vec),
                    ),
                )
        conn.commit()
    db_seconds = round(time.perf_counter() - t_db0, 3)

    ASSETS_JSON_PATH.write_text(json.dumps(assets_registry, ensure_ascii=False, indent=2), encoding="utf-8")
    CHUNKS_JSON_PATH.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    vis_cost = estimate_llm_cost_usd(cfg.vision_model, vis_in_tokens, vis_out_tokens)
    by_mod: dict[str, int] = {}
    by_sec: dict[str, int] = {}
    for c in all_chunks:
        by_mod[c["modality"]] = by_mod.get(c["modality"], 0) + 1
        by_sec[c["section"]] = by_sec.get(c["section"], 0) + 1

    manifest = {
        "source_id": ADVANCED_SOURCE_ID,
        "pdf_file": str(ADVANCED_PDF.relative_to(STARTER_DIR.parent)).replace("\\", "/"),
        "total_pdf_pages": n_pages,
        "indexed_text_pages": indexed_text_pages,
        "skipped_blank_or_divider_pages": len(skipped_pages),
        "skipped_pages_detail": skipped_pages,
        "visual_assets_indexed": len(assets_registry),
        "total_chunks": len(all_chunks),
        "chunks_by_modality": by_mod,
        "chunks_by_section": by_sec,
        "embedding_model": cfg.embedding_model,
        "vision_model": cfg.vision_model,
        "timings_seconds": {
            "parse_and_crop": parse_seconds,
            "vision_extraction": vision_seconds,
            "embedding": embed_seconds,
            "postgres_upsert": db_seconds,
            "total": round(parse_seconds + vision_seconds + embed_seconds + db_seconds, 3),
        },
        "usage": {
            "vision_api_calls": vis_api_calls,
            "vision_prompt_tokens": vis_in_tokens,
            "vision_completion_tokens": vis_out_tokens,
            "vision_estimated_cost_usd": vis_cost,
            "embed_tokens": emb_tokens,
            "embed_estimated_cost_usd": emb_cost,
            "total_estimated_cost_usd": round(vis_cost + emb_cost, 8),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Ingest IPCC AR6 SYR 186-page multimodal corpus")
    p.add_argument("--skip-vision", action="store_true", help="Skip Gemini Vision calls (local test mode)")
    args = p.parse_args()
    res = run_ingest(skip_vision=args.skip_vision)
    print(json.dumps(res, ensure_ascii=False, indent=2))
