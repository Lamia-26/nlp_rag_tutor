from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from src.utils.text import normalize_spaces


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 2500
    overlap_chars: int = 300
    min_chars: int = 300


def _chunk_id(doc_id: str, p1: int, p2: int, i: int) -> str:
    return f"{doc_id}::p{p1:04d}-p{p2:04d}::c{i:04d}"


def chunk_pages(pages: List[Dict], cfg: ChunkConfig) -> List[Dict]:
    pages = sorted(pages, key=lambda r: int(r["page"]))
    if not pages:
        return []
    
    doc_id = pages[0]["doc_id"]
    pdf_name = pages[0]["pdf_name"]

    chunks: List[Dict] = []
    buf = ""
    p_start: int | None = None
    p_end: int | None = None
    idx = 0

    def flush(force: bool = False) -> None:
        nonlocal buf, p_start, p_end, idx

        text = normalize_spaces(buf)
        if not text:
            return
        if (not force) and len(text) < cfg.min_chars:
            return
        if p_start is None or p_end is None:
            return

        # snapshot pages for this chunk
        chunk_p_start = p_start
        chunk_p_end = p_end

        chunks.append({
            "chunk_id": _chunk_id(doc_id, chunk_p_start, chunk_p_end, idx),
            "doc_id": doc_id,
            "pdf_name": pdf_name,
            "page_start": chunk_p_start,
            "page_end": chunk_p_end,
            "text": text,
        })
        idx += 1

        if cfg.overlap_chars > 0 and len(buf) > cfg.overlap_chars:
            buf = buf[-cfg.overlap_chars:]
            p_start = chunk_p_end
            p_end = chunk_p_end
        else:
            buf = ""
            p_start = None
            p_end = None

    for rec in pages:
        page = int(rec["page"])
        text = (rec.get("text_clean") or "").strip()
        if not text:
            continue

        addition = f"\n[PAGE {page}]\n{text}\n"

        if p_start is None:
            p_start = page
        p_end = page

        if buf and (len(buf) + len(addition) > cfg.max_chars):
            flush(force=True)
            if p_start is None:
                p_start = page
            p_end = page

        buf += addition

        if len(buf) >= cfg.max_chars:
            flush(force=True)

    flush(force=True)
    return chunks
