from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import pdfplumber
from tqdm import tqdm

from src.ingestion.clean_pdf_text import clean_pdf_pages
from src.utils.io import write_jsonl

def _extract_pages(pdf_path: Path) -> List[str]:
    out: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(layout=True, x_tolerance=2, y_tolerance=3) or ""
            out.append(txt)
    return out


def ingest_single_pdf(pdf_path: Path, out_pages_jsonl: Path) -> Tuple[int, Dict]:
    raw_pages = _extract_pages(pdf_path)
    cleaned_pages, debug = clean_pdf_pages(raw_pages)

    doc_id = pdf_path.stem
    records: List[Dict] = []
    for i, (raw, clean) in enumerate(zip(raw_pages, cleaned_pages), start=1):
        records.append({
            "doc_id": doc_id,
            "pdf_name": pdf_path.name,
            "page": i,
            "text_raw": raw,
            "text_clean": clean,
        })

    write_jsonl(out_pages_jsonl, records)
    return len(records), {"pdf_name": pdf_path.name, "n_pages": len(raw_pages), **debug}
