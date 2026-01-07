from collections import defaultdict
from pathlib import Path

from src.utils.io import read_jsonl
PAGES = "data/interim/pages.jsonl"


PAGES_DATA = list(read_jsonl(PAGES))

page_lookup = {}
for p in PAGES_DATA:
    pdf = p.get("pdf_name") or p.get("pdf") or p.get("source")
    page = p.get("page")
    text = p.get("text_raw") or p.get("text") or ""

    if pdf is None or page is None:
        continue

    page_lookup[(Path(str(pdf)).name, int(page))] = text


def retrieve_small2big(question: str, retriever, *, expand_pages: int = 1, top_k_small: int = 8):
    """
    1) retrieve sur index small (chunks)
    2) expansion en contexte big: concat des pages voisines autour des hits
    """
    # 1) Small retrieve
    hits = retriever.retrieve(question)[:top_k_small]

    expanded = []
    seen = set()

    for h in hits:
        pdf = h.get("pdf_name")
        ps = int(h.get("page_start"))
        pe = int(h.get("page_end"))

        # élargit la fenêtre
        start = max(1, ps - expand_pages)
        end = pe + expand_pages

        key = (pdf, start, end)
        if key in seen:
            continue
        seen.add(key)

        parts = []
        for page in range(start, end + 1):
            t = page_lookup.get((pdf, page))
            if t:
                parts.append(t)

        big_text = "\n\n".join(parts).strip()
        if not big_text:
            continue

        expanded.append({
            "pdf_name": pdf,
            "page_start": start,
            "page_end": end,
            "score": float(h.get("score", 0.0)),
            "text": big_text,
            "seed_chunk_pages": (ps, pe),
        })

    # tri par score desc
    expanded.sort(key=lambda x: x["score"], reverse=True)
    return hits, expanded
