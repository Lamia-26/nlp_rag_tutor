from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple

from src.utils.text import (
    is_page_number_line,
    split_and_clean_lines,
    fix_hyphenation_across_lines,
    join_lines,
)

@dataclass
class CleaningStats:
    removed_num_lines: int = 0
    removed_header_lines: int = 0
    removed_footer_lines: int = 0
    kept_lines: int = 0

def _collect_candidates(pages_lines: List[List[str]], header_n: int, footer_n: int) -> Tuple[Counter, Counter]:
    hc, fc = Counter(), Counter()
    for lines in pages_lines:
        if not lines:
            continue
        for ln in lines[:header_n]:
            hc[ln] += 1
        for ln in lines[-footer_n:]:
            fc[ln] += 1
    return hc, fc

def _frequent(counter: Counter, n_pages: int, min_ratio: float, max_len: int) -> Set[str]:
    if n_pages <= 0:
        return set()
    thr = max(3, int(n_pages * min_ratio))
    out = set()
    for ln, c in counter.items():
        if c >= thr and len(ln) <= max_len:
            out.add(ln)
    return out

def clean_pdf_pages(
    raw_pages: List[str],
    *,
    header_n: int = 2,
    footer_n: int = 2,
    min_ratio: float = 0.6,
    max_len: int = 90,
) -> Tuple[List[str], Dict]:
    """
    Livre long: on retire numéros de page et headers/footers fréquents.
    (SLP3 a souvent des headers/footers répétitifs selon la version.)
    """
    pages_lines = [split_and_clean_lines(t) for t in raw_pages]
    n_pages = len(pages_lines)

    hc, fc = _collect_candidates(pages_lines, header_n, footer_n)
    frequent_headers = _frequent(hc, n_pages, min_ratio, max_len)
    frequent_footers = _frequent(fc, n_pages, min_ratio, max_len)

    stats = CleaningStats()
    cleaned_pages: List[str] = []

    for lines in pages_lines:
        if not lines:
            cleaned_pages.append("")
            continue

        # remove page numbers anywhere (books often have them alone)
        tmp = []
        for ln in lines:
            if is_page_number_line(ln):
                stats.removed_num_lines += 1
                continue
            tmp.append(ln)
        lines = tmp

        # remove frequent headers/footers
        tmp = []
        for ln in lines:
            if ln in frequent_headers:
                stats.removed_header_lines += 1
                continue
            if ln in frequent_footers:
                stats.removed_footer_lines += 1
                continue
            tmp.append(ln)
        lines = tmp

        # fix hyphenation (book)
        lines = fix_hyphenation_across_lines(lines)

        stats.kept_lines += len(lines)
        cleaned_pages.append(join_lines(lines))

    debug = {
        "stats": asdict(stats),
        "frequent_headers": sorted(list(frequent_headers))[:30],
        "frequent_footers": sorted(list(frequent_footers))[:30],
    }
    return cleaned_pages, debug
