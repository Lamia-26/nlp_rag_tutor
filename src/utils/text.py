from __future__ import annotations
import re
from typing import List

_WS_RE = re.compile(r"[ \t]+")
_NUM_ONLY_RE = re.compile(r"^\s*\d{1,5}\s*$")

def normalize_spaces(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = _WS_RE.sub(" ", s)
    return s.strip()

def is_page_number_line(s: str) -> bool:
    return bool(_NUM_ONLY_RE.match(s))

def fix_common_pdf_artifacts(s: str) -> str:
    # ligatures + odd chars sometimes present in PDFs
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = s.replace("\u200b", "")  # zero-width space
    return s

def split_and_clean_lines(text: str) -> List[str]:
    text = fix_common_pdf_artifacts(text or "")
    lines = []
    for ln in text.splitlines():
        ln = normalize_spaces(ln)
        if ln:
            lines.append(ln)
    return lines

def fix_hyphenation_across_lines(lines: List[str]) -> List[str]:
    """
    Merge 'classifi-' + 'cation' => 'classification' (common in books)
    """
    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if re.search(r"[A-Za-z]-$", cur) and re.match(r"^[a-z]", nxt):
                out.append(cur[:-1] + nxt)
                i += 2
                continue
        out.append(cur)
        i += 1
    return out

def join_lines(lines: List[str]) -> str:
    return "\n".join(lines).strip()
