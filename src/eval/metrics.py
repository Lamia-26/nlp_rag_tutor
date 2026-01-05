from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def keyword_in_text(keyword: str, text: str) -> bool:
    k = _norm(keyword)
    t = _norm(text)
    return k in t


@dataclass
class RetrievalRowResult:
    hit: bool
    rank: Optional[int]  # 1-based rank of first hit if any


def eval_retrieval_by_keyword(hits_texts: List[str], gold_keyword: str) -> RetrievalRowResult:
    if not gold_keyword:
        return RetrievalRowResult(hit=False, rank=None)

    for i, txt in enumerate(hits_texts, start=1):
        if keyword_in_text(gold_keyword, txt):
            return RetrievalRowResult(hit=True, rank=i)
    return RetrievalRowResult(hit=False, rank=None)


def mrr_from_ranks(ranks: List[Optional[int]]) -> float:
    vals = []
    for r in ranks:
        if r is None:
            vals.append(0.0)
        else:
            vals.append(1.0 / float(r))
    return sum(vals) / max(1, len(vals))


def recall_at_k(hits: List[bool]) -> float:
    return sum(1 for h in hits if h) / max(1, len(hits))


def exact_match(pred: str, gold: str) -> bool:
    return _norm(pred) == _norm(gold)


def answer_contains(pred: str, gold: str) -> bool:
    # plus permissif que Exact Match
    return _norm(gold) in _norm(pred)
