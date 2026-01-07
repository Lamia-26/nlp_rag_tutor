from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.retrieval.embedder import EmbeddingConfig
from src.retrieval.retriever import Retriever, RetrieverConfig
from src.eval.metrics import (
    eval_retrieval_by_keyword,
    mrr_from_ranks,
    recall_at_k,
    exact_match,
    answer_contains,
)

from src.rag.pipeline import RagTutor, RagConfig
from src.rag.llm_groq import GroqConfig


@dataclass(frozen=True)
class EvalConfig:
    questions_csv: Path
    out_dir: Path
    top_k: int = 8
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_llm: bool = False
    llm_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2
    max_tokens: int = 600


def read_questions(csv_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "question": (r.get("question") or "").strip(),
                "gold_keyword": (r.get("gold_keyword") or "").strip(),
                "gold_answer": (r.get("gold_answer") or "").strip(),
            })
    return [r for r in rows if r["question"]]


def run_evaluation(index_dir: Path, cfg: EvalConfig) -> Dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_questions(cfg.questions_csv)
    if not rows:
        raise ValueError("questions.csv is empty or invalid.")

    retriever = Retriever(
        index_dir=index_dir,
        embed_cfg=EmbeddingConfig(model_name=cfg.embed_model, normalize=True),
        cfg=RetrieverConfig(top_k=cfg.top_k),
    )

    tutor: Optional[RagTutor] = None
    if cfg.use_llm:
        tutor = RagTutor(
            index_dir=index_dir,
            rag_cfg=RagConfig(top_k=cfg.top_k, embed_model=cfg.embed_model),
            llm_cfg=GroqConfig(model=cfg.llm_model, temperature=cfg.temperature, max_tokens=cfg.max_tokens),
        )

    per_row: List[Dict] = []
    hits_list: List[bool] = []
    ranks_list: List[Optional[int]] = []

    # génération (optionnel)
    em_list: List[bool] = []
    contains_list: List[bool] = []

    for r in rows:
        q = r["question"]
        gold_keyword = r["gold_keyword"]
        gold_answer = r["gold_answer"]

        hits = retriever.retrieve(q)
        hits_texts = [h.get("text", "") for h in hits]

        rr = eval_retrieval_by_keyword(hits_texts, gold_keyword)
        hits_list.append(rr.hit)
        ranks_list.append(rr.rank)

        out = {
            "question": q,
            "gold_keyword": gold_keyword,
            "hit@k": rr.hit,
            "rank_first_hit": rr.rank if rr.rank is not None else "",
            "top_sources": " | ".join([f"{h['pdf_name']}:{h['page_start']}-{h['page_end']}" for h in hits[:5]]),
        }

        if tutor is not None:
            rag_out = tutor.answer(q)
            pred = rag_out["answer"]
            out["pred_answer"] = pred
            out["gold_answer"] = gold_answer

            if gold_answer:
                em = exact_match(pred, gold_answer)
                cont = answer_contains(pred, gold_answer)
                em_list.append(em)
                contains_list.append(cont)
                out["exact_match"] = em
                out["answer_contains"] = cont
            else:
                out["exact_match"] = ""
                out["answer_contains"] = ""

        per_row.append(out)

    # aggregate metrics
    metrics = {
        "n_questions": len(rows),
        f"recall@{cfg.top_k}": recall_at_k(hits_list),
        "mrr": mrr_from_ranks(ranks_list),
    }

    if cfg.use_llm:
        if em_list:
            metrics["exact_match_rate"] = sum(1 for x in em_list if x) / len(em_list)
        if contains_list:
            metrics["answer_contains_rate"] = sum(1 for x in contains_list if x) / len(contains_list)

    out_rows_path = cfg.out_dir / "per_question.csv"
    with out_rows_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(per_row[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_row)

    out_metrics_path = cfg.out_dir / "metrics.txt"
    with out_metrics_path.open("w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    return metrics
