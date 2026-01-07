from __future__ import annotations
import argparse
from pathlib import Path
import json

from src.config import DEFAULT_PATHS
from src.utils.io import read_jsonl, write_jsonl
from src.ingestion.extract_pdf import ingest_single_pdf
from src.chunking.chunker import chunk_pages, ChunkConfig
from src.retrieval.build_index import build_index
from src.retrieval.embedder import EmbeddingConfig
from src.retrieval.retriever import Retriever, RetrieverConfig
from src.rag.pipeline import RagTutor, RagConfig
from src.rag.llm_groq import GroqConfig
from src.eval.evaluate import run_evaluation, EvalConfig


def _find_single_pdf(pdf_dir: Path) -> Path:
    pdfs = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])
    if len(pdfs) != 1:
        raise RuntimeError(f"Expected exactly 1 PDF in {pdf_dir}, found {len(pdfs)}")
    return pdfs[0]

def cmd_ingest(args: argparse.Namespace) -> None:
    pdf_dir = Path(args.pdf_dir)
    out = Path(args.out)
    pdf_path = _find_single_pdf(pdf_dir)
    n, debug = ingest_single_pdf(pdf_path, out)
    print(f"[OK] pages={n} -> {out}")
    print(json.dumps(debug, ensure_ascii=False, indent=2))

def cmd_chunk(args: argparse.Namespace) -> None:
    pages = list(read_jsonl(Path(args.pages)))
    cfg = ChunkConfig(max_chars=args.max_chars, overlap_chars=args.overlap_chars, min_chars=args.min_chars)
    chunks = chunk_pages(pages, cfg)
    write_jsonl(Path(args.out), chunks)
    print(f"[OK] chunks={len(chunks)} -> {args.out}")

def cmd_index(args: argparse.Namespace) -> None:
    n, dim = build_index(
        chunks_jsonl=Path(args.chunks),
        index_dir=Path(args.index_dir),
        embed_cfg=EmbeddingConfig(model_name=args.embed_model, normalize=True),
    )
    print(f"[OK] indexed chunks={n} dim={dim} -> {args.index_dir}")

def cmd_search(args: argparse.Namespace) -> None:
    r = Retriever(
        index_dir=Path(args.index_dir),
        embed_cfg=EmbeddingConfig(model_name=args.embed_model, normalize=True),
        cfg=RetrieverConfig(top_k=args.top_k),
    )
    hits = r.retrieve(args.query)
    for i, h in enumerate(hits, start=1):
        print("-" * 90)
        print(f"#{i} score={h['score']:.4f} pages={h['page_start']}-{h['page_end']} chunk={h['chunk_id']}")
        txt = (h["text"] or "").replace("\n", " ")
        print(txt[:450] + ("..." if len(txt) > 450 else ""))


def cmd_ask(args: argparse.Namespace) -> None:
    tutor = RagTutor(
        index_dir=Path(args.index_dir),
        rag_cfg=RagConfig(top_k=args.top_k, embed_model=args.embed_model),
        llm_cfg=GroqConfig(model=args.llm_model, temperature=args.temperature, max_tokens=args.max_tokens),
    )
    out = tutor.answer(args.question)
    print("\n=== ANSWER ===\n")
    print(out["answer"])
    print("\n=== SOURCES ===")
    for i, s in enumerate(out["sources"], start=1):
        print(f"{i}. {s['pdf_name']} pages {s['page_start']}-{s['page_end']} score={s['score']:.4f}")
    if out.get("usage"):
        print("\n=== USAGE ===")
        print(out["usage"])
        
        
def cmd_evaluate(args: argparse.Namespace) -> None:
    metrics = run_evaluation(
        index_dir=Path(args.index_dir),
        cfg=EvalConfig(
            questions_csv=Path(args.questions_csv),
            out_dir=Path(args.out_dir),
            top_k=args.top_k,
            embed_model=args.embed_model,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
    )
    print("[OK] Evaluation done.")
    for k, v in metrics.items():
        print(f"{k}: {v}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("nlp-rag-tutor (single-PDF mode)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest")
    s.add_argument("--pdf_dir", default=str(DEFAULT_PATHS.pdf_dir))
    s.add_argument("--out", default=str(DEFAULT_PATHS.pages_jsonl))
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("chunk")
    s.add_argument("--pages", default=str(DEFAULT_PATHS.pages_jsonl))
    s.add_argument("--out", default=str(DEFAULT_PATHS.chunks_jsonl))
    s.add_argument("--max_chars", type=int, default=2500)
    s.add_argument("--overlap_chars", type=int, default=300)
    s.add_argument("--min_chars", type=int, default=300)
    s.set_defaults(func=cmd_chunk)

    s = sub.add_parser("index")
    s.add_argument("--chunks", default=str(DEFAULT_PATHS.chunks_jsonl))
    s.add_argument("--index_dir", default=str(DEFAULT_PATHS.index_dir))
    s.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--index_dir", default=str(DEFAULT_PATHS.index_dir))
    s.add_argument("--top_k", type=int, default=8)
    s.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("ask")
    s.add_argument("question")
    s.add_argument("--index_dir", default=str(DEFAULT_PATHS.index_dir))
    s.add_argument("--top_k", type=int, default=8)
    s.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    s.add_argument("--llm_model", default="llama-3.1-8b-instant")
    s.add_argument("--temperature", type=float, default=0.2)
    s.add_argument("--max_tokens", type=int, default=700)
    s.set_defaults(func=cmd_ask)
    
    s = sub.add_parser("evaluate", help="Evaluate retrieval (Recall@k, MRR) and optionally generation")
    s.add_argument("--questions_csv", default="data/eval/questions.csv")
    s.add_argument("--out_dir", default="data/eval/run_001")
    s.add_argument("--index_dir", default=str(DEFAULT_PATHS.index_dir))
    s.add_argument("--top_k", type=int, default=8)
    s.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")

    s.add_argument("--use_llm", action="store_true")
    s.add_argument("--llm_model", default="llama-3.1-8b-instant")
    s.add_argument("--temperature", type=float, default=0.2)
    s.add_argument("--max_tokens", type=int, default=600)

    s.set_defaults(func=cmd_evaluate)

    return p

def main() -> None:
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
