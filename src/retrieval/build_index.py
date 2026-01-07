from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.io import read_jsonl
from src.retrieval.embedder import Embedder, EmbeddingConfig
from src.retrieval.vectorstore_faiss import FaissVectorStore

def build_index(chunks_jsonl: Path, index_dir: Path, embed_cfg: EmbeddingConfig) -> Tuple[int, int]:
    chunks: List[Dict] = list(read_jsonl(chunks_jsonl))
    if not chunks:
        raise ValueError("No chunks to index.")

    texts = [c["text"] for c in chunks]
    metas = [{
        "chunk_id": c["chunk_id"],
        "doc_id": c["doc_id"],
        "pdf_name": c["pdf_name"],
        "page_start": c["page_start"],
        "page_end": c["page_end"],
        "text": c["text"], 
    } for c in chunks]

    emb = Embedder(embed_cfg).embed(texts, batch_size=32)
    store = FaissVectorStore(dim=emb.shape[1])
    store.add(emb, metas)
    store.save(index_dir)
    return len(chunks), emb.shape[1]
