from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.retrieval.embedder import Embedder, EmbeddingConfig
from src.retrieval.vectorstore_faiss import FaissVectorStore

@dataclass(frozen=True)
class RetrieverConfig:
    top_k: int = 8

class Retriever:
    def __init__(self, index_dir: Path, embed_cfg: EmbeddingConfig, cfg: RetrieverConfig):
        self.cfg = cfg
        self.embedder = Embedder(embed_cfg)
        self.store = FaissVectorStore.load(index_dir)

    def retrieve(self, query: str) -> List[Dict]:
        qv = self.embedder.embed([query], batch_size=1)
        return self.store.search(qv, top_k=self.cfg.top_k)
