from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    normalize: bool = True

class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        arr = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=True,
        )
        return arr.astype("float32")
