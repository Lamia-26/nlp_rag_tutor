from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import faiss
import numpy as np

class FaissVectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine if normalized
        self.meta: List[Dict] = []

    def add(self, vectors: np.ndarray, metas: List[Dict]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        self.index.add(vectors)
        self.meta.extend(metas)

    def search(self, q: np.ndarray, top_k: int) -> List[Dict]:
        if q.ndim == 1:
            q = q.reshape(1, -1)
        scores, idxs = self.index.search(q.astype("float32"), top_k)
        res = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            m = dict(self.meta[idx])
            m["score"] = float(score)
            res.append(m)
        return res

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        with (index_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @staticmethod
    def load(index_dir: Path) -> "FaissVectorStore":
        index = faiss.read_index(str(index_dir / "index.faiss"))
        store = FaissVectorStore(index.d)
        store.index = index
        metas = []
        with (index_dir / "meta.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metas.append(json.loads(line))
        store.meta = metas
        return store
