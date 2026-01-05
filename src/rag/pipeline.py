from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from src.retrieval.embedder import EmbeddingConfig
from src.retrieval.retriever import Retriever, RetrieverConfig
from src.rag.prompt import tutor_messages
from src.rag.llm_groq import GroqConfig, GroqLLM

@dataclass(frozen=True)
class RagConfig:
    top_k: int = 8
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class RagTutor:
    def __init__(self, index_dir: Path, rag_cfg: RagConfig, llm_cfg: GroqConfig):
        self.retriever = Retriever(
            index_dir=index_dir,
            embed_cfg=EmbeddingConfig(model_name=rag_cfg.embed_model, normalize=True),
            cfg=RetrieverConfig(top_k=rag_cfg.top_k),
        )
        self.llm = GroqLLM(llm_cfg)

    def answer(self, question: str) -> Dict:
        hits = self.retriever.retrieve(question)
        messages = tutor_messages(question, hits)
        reply, usage = self.llm.chat(messages)
        sources = [{
            "pdf_name": h["pdf_name"],
            "page_start": h["page_start"],
            "page_end": h["page_end"],
            "score": h["score"],
            "chunk_id": h["chunk_id"],
        } for h in hits]
        return {"question": question, "answer": reply, "sources": sources, "usage": usage}
