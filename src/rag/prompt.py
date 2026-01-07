from __future__ import annotations
from typing import Dict, List

def _format_sources(hits: List[Dict], max_chars: int = 1600) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        excerpt = (h.get("text") or "")[:max_chars]
        blocks.append(
            f"[SOURCE {i}] pdf={h.get('pdf_name')} pages={h.get('page_start')}-{h.get('page_end')} score={h.get('score'):.4f}\n"
            f"{excerpt}\n"
        )
    return "\n".join(blocks).strip()

def tutor_messages(question: str, hits: List[Dict]) -> List[Dict]:
    sources = _format_sources(hits)
    system = (
    "You are an NLP tutor.\n"
    "You must answer ONLY using the provided sources.\n"
    "STRICT rules:\n"
    "1) You are forbidden to invent definitions, formulas, or notation.\n"
    "2) If you provide a formula (e.g., TF, IDF, TF-IDF), you MUST copy it word-for-word from the sources.\n"
    "3) If the exact formula is not visible in the extracted sources, say: "
    "\"The exact formula does not appear in the provided excerpts\" and only explain the intuition.\n"
    "4) Cite your sources after each important part in the form (SOURCE k, pdf, pages).\n"
    "Required structure: Definition / Intuition / Formula (if available) / Example / Key takeaways.\n"
    "Do not hallucinate.\n"
)


    user = f"Question:\n{question}\n\nSOURCES:\n{sources}\n\nRéponds."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

