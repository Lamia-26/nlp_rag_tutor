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
    "Tu es un tuteur NLP.\n"
    "Tu dois répondre UNIQUEMENT à partir des sources fournies.\n"
    "Règles STRICTES:\n"
    "1) Interdiction d'inventer des définitions, formules ou notations.\n"
    "2) Si tu donnes une formule (ex: TF, IDF, TF-IDF), tu DOIS la recopier mot pour mot depuis les sources.\n"
    "3) Si la formule exacte n'est pas visible dans les sources extraites, dis: "
    "\"La formule exacte n'apparaît pas dans les extraits fournis\" et contente-toi d'expliquer l'intuition.\n"
    "4) Cite tes sources après chaque partie importante sous la forme (SOURCE k, pdf, pages).\n"
    "Structure obligatoire: Définition / Intuition / Formule (si disponible) / Exemple / À retenir.\n"
    "Ne pas halluciner.\n"
)

    user = f"Question:\n{question}\n\nSOURCES:\n{sources}\n\nRéponds."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

