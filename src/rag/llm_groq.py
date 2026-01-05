from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from groq import Groq

@dataclass(frozen=True)
class GroqConfig:
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2
    max_tokens: int = 700

class GroqLLM:
    def __init__(self, cfg: GroqConfig):
        load_dotenv()
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Missing GROQ_API_KEY (create .env).")
        self.client = Groq(api_key=key)
        self.cfg = cfg

    def chat(self, messages: List[Dict]) -> Tuple[str, Dict]:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        text = resp.choices[0].message.content or ""
        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        return text, usage
