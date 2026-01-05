from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    pdf_dir: Path = Path("data/raw/pdfs")
    pages_jsonl: Path = Path("data/interim/pages.jsonl")
    chunks_jsonl: Path = Path("data/interim/chunks.jsonl")
    index_dir: Path = Path("data/index/faiss")

DEFAULT_PATHS = Paths()
