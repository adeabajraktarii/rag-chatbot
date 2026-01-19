import json
from pathlib import Path
import faiss

# repo root = .../rag-chatbot
ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "storage"

INDEX_PATH = STORAGE / "index.faiss"
META_PATH = STORAGE / "index_meta.jsonl"


def load_metadata(path: Path = META_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {path}. "
            "Make sure storage/index_meta.jsonl is committed to GitHub."
        )

    meta: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


def load_index(path: Path = INDEX_PATH) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FAISS index file: {path}. "
            "Make sure storage/index.faiss is committed to GitHub."
        )
    return faiss.read_index(str(path))

