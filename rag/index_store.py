import json
from pathlib import Path
import faiss

INDEX_PATH = Path("storage/index.faiss")
META_PATH = Path("storage/index_meta.jsonl")


def load_metadata(path: Path = META_PATH) -> list[dict]:
    meta: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def load_index(path: Path = INDEX_PATH) -> faiss.Index:
    return faiss.read_index(str(path))
