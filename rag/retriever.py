import json
from pathlib import Path
import numpy as np
import faiss

from rag.openai_client import embed_text

INDEX_PATH = Path("storage/index.faiss")
META_PATH = Path("storage/index_meta.jsonl")


def load_metadata():
    meta = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def retrieve(query: str, top_k: int = 5):
    # embed query
    q = np.array([embed_text(query)], dtype="float32")
    faiss.normalize_L2(q)

    # load index + metadata
    index = faiss.read_index(str(INDEX_PATH))
    meta = load_metadata()

    # search
    scores, ids = index.search(q, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        item = meta[idx]
        results.append({
            "score": float(score),
            "doc": item["doc"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
        })

    return results


if __name__ == "__main__":
    question = "What is evidence based medicine?"
    hits = retrieve(question, top_k=5)

    print("\n🔎 TOP RESULTS:")
    for h in hits:
        print(f"\nScore: {h['score']:.4f} | {h['doc']} (page {h['page']})")
        print(h["text"][:300], "...")
