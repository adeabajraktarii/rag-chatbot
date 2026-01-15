import json
from pathlib import Path
import numpy as np
import faiss

EMB_PATH = Path("storage/embeddings.jsonl")
INDEX_PATH = Path("storage/index.faiss")
META_PATH = Path("storage/index_meta.jsonl")


def build_faiss_index():
    vectors = []
    meta = []

    with EMB_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            vectors.append(r["embedding"])
            meta.append({"doc": r["doc"], "page": r["page"], "chunk_id": r["chunk_id"], "text": r["text"]})

    X = np.array(vectors, dtype="float32")

    # Cosine similarity via inner product + L2-normalization
    faiss.normalize_L2(X)
    dim = X.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(X)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with META_PATH.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"✅ FAISS index saved: {INDEX_PATH}")
    print(f"✅ Metadata saved: {META_PATH}")
    print(f"✅ Vectors indexed: {index.ntotal} (dim={dim})")


if __name__ == "__main__":
    build_faiss_index()
