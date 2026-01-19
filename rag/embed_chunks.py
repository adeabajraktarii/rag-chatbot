import json
from pathlib import Path
from rag.openai_client import embed_text

CHUNKS_PATH = Path("storage/chunks.jsonl")
OUT_PATH = Path("storage/embeddings.jsonl")


def embed_all_chunks():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with CHUNKS_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)

            text = record["text"]
            vector = embed_text(text)

            out_record = {
                "doc": record["doc"],
                "page": record["page"],
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "embedding": vector,
            }

            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            count += 1
            if count % 50 == 0:
                print(f"Embedded {count} chunks...")

    print(f"\n Done! Embedded {count} chunks → {OUT_PATH}")


if __name__ == "__main__":
    embed_all_chunks()
