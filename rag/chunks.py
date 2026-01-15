import json
from pathlib import Path
from preprocess import load_pdfs, extract_text_with_pages, clean_text, chunk_text

CHUNKS_PATH = Path("storage/chunks.jsonl")


def ensure_storage_dir():
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)


def write_chunks_jsonl():
    """
    Export all chunks with metadata to storage/chunks.jsonl
    One JSON object per line (JSONL).
    """
    ensure_storage_dir()

    pdfs = load_pdfs()
    total_chunks = 0

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for pdf_path in pdfs:
            pages = extract_text_with_pages(pdf_path)

            chunk_id = 0
            for page_num, page_text in pages:
                cleaned = clean_text(page_text)
                if not cleaned:
                    continue

                chunks = chunk_text(cleaned)
                for chunk in chunks:
                    record = {
                        "doc": pdf_path.name,
                        "page": page_num,
                        "chunk_id": chunk_id,
                        "text": chunk,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_id += 1
                    total_chunks += 1

    print(f"✅ Wrote {total_chunks} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    write_chunks_jsonl()