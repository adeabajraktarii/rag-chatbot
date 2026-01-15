from preprocess import load_pdfs, extract_text_with_pages, clean_text, chunk_text


def main():
    print("Ingestion started")

    pdfs = load_pdfs()
    print(f"Found {len(pdfs)} PDF files")

    total_chunks = 0

    for pdf_path in pdfs:
        pages = extract_text_with_pages(pdf_path)

        doc_chunks = 0
        for page_num, page_text in pages:
            cleaned = clean_text(page_text)
            if not cleaned:
                continue

            chunks = chunk_text(cleaned)
            doc_chunks += len(chunks)

        total_chunks += doc_chunks
        print(f"{pdf_path.name}: {doc_chunks} chunks")

    print(f"\nTOTAL chunks across all docs: {total_chunks}")


if __name__ == "__main__":
    main()
