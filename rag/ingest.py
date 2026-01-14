# =========================
# Imports
# =========================
from pathlib import Path
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# Configuration
# =========================
DOCS_PATH = Path("data/docs")


# ============================================================
# 1. LOADING STAGE
#    - Find PDF files
#    - Remove duplicate filenames
# ============================================================
def load_pdfs():
    """
    Load PDFs from data/docs and drop duplicates by filename.
    Keeps the first occurrence and warns if duplicates exist.
    """
    pdfs = sorted(DOCS_PATH.glob("*.pdf"))

    unique = {}
    duplicates = []

    for pdf in pdfs:
        if pdf.name in unique:
            duplicates.append(pdf.name)
        else:
            unique[pdf.name] = pdf

    if duplicates:
        print("\n⚠️ Duplicate PDF filenames detected (only first copy used):")
        for name in sorted(set(duplicates)):
            print(f" - {name}")

    return list(unique.values())


def extract_text_with_pages(pdf_path: Path):
    """
    Read a PDF and return a list of (page_number, page_text).
    """
    reader = PdfReader(str(pdf_path))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))

    return pages


# ============================================================
# 2. CLEANING STAGE
#    - Normalize newlines
#    - Fix hyphenation
#    - Remove excess whitespace
# ============================================================
def clean_text(text: str) -> str:
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenation across line breaks: "infor-\nmation" → "information"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Replace single newlines inside paragraphs with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse multiple spaces or tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# 3. CHUNKING STAGE
#    - Split cleaned text into overlapping chunks
# ============================================================
def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# ============================================================
# Main ingestion flow
# ============================================================
def main():
    print("Ingestion started")

    pdfs = load_pdfs()
    print(f"Found {len(pdfs)} PDF files")

    total_chunks = 0

    # Process all PDFs (currently only counting chunks)
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

