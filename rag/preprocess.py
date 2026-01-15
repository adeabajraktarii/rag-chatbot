from pathlib import Path
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_PATH = Path("data/docs")


# ============================================================
# 1) LOADING
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
# 2) CLEANING
# ============================================================
def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # fix hyphenation
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)          # single newline -> space
    text = re.sub(r"[ \t]+", " ", text)                   # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)                # collapse blank lines
    return text.strip()


# ============================================================
# 3) CHUNKING
# ============================================================
def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
