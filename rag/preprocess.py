from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCS_PATH = Path("data/docs")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_pdfs(docs_path: Path = DOCS_PATH) -> list[Path]:
    """Load PDFs from data/docs and drop duplicates by filename."""
    pdfs = sorted(docs_path.glob("*.pdf"))

    unique: dict[str, Path] = {}
    duplicates: set[str] = set()

    for pdf in pdfs:
        if pdf.name in unique:
            duplicates.add(pdf.name)
        else:
            unique[pdf.name] = pdf

    if duplicates:
        print("\n Duplicate PDF filenames detected (only first copy used):")
        for name in sorted(duplicates):
            print(f" - {name}")

    return list(unique.values())


def extract_text_with_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Read a PDF and return a list of (page_number, page_text)."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f" Failed to read PDF: {pdf_path.name} ({e})")
        return []

    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))

    return pages


def clean_text(text: str) -> str:
    """Normalize whitespace and fix common PDF extraction artifacts."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)     # fix hyphenation
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)     # single newline 
    text = re.sub(r"[ \t]+", " ", text)              # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)           # collapse blank lines
    return text.strip()


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
