from pathlib import Path
from pypdf import PdfReader

DOCS_PATH = Path("data/docs")

print("Ingestion started")

pdf_files = list(DOCS_PATH.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files")

for pdf in pdf_files:
    print(pdf.name)
