"""Document ingestion pipeline.

Extracts text from PDF documents, processes them through the chunking
pipeline, generates embeddings, and indexes into ChromaDB.

Usage:
    python ingest.py
"""

import sys
from pathlib import Path

from pypdf import PdfReader

from config import DATA_DIR
from chunking import chunk_document, count_tokens
from embeddings import index_chunks, get_collection


def extract_pdf_text(pdf_path: Path) -> list[dict]:
    """Extract text from a PDF file with page-level metadata.

    Returns a list of dicts with 'text' and 'page_number' keys.
    """
    reader = PdfReader(str(pdf_path))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page_number": i + 1,
            })

    return pages


def extract_text_file(file_path: Path) -> list[dict]:
    """Extract text from a plain text file."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return [{"text": text, "page_number": 1}]


def process_document(file_path: Path) -> list[dict]:
    """Process a single document into chunks with metadata.

    Supports PDF and plain text files.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        pages = extract_pdf_text(file_path)
    elif suffix in (".txt", ".md"):
        pages = extract_text_file(file_path)
    else:
        print(f"  Skipping unsupported file type: {file_path.name}")
        return []

    if not pages:
        print(f"  No text extracted from {file_path.name}")
        return []

    # Combine all pages into a single text for chunking
    # but track page boundaries for metadata
    full_text = "\n\n".join(p["text"] for p in pages)
    total_tokens = count_tokens(full_text)

    print(f"  Extracted {len(pages)} pages, {total_tokens} tokens")

    # Build page offset map for assigning page numbers to chunks
    page_map = _build_page_map(pages, full_text)

    # Chunk the document
    chunks = chunk_document(
        text=full_text,
        source=file_path.name,
        base_metadata={"file_path": str(file_path.name)},
    )

    # Enrich chunks with page numbers
    for chunk in chunks:
        chunk_start = full_text.find(chunk["text"][:50])
        if chunk_start >= 0:
            chunk["metadata"]["page_number"] = _get_page_for_offset(page_map, chunk_start)

    return chunks


def _build_page_map(pages: list[dict], full_text: str) -> list[tuple[int, int]]:
    """Build a map of character offsets to page numbers."""
    page_map = []
    offset = 0
    for page in pages:
        page_map.append((offset, page["page_number"]))
        offset += len(page["text"]) + 2  # +2 for \n\n separator
    return page_map


def _get_page_for_offset(page_map: list[tuple[int, int]], offset: int) -> int:
    """Get the page number for a given character offset."""
    page_num = 1
    for start, pn in page_map:
        if start <= offset:
            page_num = pn
        else:
            break
    return page_num


def run_ingestion():
    """Run the full ingestion pipeline."""
    print("=" * 60)
    print("SBA Lending Document Ingestion Pipeline")
    print("=" * 60)

    # Find all documents
    if not DATA_DIR.exists():
        print(f"\nError: Data directory not found: {DATA_DIR}")
        print("Please add PDF or text documents to the data/raw/ directory.")
        sys.exit(1)

    files = list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.md"))

    if not files:
        print(f"\nNo documents found in {DATA_DIR}")
        print("Please add PDF or text documents to the data/raw/ directory.")
        sys.exit(1)

    print(f"\nFound {len(files)} document(s):")
    for f in files:
        print(f"  - {f.name}")

    # Process all documents
    all_chunks = []
    for file_path in files:
        print(f"\nProcessing: {file_path.name}")
        chunks = process_document(file_path)
        all_chunks.extend(chunks)
        print(f"  Generated {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks generated. Check your documents.")
        sys.exit(1)

    # Index into ChromaDB
    print("\nIndexing into ChromaDB...")
    collection = get_collection()
    indexed = index_chunks(all_chunks, collection)

    print(f"\nDone! {indexed} chunks in ChromaDB collection '{collection.name}'")
    print(f"Collection stored at: {CHROMA_DIR}")


if __name__ == "__main__":
    run_ingestion()
