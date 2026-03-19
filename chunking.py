"""Intelligent document chunking with metadata preservation.

Implements section-aware chunking that respects document structure,
rather than naive fixed-size splitting. Each chunk carries rich metadata
for downstream retrieval and source attribution.
"""

import re

import tiktoken

from config import CHUNK_SIZE, CHUNK_OVERLAP

_encoder = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    return len(_encoder.encode(text))


def _detect_section_boundaries(text: str) -> list[int]:
    """Detect section boundaries based on common heading patterns.

    Looks for patterns like:
    - ALL CAPS lines (e.g., "ELIGIBILITY REQUIREMENTS")
    - Lines starting with numbers/letters followed by periods (e.g., "1. Overview")
    - Lines starting with "Section", "Chapter", "Part"
    - Double newlines followed by short lines (likely headings)
    """
    boundaries = [0]
    lines = text.split("\n")
    pos = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_boundary = False

        if stripped and len(stripped) < 100:
            # ALL CAPS heading
            if stripped.isupper() and len(stripped) > 3:
                is_boundary = True
            # Numbered section (e.g., "1. Overview", "A. Requirements")
            elif re.match(r"^[0-9]+\.\s+[A-Z]", stripped):
                is_boundary = True
            elif re.match(r"^[A-Z]\.\s+[A-Z]", stripped):
                is_boundary = True
            # Section/Chapter/Part headings
            elif re.match(r"^(Section|Chapter|Part|Article)\s+\d", stripped, re.IGNORECASE):
                is_boundary = True

        if is_boundary and pos > 0:
            boundaries.append(pos)

        pos += len(line) + 1  # +1 for newline

    return boundaries


def _extract_section_title(text: str) -> str:
    """Extract a section title from the beginning of a text chunk."""
    lines = text.strip().split("\n")
    for line in lines[:3]:
        stripped = line.strip()
        if stripped and len(stripped) < 100:
            if stripped.isupper() or re.match(r"^[0-9]+\.\s+", stripped):
                return stripped[:80]
    return ""


def chunk_document(
    text: str,
    source: str,
    base_metadata: dict | None = None,
) -> list[dict]:
    """Split a document into chunks with metadata.

    Uses a two-pass approach:
    1. First, detect natural section boundaries
    2. Then, split sections that exceed CHUNK_SIZE into smaller chunks
       with token-based overlap

    Args:
        text: Full document text.
        source: Source document identifier (filename).
        base_metadata: Additional metadata to attach to every chunk.

    Returns:
        List of chunk dicts with keys: text, metadata.
    """
    if base_metadata is None:
        base_metadata = {}

    # Clean text
    text = _clean_text(text)

    # Detect sections
    boundaries = _detect_section_boundaries(text)
    boundaries.append(len(text))

    sections = []
    for i in range(len(boundaries) - 1):
        section_text = text[boundaries[i]:boundaries[i + 1]].strip()
        if section_text:
            sections.append(section_text)

    # If no sections detected, treat entire text as one section
    if not sections:
        sections = [text]

    chunks = []
    chunk_index = 0

    for section in sections:
        section_title = _extract_section_title(section)
        section_chunks = _split_by_tokens(section, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_text in section_chunks:
            if not chunk_text.strip():
                continue

            metadata = {
                "source": source,
                "section_title": section_title,
                "chunk_index": chunk_index,
                "token_count": count_tokens(chunk_text),
                **base_metadata,
            }

            chunks.append({
                "text": chunk_text.strip(),
                "metadata": metadata,
            })
            chunk_index += 1

    return chunks


def _split_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into chunks of approximately max_tokens with overlap.

    Splits on sentence boundaries when possible to avoid cutting mid-sentence.
    """
    tokens = _encoder.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    # Split into sentences for cleaner boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_tokens = []
    current_text_parts = []

    for sentence in sentences:
        sentence_tokens = _encoder.encode(sentence)

        if len(current_tokens) + len(sentence_tokens) > max_tokens and current_tokens:
            # Emit current chunk
            chunks.append(" ".join(current_text_parts))

            # Calculate overlap: keep last overlap_tokens worth of sentences
            overlap_text_parts = []
            overlap_count = 0
            for part in reversed(current_text_parts):
                part_tokens = len(_encoder.encode(part))
                if overlap_count + part_tokens > overlap_tokens:
                    break
                overlap_text_parts.insert(0, part)
                overlap_count += part_tokens

            current_text_parts = overlap_text_parts
            current_tokens = _encoder.encode(" ".join(current_text_parts)) if current_text_parts else []

        current_text_parts.append(sentence)
        current_tokens = _encoder.encode(" ".join(current_text_parts))

    if current_text_parts:
        chunks.append(" ".join(current_text_parts))

    return chunks


def _clean_text(text: str) -> str:
    """Clean and normalize document text."""
    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    # Remove excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Remove page number patterns
    text = re.sub(r"\n\s*Page\s+\d+\s*(?:of\s+\d+)?\s*\n", "\n", text, flags=re.IGNORECASE)
    # Remove common header/footer artifacts
    text = re.sub(r"\n\s*Effective Date:.*?\n", "\n", text)
    return text.strip()
