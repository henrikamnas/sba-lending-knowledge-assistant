"""Embedding generation and ChromaDB indexing.

Handles batch embedding via OpenAI and storage in ChromaDB
with full metadata preservation.
"""

import hashlib
from typing import Optional

import chromadb
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
)


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(client: Optional[chromadb.PersistentClient] = None) -> chromadb.Collection:
    """Get or create the SBA documents collection."""
    if client is None:
        client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def generate_chunk_id(text: str, source: str, chunk_index: int) -> str:
    """Generate a deterministic ID for a chunk (for idempotent upserts)."""
    content = f"{source}::{chunk_index}::{text[:100]}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI API.

    Processes in batches to stay within API limits.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def index_chunks(
    chunks: list[dict],
    collection: Optional[chromadb.Collection] = None,
    batch_size: int = 100,
) -> int:
    """Index chunks into ChromaDB with embeddings.

    Args:
        chunks: List of chunk dicts with 'text' and 'metadata' keys.
        collection: ChromaDB collection (creates default if None).
        batch_size: Number of chunks to process at once.

    Returns:
        Number of chunks indexed.
    """
    if collection is None:
        collection = get_collection()

    # Check existing count for idempotency
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection already has {existing_count} chunks. Skipping indexing.")
        print("To re-index, delete the chroma_db directory and run again.")
        return existing_count

    total = len(chunks)
    indexed = 0

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids = [
            generate_chunk_id(c["text"], c["metadata"]["source"], c["metadata"]["chunk_index"])
            for c in batch
        ]

        # Generate embeddings
        embeddings = embed_texts(texts)

        # Convert metadata values to strings (ChromaDB requirement)
        clean_metadatas = []
        for m in metadatas:
            clean_metadatas.append({k: str(v) for k, v in m.items()})

        # Upsert into ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=clean_metadatas,
        )

        indexed += len(batch)
        print(f"  Indexed {indexed}/{total} chunks...")

    return indexed
