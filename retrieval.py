"""Retrieval pipeline for semantic search over SBA documents.

Provides ranked retrieval with relevance scores and source references,
with optional metadata filtering and keyword boosting.
"""

import re
from dataclasses import dataclass

from embeddings import get_collection, embed_texts
from config import TOP_K, SIMILARITY_THRESHOLD


@dataclass
class RetrievalResult:
    """A single retrieval result with text, metadata, and score."""
    text: str
    source: str
    section_title: str
    page_number: int
    chunk_index: int
    relevance_score: float

    @property
    def citation(self) -> str:
        """Format a citation string for this result."""
        parts = [self.source]
        if self.section_title:
            parts.append(f"§ {self.section_title}")
        if self.page_number:
            parts.append(f"p. {self.page_number}")
        return ", ".join(parts)


# Key financial/legal terms to boost relevance when they appear in both query and result
DOMAIN_TERMS = {
    "eligibility", "collateral", "guaranty", "guarantee", "interest rate",
    "loan amount", "maximum", "minimum", "sba", "7(a)", "borrower",
    "lender", "default", "repayment", "term", "maturity", "fee",
    "credit", "underwriting", "disbursement", "liquidation",
    "franchise", "startup", "refinance", "express", "advantage",
}


def _keyword_boost(query: str, document: str, boost_factor: float = 0.05) -> float:
    """Calculate a small relevance boost for domain term matches."""
    query_lower = query.lower()
    doc_lower = document.lower()

    boost = 0.0
    for term in DOMAIN_TERMS:
        if term in query_lower and term in doc_lower:
            boost += boost_factor

    return min(boost, 0.15)  # Cap the total boost


def retrieve(
    query: str,
    top_k: int = TOP_K,
    source_filter: str | None = None,
    min_score: float = SIMILARITY_THRESHOLD,
) -> list[RetrievalResult]:
    """Retrieve relevant document chunks for a query.

    Args:
        query: User's question.
        top_k: Number of results to return.
        source_filter: Optional filter by source document name.
        min_score: Minimum relevance score (0-1, higher is more relevant).

    Returns:
        List of RetrievalResult objects, ranked by relevance.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    # Build query parameters
    query_embedding = embed_texts([query])[0]

    where_filter = None
    if source_filter:
        where_filter = {"source": source_filter}

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k * 2, collection.count()),  # Fetch extra for post-filtering
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    # Process results
    retrieval_results = []
    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB returns cosine distance; convert to similarity score
        similarity = 1 - distance

        # Apply keyword boost
        boost = _keyword_boost(query, doc)
        final_score = similarity + boost

        if final_score < min_score:
            continue

        retrieval_results.append(RetrievalResult(
            text=doc,
            source=metadata.get("source", "Unknown"),
            section_title=metadata.get("section_title", ""),
            page_number=int(metadata.get("page_number", 0)),
            chunk_index=int(metadata.get("chunk_index", 0)),
            relevance_score=round(final_score, 4),
        ))

    # Sort by score descending and return top_k
    retrieval_results.sort(key=lambda r: r.relevance_score, reverse=True)
    return retrieval_results[:top_k]


def get_collection_stats() -> dict:
    """Get statistics about the indexed collection."""
    collection = get_collection()
    count = collection.count()

    if count == 0:
        return {"total_chunks": 0, "documents": [], "status": "empty"}

    # Get unique sources
    all_metadata = collection.get(include=["metadatas"])
    sources = set()
    for m in all_metadata["metadatas"]:
        sources.add(m.get("source", "Unknown"))

    return {
        "total_chunks": count,
        "documents": sorted(sources),
        "num_documents": len(sources),
        "status": "ready",
    }
