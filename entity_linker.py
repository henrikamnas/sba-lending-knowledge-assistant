"""Entity linking: extract domain entities from free text and link to the knowledge graph.

Uses gpt-4o-mini to identify SBA lending entities in user-provided text,
then maps them to canonical nodes in the knowledge graph.
"""

import json

import networkx as nx
from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, ENTITY_TYPES, ENTITY_COLORS

ENTITY_EXTRACTION_PROMPT = """You are an expert at identifying SBA lending entities in text.

Given the following text, identify ALL domain-specific entities related to SBA lending.

Entity types: {entity_types}

For each entity found, provide:
- "text": the exact text span as it appears in the input
- "start": character offset where the entity starts
- "end": character offset where the entity ends
- "entity_type": one of the entity types listed above
- "canonical_name": the standard/official name for this entity (e.g., "SBA 7(a)" not "7a")
- "definition": a brief definition based on SBA regulations (1-2 sentences)

Rules:
- Find ALL entities, including amounts, percentages, program names, requirements, and terms
- Use exact character offsets matching the input text
- Be precise with start/end positions
- canonical_name should be the standard SBA term

Return valid JSON only, no markdown fences:
{{"entities": [...]}}

Text to analyze:
---
{text}
---"""


def extract_and_link_entities(text: str, graph: nx.DiGraph | None = None) -> list[dict]:
    """Extract entities from text and optionally link to knowledge graph nodes."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = ENTITY_EXTRACTION_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        text=text,
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    entities = result.get("entities", [])

    # Link to knowledge graph nodes if graph is available
    if graph and len(graph) > 0:
        for entity in entities:
            entity["kg_node_id"] = _find_matching_node(
                graph, entity.get("canonical_name", ""), entity.get("entity_type", "")
            )

    # Sort by start position
    entities.sort(key=lambda e: e.get("start", 0))

    return entities


def _find_matching_node(graph: nx.DiGraph, canonical_name: str, entity_type: str) -> str | None:
    """Find the best matching node in the knowledge graph."""
    canonical_lower = canonical_name.lower()

    # Exact label match
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("label", "").lower() == canonical_lower:
            return node_id

    # Partial match
    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get("label", "").lower()
        if canonical_lower in label or label in canonical_lower:
            if not entity_type or attrs.get("type") == entity_type:
                return node_id

    # Looser partial match ignoring type
    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get("label", "").lower()
        if canonical_lower in label or label in canonical_lower:
            return node_id

    return None


def highlight_entities_html(text: str, entities: list[dict]) -> str:
    """Produce annotated HTML with highlighted entity spans."""
    if not entities:
        return f"<p>{_escape_html(text)}</p>"

    # Sort by start position and handle overlaps
    sorted_entities = sorted(entities, key=lambda e: e.get("start", 0))

    html_parts = []
    last_end = 0

    for entity in sorted_entities:
        start = entity.get("start", 0)
        end = entity.get("end", 0)

        if start < last_end:
            continue  # Skip overlapping entities

        # Add text before entity
        if start > last_end:
            html_parts.append(_escape_html(text[last_end:start]))

        # Add highlighted entity
        entity_type = entity.get("entity_type", "Unknown")
        color = ENTITY_COLORS.get(entity_type, "#888888")
        entity_text = _escape_html(text[start:end])
        tooltip = _escape_html(entity.get("definition", ""))
        canonical = _escape_html(entity.get("canonical_name", ""))

        html_parts.append(
            f'<span style="background-color: {color}33; border-bottom: 2px solid {color}; '
            f'padding: 2px 4px; border-radius: 3px; cursor: help;" '
            f'title="{canonical}: {tooltip}">'
            f'{entity_text}</span>'
        )

        last_end = end

    # Add remaining text
    if last_end < len(text):
        html_parts.append(_escape_html(text[last_end:]))

    return f'<p style="line-height: 2; font-size: 16px;">{"".join(html_parts)}</p>'


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


EXAMPLE_TEXTS = [
    {
        "title": "Loan Application Excerpt",
        "text": (
            "We are applying for an SBA 7(a) loan in the amount of $2,000,000 to purchase "
            "commercial real estate for our manufacturing business. We currently have 200 employees "
            "and annual receipts of $15 million. Our company has been operating for 12 years and we "
            "have already invested significant personal equity into the business. We are seeking a "
            "25-year term with a fixed interest rate."
        ),
    },
    {
        "title": "Lender Email",
        "text": (
            "As a Preferred Lender (PLP), we can process your SBA Express loan application without "
            "prior SBA approval. The maximum amount for Express loans is $500,000 with a 50% SBA "
            "guaranty. We will need your SBA Form 1919, personal financial statements, and three "
            "years of business tax returns. The current interest rate would be prime plus 2.75%."
        ),
    },
    {
        "title": "Borrower Question",
        "text": (
            "I need a 7(a) loan for $2M to refinance existing debt and purchase new equipment. "
            "Can I prepay early without penalty? What is the guaranty fee on a loan this size? "
            "I heard the SBA guarantees up to 75% for loans over $150,000."
        ),
    },
]
