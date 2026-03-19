"""Build the SBA knowledge graph by extracting entities and relations from documents.

Run once offline:  python build_knowledge_graph.py [--force]
Reads document chunks from ChromaDB and uses gpt-4o-mini to extract
structured entities and relations, then saves to kg_data/sba_knowledge_graph.json.
"""

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, KG_DATA_DIR, KG_FILE, ENTITY_TYPES
from embeddings import get_collection

EXTRACTION_PROMPT = """You are an expert at extracting structured knowledge from SBA lending documents.

Given the following document chunk, extract ALL entities and relationships.

Entity types: {entity_types}

Relation types:
- HAS_REQUIREMENT (LoanProgram -> Requirement)
- MAX_AMOUNT (LoanProgram -> Amount)
- APPLIES_TO (FinancialTerm -> LoanProgram)
- GOVERNED_BY (LoanProgram -> Regulation)
- TYPE_OF (LoanProgram -> LoanProgram, or LenderType -> LenderType)
- REQUIRES_DOCUMENT (LoanProgram -> Requirement)
- HAS_FEE (LoanProgram -> Amount)
- INTEREST_RATE (LoanProgram -> FinancialTerm)

Rules:
- Use canonical names (e.g., "SBA 7(a)" not "7a loan" or "the program")
- Each entity needs a unique id (snake_case), a label (display name), a type, and a brief definition
- Each relation needs source (entity id), target (entity id), and relation type
- Extract EVERY fact — loan amounts, percentages, requirements, programs, terms
- Be thorough but avoid duplicates within this chunk

Return valid JSON only, no markdown fences:
{{"nodes": [{{"id": "...", "label": "...", "type": "...", "definition": "..."}}], "edges": [{{"source": "...", "target": "...", "relation": "..."}}]}}

Document chunk:
---
{chunk_text}
---"""


def extract_from_chunk(client: OpenAI, chunk_text: str) -> dict:
    """Extract entities and relations from a single chunk."""
    prompt = EXTRACTION_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        chunk_text=chunk_text,
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return json.loads(content)


def merge_graphs(extractions: list[dict]) -> dict:
    """Merge multiple extraction results, deduplicating by node id."""
    nodes_by_id = {}
    edges_set = set()
    edges = []

    for extraction in extractions:
        for node in extraction.get("nodes", []):
            nid = node["id"]
            if nid not in nodes_by_id:
                nodes_by_id[nid] = node
            else:
                # Keep the longer definition
                existing_def = nodes_by_id[nid].get("definition", "")
                new_def = node.get("definition", "")
                if len(new_def) > len(existing_def):
                    nodes_by_id[nid]["definition"] = new_def

        for edge in extraction.get("edges", []):
            key = (edge["source"], edge["target"], edge["relation"])
            if key not in edges_set:
                edges_set.add(key)
                edges.append(edge)

    # Remove edges that reference non-existent nodes
    valid_ids = set(nodes_by_id.keys())
    edges = [e for e in edges if e["source"] in valid_ids and e["target"] in valid_ids]

    return {"nodes": list(nodes_by_id.values()), "edges": edges}


def build_knowledge_graph(force: bool = False) -> Path:
    """Build and save the knowledge graph."""
    if KG_FILE.exists() and not force:
        print(f"Knowledge graph already exists at {KG_FILE}")
        print("Use --force to rebuild.")
        return KG_FILE

    # Get chunks from ChromaDB
    collection = get_collection()
    count = collection.count()
    if count == 0:
        print("No documents indexed in ChromaDB. Run `python ingest.py` first.")
        sys.exit(1)

    print(f"Found {count} chunks in ChromaDB")
    results = collection.get(include=["documents"])
    chunks = results["documents"]

    # Extract entities from each chunk
    client = OpenAI(api_key=OPENAI_API_KEY)
    extractions = []

    for i, chunk in enumerate(chunks):
        print(f"  Extracting from chunk {i + 1}/{len(chunks)}...")
        try:
            extraction = extract_from_chunk(client, chunk)
            extractions.append(extraction)
        except Exception as e:
            print(f"  Warning: Failed on chunk {i + 1}: {e}")
            continue

    # Merge and deduplicate
    print("Merging and deduplicating...")
    graph = merge_graphs(extractions)
    print(f"  {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    # Save
    KG_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(KG_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print(f"Saved to {KG_FILE}")
    return KG_FILE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SBA knowledge graph")
    parser.add_argument("--force", action="store_true", help="Rebuild even if file exists")
    args = parser.parse_args()

    build_knowledge_graph(force=args.force)
