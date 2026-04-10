"""Configuration and constants for the RAG Knowledge Assistant."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_DIMENSIONS = 1536

# Chunking
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens

# Retrieval
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

# ChromaDB
COLLECTION_NAME = "sba_lending_docs"

# Knowledge Graph
KG_DATA_DIR = PROJECT_ROOT / "kg_data"
KG_FILE = KG_DATA_DIR / "sba_knowledge_graph.json"
ENTITY_TYPES = [
    "LoanProgram",
    "Requirement",
    "LenderType",
    "FinancialTerm",
    "Regulation",
    "Amount",
]

ENTITY_COLORS = {
    "LoanProgram": "#2563EB",
    "Requirement": "#D97706",
    "LenderType": "#059669",
    "FinancialTerm": "#DC2626",
    "Regulation": "#7C3AED",
    "Amount": "#0891B2",
}
