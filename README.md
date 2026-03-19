# SBA Lending Knowledge Assistant

A RAG (Retrieval-Augmented Generation) powered assistant that answers questions about SBA 7(a) loan regulations with source attribution. Built as an architecture demo for AI-driven knowledge management in fintech/lending.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐     ┌──────────┐
│  Streamlit   │────▶│  Retrieval   │────▶│  ChromaDB  │     │  OpenAI  │
│  Frontend    │     │  Pipeline    │     │ Vector DB  │     │   API    │
│              │◀────│              │◀────│            │     │          │
│  (app.py)    │     │(retrieval.py)│     │(embeddings)│     │(gpt-4o-  │
│              │────▶│              │────▶│            │     │  mini)   │
│              │     │   LLM Layer  │────────────────────────▶│          │
│              │◀────│  (llm.py)    │◀───────────────────────│          │
└─────────────┘     └──────────────┘     └────────────┘     └──────────┘
                           ▲
                    ┌──────┴───────┐
                    │  Ingestion   │
                    │  Pipeline    │
                    │ (ingest.py)  │
                    │ (chunking.py)│
                    └──────────────┘
```

**Key design decisions:**
- **Modular architecture** — separate concerns for ingestion, chunking, embedding, retrieval, and LLM
- **Metadata-rich chunks** — section-aware chunking with source, page, and section tracking
- **Source attribution** — every answer cites the specific document, section, and page
- **Keyword boosting** — domain-specific terms get relevance boosts in retrieval
- **Streaming responses** — tokens appear as they're generated for better UX
- **Idempotent indexing** — re-running ingestion won't duplicate data

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Vector DB | ChromaDB (embedded, persistent) |
| LLM | OpenAI gpt-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Language | Python 3.10+ |

## Quick Start

### 1. Install dependencies

```bash
cd rag-knowledge-assistant
pip install -r requirements.txt
```

### 2. Set up API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Ingest documents

```bash
python ingest.py
```

This processes the SBA documents in `data/raw/`, chunks them intelligently, generates embeddings, and indexes them in ChromaDB.

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Ask questions

Try these example queries:
- "What are the eligibility requirements for an SBA 7(a) loan?"
- "What is the maximum loan amount?"
- "What interest rates apply to SBA loans?"
- "What is the difference between PLP and CLP lenders?"
- "Can I prepay my SBA loan without penalty?"

## Project Structure

```
rag-knowledge-assistant/
├── app.py              # Streamlit main app
├── ingest.py           # Document ingestion pipeline
├── chunking.py         # Section-aware chunking with metadata
├── embeddings.py       # Embedding generation + ChromaDB indexing
├── retrieval.py        # Semantic search with keyword boosting
├── llm.py              # LLM integration with prompt templates
├── config.py           # Configuration constants
├── requirements.txt
├── .streamlit/
│   └── config.toml     # Streamlit theme
├── data/
│   └── raw/            # Source SBA documents
├── chroma_db/          # Persisted vector store (gitignored)
└── README.md
```

## Deployment (Streamlit Cloud)

1. Push to GitHub
2. Connect repo to [Streamlit Cloud](https://share.streamlit.io)
3. Add `OPENAI_API_KEY` in Streamlit secrets
4. Set main file path to `rag-knowledge-assistant/app.py`

## Author

Henrik Axelsson — AI Knowledge Data Engineer demo project
