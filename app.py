"""SBA Lending Knowledge Assistant — Streamlit App.

A RAG-powered assistant that answers questions about SBA 7(a)
loan regulations with source attribution, knowledge graph exploration,
and entity linking.
"""

import streamlit as st

from config import OPENAI_API_KEY

# --- Page Config ---
st.set_page_config(
    page_title="SBA Lending Knowledge Assistant",
    page_icon="📋",
    layout="wide",
)

# --- Helper to safely read Streamlit secrets ---
def _get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except FileNotFoundError:
        return default

# --- Simple Password Gate ---
APP_PASSWORD = _get_secret("APP_PASSWORD")
if APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("### Please enter the access code to continue")
        pwd = st.text_input("Access code", type="password")
        if pwd:
            if pwd == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect access code.")
        st.stop()

# --- Check API Key ---
api_key = OPENAI_API_KEY or _get_secret("OPENAI_API_KEY")
if not api_key:
    st.error("Please set your OpenAI API key in `.env` or Streamlit secrets.")
    st.stop()

# Override config if using Streamlit secrets
if not OPENAI_API_KEY and api_key:
    import config
    config.OPENAI_API_KEY = api_key

# --- Auto-index on first run if DB is empty ---
from embeddings import get_collection

@st.cache_resource(show_spinner="Indexing SBA documents for the first time...")
def _ensure_indexed():
    collection = get_collection()
    if collection.count() == 0:
        from ingest import run_ingestion
        run_ingestion()
    return True

_ensure_indexed()

# --- Navigation ---
pg = st.navigation([
    st.Page("pages/0_QA_Assistant.py", title="Q&A Assistant", icon="📋"),
    st.Page("pages/1_Knowledge_Graph.py", title="Knowledge Graph", icon="🔗"),
    st.Page("pages/2_Entity_Linker.py", title="Entity Linker", icon="🏷️"),
])

pg.run()
