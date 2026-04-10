"""Q&A Assistant — RAG-powered SBA lending Q&A with source attribution."""

import streamlit as st

from config import LLM_MODEL, EMBEDDING_MODEL, TOP_K
from retrieval import retrieve, get_collection_stats
from llm import generate_response_stream

# --- Sidebar ---
with st.sidebar:
    st.markdown("Answers are sourced from official SBA 7(a) program documents.")

    # Collection stats
    stats = get_collection_stats()

    if stats["status"] == "ready":
        st.metric("Indexed Chunks", stats["total_chunks"])
        st.metric("Source Documents", stats["num_documents"])
        with st.expander("Documents"):
            for doc in stats["documents"]:
                st.markdown(f"- `{doc}`")
    else:
        st.warning("No documents indexed. Run `python ingest.py` first.")

    with st.expander("Technical details"):
        st.markdown(f"""
        - LLM: `{LLM_MODEL}`
        - Embeddings: `{EMBEDDING_MODEL}`
        - Retrieval: top-{TOP_K} chunks, cosine similarity
        """)

# --- Main Content ---
st.header("Q&A Assistant")

# --- Demo Questions ---
DEMO_QUESTIONS = [
    "What are the eligibility requirements for an SBA 7(a) loan?",
    "What is the maximum loan amount?",
    "Can I prepay my SBA loan without penalty?",
    "What is the difference between PLP and CLP lenders?",
    "What documents do I need to apply?",
    "What interest rates apply to SBA 7(a) loans?",
]

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Show demo questions when chat is empty ---
if not st.session_state.messages:
    st.markdown("Example questions:")
    cols = st.columns(2)
    for i, q in enumerate(DEMO_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"demo_{i}", use_container_width=True):
                st.session_state["_pending_question"] = q
                st.rerun()

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for src in message["sources"]:
                    score_pct = f"{src['score']:.0%}"
                    st.markdown(f"**{src['citation']}** (relevance: {score_pct})")
                    st.markdown(f"> {src['text'][:300]}...")

# --- Chat Input ---
_pending = st.session_state.pop("_pending_question", None)
if question := (_pending or st.chat_input("Ask about SBA 7(a) lending...")):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Searching knowledge base..."):
        results = retrieve(question)

    with st.chat_message("assistant"):
        if not results:
            response_text = (
                "I couldn't find relevant information in the knowledge base for that question. "
                "Try asking about SBA 7(a) loan eligibility, terms, rates, or lender requirements."
            )
            st.markdown(response_text)
        else:
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-6:]
            ]
            response_text = st.write_stream(
                generate_response_stream(question, results, chat_history)
            )

            source_data = []
            with st.expander("Sources"):
                for result in results:
                    score_pct = f"{result.relevance_score:.0%}"
                    st.markdown(f"**{result.citation}** (relevance: {score_pct})")
                    st.markdown(f"> {result.text[:300]}...")
                    source_data.append(
                        {
                            "citation": result.citation,
                            "score": result.relevance_score,
                            "text": result.text,
                        }
                    )

    msg = {"role": "assistant", "content": response_text}
    if results:
        msg["sources"] = source_data
    st.session_state.messages.append(msg)
