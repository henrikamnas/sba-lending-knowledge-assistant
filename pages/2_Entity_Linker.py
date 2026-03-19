"""Entity Linker — paste free text to identify and link SBA lending entities."""

import streamlit as st

from config import ENTITY_COLORS, KG_FILE
from entity_linker import (
    extract_and_link_entities,
    highlight_entities_html,
    EXAMPLE_TEXTS,
)
from knowledge_graph import load_graph

# --- Load KG if available ---
@st.cache_resource
def _load_kg():
    if KG_FILE.exists():
        return load_graph()
    return None

graph = _load_kg()

# --- Sidebar ---
with st.sidebar:
    st.title("Entity Linker")
    st.markdown(
        "Paste or type any text about SBA lending and this tool will identify "
        "domain-specific entities, classify them, and link them to the knowledge graph."
    )

    st.divider()

    # Legend
    st.subheader("Entity Types")
    for etype, color in ENTITY_COLORS.items():
        st.markdown(
            f'<span style="color:{color}; font-size:16px;">&#9679;</span> {etype}',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("Built by Henrik Axelsson as an AI/NER demo.")

# --- Main Content ---
st.title("Entity Linker")
st.markdown(
    "Identify and link SBA lending entities in free text. "
    "Entities are highlighted and mapped to official SBA definitions."
)

# --- Example buttons ---
st.markdown("**Try an example:**")
example_cols = st.columns(len(EXAMPLE_TEXTS))
for col, example in zip(example_cols, EXAMPLE_TEXTS):
    with col:
        if st.button(example["title"], key=f"ex_{example['title']}", use_container_width=True):
            st.session_state["el_text_area"] = example["text"]
            st.rerun()

# --- Text input ---
input_text = st.text_area(
    "Enter text to analyze",
    height=150,
    placeholder="Paste any text about SBA lending — loan applications, emails, questions...",
    key="el_text_area",
)

# --- Analyze ---
if st.button("Analyze Entities", type="primary", disabled=not input_text):
    with st.spinner("Extracting and linking entities..."):
        entities = extract_and_link_entities(input_text, graph)
        st.session_state["el_entities"] = entities
        st.session_state["el_analyzed_text"] = input_text

# --- Results ---
if "el_entities" in st.session_state and st.session_state.get("el_analyzed_text"):
    entities = st.session_state["el_entities"]
    analyzed_text = st.session_state["el_analyzed_text"]

    st.divider()

    # Highlighted text
    st.subheader("Annotated Text")
    html = highlight_entities_html(analyzed_text, entities)
    st.markdown(html, unsafe_allow_html=True)

    # Entity cards
    st.divider()
    st.subheader(f"Entities Found ({len(entities)})")

    if not entities:
        st.info("No SBA lending entities detected in this text.")
    else:
        for i, entity in enumerate(entities):
            entity_type = entity.get("entity_type", "Unknown")
            color = ENTITY_COLORS.get(entity_type, "#888888")
            canonical = entity.get("canonical_name", entity.get("text", ""))
            definition = entity.get("definition", "No definition available.")
            text_span = entity.get("text", "")
            kg_node = entity.get("kg_node_id")

            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f'<span style="color:{color}; font-size:18px;">&#9679;</span> '
                        f'**{canonical}**'
                        f'&nbsp;&nbsp;<span style="color:#888; font-size:12px;">{entity_type}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"{definition}")
                    if text_span != canonical:
                        st.markdown(f'*Matched text:* "{text_span}"')

                with col2:
                    if kg_node:
                        st.markdown(
                            '<span style="color:#1ABC9C;">Linked to KG</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<span style="color:#888;">No KG match</span>',
                            unsafe_allow_html=True,
                        )

                if i < len(entities) - 1:
                    st.divider()
