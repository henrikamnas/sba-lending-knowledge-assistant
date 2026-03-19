"""Knowledge Graph Explorer — interactive visualization of SBA lending entities and relations."""

import streamlit as st
import streamlit.components.v1 as components

from config import ENTITY_TYPES, ENTITY_COLORS, KG_FILE
from knowledge_graph import (
    load_graph,
    search_graph,
    get_neighbors,
    get_subgraph,
    get_node_details,
    build_pyvis_html,
    get_graph_stats,
)

# --- Check KG exists ---
if not KG_FILE.exists():
    st.error(
        "Knowledge graph not built yet. Run `python build_knowledge_graph.py` to generate it."
    )
    st.stop()

# --- Load graph ---
@st.cache_resource
def _load_kg():
    return load_graph()

G = _load_kg()
stats = get_graph_stats(G)

if stats["nodes"] == 0:
    st.error("Knowledge graph is empty. Rebuild with `python build_knowledge_graph.py --force`.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.title("Graph Explorer")
    st.markdown(
        "Explore entities and relationships extracted from SBA 7(a) lending documents "
        "using an LLM-powered knowledge graph."
    )

    st.divider()
    st.subheader("Graph Stats")
    st.metric("Nodes", stats["nodes"])
    st.metric("Edges", stats["edges"])

    with st.expander("Nodes by Type"):
        for t, count in sorted(stats["types"].items()):
            color = ENTITY_COLORS.get(t, "#888")
            st.markdown(
                f'<span style="color:{color}">&#9679;</span> **{t}**: {count}',
                unsafe_allow_html=True,
            )

    st.divider()

    # Filters
    st.subheader("Filters")
    selected_types = st.multiselect(
        "Entity Types",
        options=ENTITY_TYPES,
        default=ENTITY_TYPES,
        key="kg_type_filter",
    )

    search_query = st.text_input("Search nodes", key="kg_search", placeholder="e.g., 7(a), collateral...")

    st.divider()
    st.caption("Built by Henrik Axelsson as an AI/KG demo.")

# --- Main Content ---
st.title("Knowledge Graph Explorer")
st.markdown(
    f"Interactive visualization of **{stats['nodes']} entities** and "
    f"**{stats['edges']} relationships** extracted from SBA lending documents."
)

# --- Legend ---
legend_cols = st.columns(len(ENTITY_COLORS))
for col, (etype, color) in zip(legend_cols, ENTITY_COLORS.items()):
    col.markdown(
        f'<span style="color:{color}; font-size:20px;">&#9679;</span> {etype}',
        unsafe_allow_html=True,
    )

# --- Node inspector (above graph so selection drives the visualization) ---
st.divider()
st.subheader("Inspect Node")

node_options = sorted(
    [(G.nodes[n].get("label", n), n) for n in G.nodes],
    key=lambda x: x[0],
)
node_labels = [label for label, _ in node_options]
node_ids = [nid for _, nid in node_options]

selected_label = st.selectbox(
    "Select a node — the graph will focus on it and its connections",
    options=node_labels,
    index=None,
    placeholder="Choose a node...",
    key="kg_node_select",
)

# --- Determine which graph to display ---
selected_id = None
if selected_label:
    selected_id = node_ids[node_labels.index(selected_label)]

if selected_id:
    # Show only the selected node and its 1-hop neighborhood
    neighborhood = get_neighbors(G, selected_id, depth=1)
    display_graph = get_subgraph(G, neighborhood)
elif selected_types != ENTITY_TYPES or search_query:
    if search_query:
        matching_ids = set()
        for t in selected_types:
            matching_ids.update(search_graph(G, search_query, entity_type=t))
        expanded = set()
        for nid in matching_ids:
            expanded.update(get_neighbors(G, nid, depth=1))
        display_nodes = expanded
    else:
        display_nodes = {
            nid for nid, attrs in G.nodes(data=True)
            if attrs.get("type") in selected_types
        }
    display_graph = get_subgraph(G, display_nodes)
else:
    display_graph = G

# --- Graph visualization ---
if len(display_graph) > 0:
    html = build_pyvis_html(display_graph, height="550px", selected_node=selected_id)
    components.html(html, height=570, scrolling=False)
else:
    st.info("No nodes match the current filters.")

# --- Node details panel ---
if selected_id:
    details = get_node_details(G, selected_id)
    if details:
        st.divider()
        col1, col2 = st.columns([2, 1])
        with col1:
            color = ENTITY_COLORS.get(details["type"], "#888")
            st.markdown(
                f'### <span style="color:{color}">&#9679;</span> {details["label"]}',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Type:** {details['type']}")
            st.markdown(f"**Definition:** {details['definition']}")
            st.markdown(f"**Connections:** {details['degree']}")

        with col2:
            if details["outgoing"]:
                st.markdown("**Outgoing Relations:**")
                for rel in details["outgoing"]:
                    st.markdown(f"- *{rel['relation']}* -> {rel['target_label']}")

            if details["incoming"]:
                st.markdown("**Incoming Relations:**")
                for rel in details["incoming"]:
                    st.markdown(f"- {rel['source_label']} *{rel['relation']}* ->")
