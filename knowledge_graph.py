"""Knowledge graph loading, querying, and visualization utilities.

Loads the pre-built SBA knowledge graph from JSON into a NetworkX graph
and provides search, traversal, and subgraph extraction functions.
"""

import json

import networkx as nx

from config import KG_FILE, ENTITY_COLORS


def load_graph() -> nx.DiGraph:
    """Load the pre-built knowledge graph from JSON into a NetworkX DiGraph."""
    if not KG_FILE.exists():
        return nx.DiGraph()

    with open(KG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()

    for node in data.get("nodes", []):
        G.add_node(
            node["id"],
            label=node.get("label", node["id"]),
            type=node.get("type", "Unknown"),
            definition=node.get("definition", ""),
        )

    for edge in data.get("edges", []):
        if edge["source"] in G and edge["target"] in G:
            G.add_edge(
                edge["source"],
                edge["target"],
                relation=edge.get("relation", "RELATED_TO"),
            )

    return G


def search_graph(G: nx.DiGraph, query: str, entity_type: str | None = None) -> list[str]:
    """Find nodes matching a search query by label, with optional type filter."""
    query_lower = query.lower()
    results = []

    for node_id, attrs in G.nodes(data=True):
        if entity_type and attrs.get("type") != entity_type:
            continue
        label = attrs.get("label", "").lower()
        definition = attrs.get("definition", "").lower()
        if query_lower in label or query_lower in definition:
            results.append(node_id)

    return results


def get_neighbors(G: nx.DiGraph, node_id: str, depth: int = 1) -> set[str]:
    """Get all nodes within `depth` hops of the given node (both directions)."""
    if node_id not in G:
        return set()

    visited = {node_id}
    frontier = {node_id}

    for _ in range(depth):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.successors(n))
            next_frontier.update(G.predecessors(n))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier

    return visited


def get_subgraph(G: nx.DiGraph, node_ids: set[str]) -> nx.DiGraph:
    """Extract a subgraph containing only the specified nodes and edges between them."""
    return G.subgraph(node_ids).copy()


def get_node_details(G: nx.DiGraph, node_id: str) -> dict | None:
    """Get full details for a node including its connections."""
    if node_id not in G:
        return None

    attrs = G.nodes[node_id]
    outgoing = [
        {"target": t, "target_label": G.nodes[t].get("label", t), "relation": d.get("relation", "")}
        for _, t, d in G.out_edges(node_id, data=True)
    ]
    incoming = [
        {"source": s, "source_label": G.nodes[s].get("label", s), "relation": d.get("relation", "")}
        for s, _, d in G.in_edges(node_id, data=True)
    ]

    return {
        "id": node_id,
        "label": attrs.get("label", node_id),
        "type": attrs.get("type", "Unknown"),
        "definition": attrs.get("definition", ""),
        "outgoing": outgoing,
        "incoming": incoming,
        "degree": G.degree(node_id),
    }


def build_pyvis_html(G: nx.DiGraph, height: str = "600px", selected_node: str | None = None) -> str:
    """Build an interactive pyvis HTML visualization of the graph."""
    from pyvis.network import Network

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#0E1117",
        font_color="white",
        select_menu=False,
        filter_menu=False,
    )

    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.09,
    )

    for node_id, attrs in G.nodes(data=True):
        entity_type = attrs.get("type", "Unknown")
        color = ENTITY_COLORS.get(entity_type, "#888888")
        size = 15 + min(G.degree(node_id) * 3, 30)
        label = attrs.get("label", node_id)
        title = f"{label}\nType: {entity_type}\n{attrs.get('definition', '')}"

        border_color = "#FFD700" if node_id == selected_node else color
        border_width = 4 if node_id == selected_node else 1

        net.add_node(
            node_id,
            label=label,
            color={"background": color, "border": border_color},
            size=size,
            title=title,
            borderWidth=border_width,
            font={"size": 12, "color": "white"},
        )

    for source, target, attrs in G.edges(data=True):
        relation = attrs.get("relation", "")
        net.add_edge(
            source,
            target,
            title=relation,
            label=relation,
            color="#555555",
            font={"size": 8, "color": "#999999", "align": "middle"},
            arrows="to",
        )

    net.set_options("""
    {
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
        }
    }
    """)

    return net.generate_html()


def get_graph_stats(G: nx.DiGraph) -> dict:
    """Get summary statistics about the knowledge graph."""
    if len(G) == 0:
        return {"nodes": 0, "edges": 0, "types": {}}

    type_counts = {}
    for _, attrs in G.nodes(data=True):
        t = attrs.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "nodes": len(G.nodes),
        "edges": len(G.edges),
        "types": type_counts,
    }
