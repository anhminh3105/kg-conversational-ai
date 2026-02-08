#!/usr/bin/env python3
"""
Visualize the Neo4j Knowledge Graph (Triplet-based schema).

This script creates PNG visualizations of the knowledge graph stored in Neo4j.
The graph uses Triplet nodes with (subject, predicate, object) properties.

Features:
- Full graph visualization
- Entity-focused subgraph visualization
- Node coloring by connectivity (hubs vs leaves)
- Edge labels showing relationship types
- Graph statistics overlay
- Support for large graphs via sampling

Usage:
    # Visualize full graph
    python scripts/visualize_kg.py --output outputs/kg_full.png
    
    # Visualize subgraph around an entity
    python scripts/visualize_kg.py --entity "Einstein" --depth 2 --output outputs/kg_einstein.png
    
    # Limit nodes for large graphs
    python scripts/visualize_kg.py --max-nodes 100 --output outputs/kg_sample.png
    
    # Custom Neo4j connection
    python scripts/visualize_kg.py --neo4j-password mypassword --output outputs/kg.png

Prerequisites:
    - Neo4j running at bolt://localhost:7687
    - Knowledge graph data loaded (Triplet nodes)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from neo4j import GraphDatabase

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Simple Neo4j connection manager."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def query(self, query: str, parameters: dict = None) -> List[dict]:
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]


def fetch_triplets(
    conn: Neo4jConnection,
    max_triplets: Optional[int] = None,
) -> List[Dict]:
    """
    Fetch all triplets from Neo4j.
    
    Args:
        conn: Neo4j connection
        max_triplets: Optional limit on number of triplets
        
    Returns:
        List of triplet dictionaries with subject, predicate, object
    """
    logger.info("Fetching triplets from Neo4j...")
    
    query = """
        MATCH (t:Triplet)
        RETURN t.subject AS subject, 
               t.predicate AS predicate, 
               t.object AS object,
               t.source AS source
    """
    
    if max_triplets:
        query += f" LIMIT {max_triplets}"
    
    triplets = conn.query(query)
    logger.info(f"Retrieved {len(triplets)} triplets")
    
    return triplets


def fetch_entity_subgraph(
    conn: Neo4jConnection,
    entity: str,
    depth: int = 2,
    max_triplets: int = 100,
) -> List[Dict]:
    """
    Fetch triplets related to a specific entity within N hops.
    
    Args:
        conn: Neo4j connection
        entity: Entity name to center the subgraph on
        depth: Number of hops from the entity
        max_triplets: Maximum triplets to return
        
    Returns:
        List of triplet dictionaries
    """
    logger.info(f"Fetching subgraph for entity '{entity}' with depth {depth}...")
    
    # Start with direct connections
    visited_entities: Set[str] = {entity.lower()}
    current_entities = [entity]
    all_triplets = []
    
    for hop in range(depth):
        if not current_entities:
            break
            
        # Fetch triplets involving current entities
        query = """
            MATCH (t:Triplet)
            WHERE toLower(t.subject) IN $entities 
                  OR toLower(t.object) IN $entities
            RETURN DISTINCT t.subject AS subject, 
                   t.predicate AS predicate, 
                   t.object AS object,
                   t.source AS source
            LIMIT $limit
        """
        
        triplets = conn.query(query, {
            "entities": [e.lower() for e in current_entities],
            "limit": max_triplets - len(all_triplets),
        })
        
        # Collect new entities for next hop
        next_entities = []
        for t in triplets:
            all_triplets.append(t)
            
            for entity_field in ["subject", "object"]:
                e = t[entity_field]
                if e.lower() not in visited_entities:
                    visited_entities.add(e.lower())
                    next_entities.append(e)
        
        current_entities = next_entities
        
        if len(all_triplets) >= max_triplets:
            break
    
    logger.info(f"Retrieved {len(all_triplets)} triplets for subgraph")
    return all_triplets


def build_networkx_graph(triplets: List[Dict]) -> nx.DiGraph:
    """
    Convert triplets to a NetworkX directed graph.
    
    Subjects and objects become nodes, predicates become edge labels.
    
    Args:
        triplets: List of triplet dictionaries
        
    Returns:
        NetworkX DiGraph
    """
    logger.info("Building NetworkX graph...")
    
    G = nx.DiGraph()
    
    for t in triplets:
        subject = t["subject"]
        predicate = t["predicate"]
        obj = t["object"]
        source = t.get("source", "original")
        
        # Add nodes
        G.add_node(subject)
        G.add_node(obj)
        
        # Add edge with predicate as label
        # If edge already exists, append predicate to list
        if G.has_edge(subject, obj):
            existing = G[subject][obj].get("predicates", [])
            if predicate not in existing:
                existing.append(predicate)
            G[subject][obj]["predicates"] = existing
        else:
            G.add_edge(subject, obj, predicates=[predicate], source=source)
    
    logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G


def get_node_colors(G: nx.DiGraph) -> Tuple[List[str], dict]:
    """
    Assign colors to nodes based on their connectivity.
    
    High-degree nodes (hubs) get warm colors, low-degree (leaves) get cool colors.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Tuple of (color list, color mapping info)
    """
    # Calculate degree for each node (in + out for DiGraph)
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 0
    
    # Normalize degrees to [0, 1]
    colors = []
    for node in G.nodes():
        degree = degrees[node]
        if max_degree > min_degree:
            normalized = (degree - min_degree) / (max_degree - min_degree)
        else:
            normalized = 0.5
        
        # Use a colormap: cool colors (blue) for low degree, warm (red/orange) for high
        # plt.cm.coolwarm goes from blue (0) to red (1)
        colors.append(plt.cm.coolwarm(normalized))
    
    color_info = {
        "min_degree": min_degree,
        "max_degree": max_degree,
    }
    
    return colors, color_info


def get_edge_labels(G: nx.DiGraph, max_label_length: int = 20) -> Dict:
    """
    Create edge labels from predicates.
    
    Args:
        G: NetworkX graph
        max_label_length: Maximum length for edge labels
        
    Returns:
        Dictionary mapping edges to labels
    """
    edge_labels = {}
    
    for u, v, data in G.edges(data=True):
        predicates = data.get("predicates", [])
        if predicates:
            # Join multiple predicates with comma
            label = ", ".join(predicates)
            # Truncate if too long
            if len(label) > max_label_length:
                label = label[:max_label_length-3] + "..."
            edge_labels[(u, v)] = label
    
    return edge_labels


def calculate_statistics(G: nx.DiGraph, triplets: List[Dict]) -> Dict:
    """
    Calculate graph statistics for display.
    
    Args:
        G: NetworkX graph
        triplets: Original triplet list
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_triplets": len(triplets),
    }
    
    # Count predicates
    predicate_counts = Counter()
    for t in triplets:
        predicate_counts[t["predicate"]] += 1
    
    stats["top_predicates"] = predicate_counts.most_common(5)
    
    # Find most connected nodes
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    stats["top_nodes"] = top_nodes
    
    # Count sources
    source_counts = Counter()
    for t in triplets:
        source_counts[t.get("source", "original")] += 1
    stats["sources"] = dict(source_counts)
    
    return stats


def visualize_graph(
    G: nx.DiGraph,
    triplets: List[Dict],
    output_path: str,
    title: str = "Knowledge Graph Visualization",
    figsize: Tuple[int, int] = (20, 16),
    show_edge_labels: bool = True,
    node_size_base: int = 300,
    font_size: int = 8,
) -> str:
    """
    Visualize the graph and save to PNG.
    
    Args:
        G: NetworkX graph
        triplets: Original triplet list (for statistics)
        output_path: Path to save the PNG
        title: Title for the visualization
        figsize: Figure size (width, height)
        show_edge_labels: Whether to show predicate labels on edges
        node_size_base: Base size for nodes
        font_size: Font size for labels
        
    Returns:
        Path to saved image
    """
    logger.info("Generating visualization...")
    
    if G.number_of_nodes() == 0:
        logger.warning("Graph is empty, nothing to visualize")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate layout
    # For larger graphs, use spring layout with more iterations
    if G.number_of_nodes() > 50:
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    else:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Get node colors based on degree
    node_colors, color_info = get_node_colors(G)
    
    # Calculate node sizes based on degree
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = [
        node_size_base + (degrees[node] / max_degree) * node_size_base * 3
        for node in G.nodes()
    ]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax,
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#888888',
        alpha=0.5,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )
    
    # Draw node labels
    # For large graphs, only label high-degree nodes
    if G.number_of_nodes() > 30:
        # Only label nodes with degree > median
        median_degree = sorted(degrees.values())[len(degrees) // 2]
        labels = {node: node for node, deg in degrees.items() if deg >= median_degree}
    else:
        labels = {node: node for node in G.nodes()}
    
    # Truncate long labels
    labels = {k: (v[:15] + "..." if len(v) > 15 else v) for k, v in labels.items()}
    
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=font_size,
        font_weight='bold',
        ax=ax,
    )
    
    # Draw edge labels if requested and graph is small enough
    if show_edge_labels and G.number_of_edges() <= 50:
        edge_labels = get_edge_labels(G, max_label_length=15)
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=font_size - 2,
            font_color='#555555',
            ax=ax,
        )
    
    # Calculate statistics
    stats = calculate_statistics(G, triplets)
    
    # Add title
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    
    # Add statistics text box
    stats_lines = [
        f"Nodes: {stats['num_nodes']}  |  Edges: {stats['num_edges']}  |  Triplets: {stats['num_triplets']}",
        "",
        "Top Entities (by connections):",
    ]
    for node, degree in stats['top_nodes'][:3]:
        node_display = node[:20] + "..." if len(node) > 20 else node
        stats_lines.append(f"  • {node_display}: {degree}")
    
    stats_lines.append("")
    stats_lines.append("Top Predicates:")
    for pred, count in stats['top_predicates'][:3]:
        pred_display = pred[:20] + "..." if len(pred) > 20 else pred
        stats_lines.append(f"  • {pred_display}: {count}")
    
    stats_text = "\n".join(stats_lines)
    
    # Add text box
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=props,
    )
    
    # Add color legend
    legend_patches = [
        mpatches.Patch(color=plt.cm.coolwarm(0.0), label=f'Low connectivity (degree={color_info["min_degree"]})'),
        mpatches.Patch(color=plt.cm.coolwarm(0.5), label='Medium connectivity'),
        mpatches.Patch(color=plt.cm.coolwarm(1.0), label=f'High connectivity (degree={color_info["max_degree"]})'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
    
    # Remove axis
    ax.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Visualization saved to: {output_path}")
    
    return output_path


def print_graph_summary(G: nx.DiGraph, triplets: List[Dict]):
    """Print a summary of the graph to console."""
    stats = calculate_statistics(G, triplets)
    
    print("\n" + "=" * 60)
    print("Knowledge Graph Summary")
    print("=" * 60)
    
    print(f"\nGraph Size:")
    print(f"  Nodes:    {stats['num_nodes']}")
    print(f"  Edges:    {stats['num_edges']}")
    print(f"  Triplets: {stats['num_triplets']}")
    
    print(f"\nData Sources:")
    for source, count in stats['sources'].items():
        print(f"  {source}: {count}")
    
    print(f"\nTop Connected Entities:")
    for node, degree in stats['top_nodes']:
        print(f"  {node}: {degree} connections")
    
    print(f"\nTop Predicates (relationship types):")
    for pred, count in stats['top_predicates']:
        print(f"  {pred}: {count} occurrences")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Neo4j Knowledge Graph as PNG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--output", "-o",
        default="outputs/knowledge_graph.png",
        help="Output path for PNG file (default: outputs/knowledge_graph.png)"
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j Bolt URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--neo4j-password",
        default=None,
        help="Neo4j password (default: from NEO4J_PASSWORD env or 'password123')"
    )
    parser.add_argument(
        "--entity", "-e",
        default=None,
        help="Center visualization around a specific entity (creates subgraph)"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=2,
        help="Depth of subgraph when using --entity (default: 2)"
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Maximum number of triplets to visualize (for large graphs)"
    )
    parser.add_argument(
        "--no-edge-labels",
        action="store_true",
        help="Don't show predicate labels on edges"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[20, 16],
        metavar=('WIDTH', 'HEIGHT'),
        help="Figure size in inches (default: 20 16)"
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom title for the visualization"
    )
    
    args = parser.parse_args()
    
    # Get password
    neo4j_password = args.neo4j_password
    if neo4j_password is None:
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "password123")
    
    print("=" * 60)
    print("Neo4j Knowledge Graph Visualization")
    print("=" * 60)
    
    # Connect to Neo4j
    print(f"\nConnecting to Neo4j at {args.neo4j_uri}...")
    try:
        conn = Neo4jConnection(args.neo4j_uri, args.neo4j_user, neo4j_password)
    except Exception as e:
        print(f"ERROR: Failed to connect to Neo4j: {e}")
        print("Make sure Neo4j is running and accessible.")
        sys.exit(1)
    
    try:
        # Fetch triplets
        if args.entity:
            triplets = fetch_entity_subgraph(
                conn,
                args.entity,
                depth=args.depth,
                max_triplets=args.max_nodes or 100,
            )
            default_title = f"Knowledge Graph: {args.entity} (depth={args.depth})"
        else:
            triplets = fetch_triplets(conn, max_triplets=args.max_nodes)
            default_title = "Knowledge Graph Visualization"
        
        if not triplets:
            print("\nNo triplets found in the database.")
            print("Make sure you have loaded data into Neo4j.")
            sys.exit(1)
        
        # Build NetworkX graph
        G = build_networkx_graph(triplets)
        
        # Print summary
        print_graph_summary(G, triplets)
        
        # Generate visualization
        title = args.title or default_title
        output_path = visualize_graph(
            G,
            triplets,
            args.output,
            title=title,
            figsize=tuple(args.figsize),
            show_edge_labels=not args.no_edge_labels,
        )
        
        if output_path:
            print(f"\nVisualization saved to: {output_path}")
            print("\nDone!")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
