"""
Visualize the imported Knowledge Graph
This script demonstrates how to visualize the knowledge graph using NetworkX and Matplotlib
"""

from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import os

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

def fetch_graph_data(conn):
    """Fetch all nodes and relationships from Neo4j"""
    print("\nFetching graph data from Neo4j...")
    
    query = """
    MATCH (p:Person)-[r:ACTED_IN|DIRECTED]->(m:Movie)
    RETURN p.name AS person, 
           type(r) AS relationship_type,
           m.title AS movie,
           m.genre AS genre
    """
    
    results = conn.query(query)
    print(f"  ✓ Retrieved {len(results)} relationships")
    
    return results

def create_networkx_graph(graph_data):
    """Convert Neo4j data to NetworkX graph"""
    print("\nCreating NetworkX graph...")
    
    G = nx.Graph()
    
    for record in graph_data:
        person = record['person']
        movie = record['movie']
        rel_type = record['relationship_type']
        genre = record['genre'] if record['genre'] else 'Unknown'
        
        # Add nodes with attributes
        G.add_node(person, node_type='person')
        G.add_node(movie, node_type='movie', genre=genre)
        
        # Add edge with relationship type
        G.add_edge(person, movie, relationship=rel_type)
    
    print(f"  ✓ Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

def visualize_graph(G, output_path):
    """Visualize the graph using matplotlib"""
    print("\nGenerating visualization...")
    
    # Create figure with larger size
    plt.figure(figsize=(16, 12))
    
    # Separate nodes by type
    person_nodes = [node for node, attr in G.nodes(data=True) if attr.get('node_type') == 'person']
    movie_nodes = [node for node, attr in G.nodes(data=True) if attr.get('node_type') == 'movie']
    
    # Create layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Draw person nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=person_nodes, 
                          node_color='#3498db',  # Blue
                          node_size=1200,
                          label='People',
                          alpha=0.9)
    
    # Draw movie nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=movie_nodes, 
                          node_color='#e74c3c',  # Red
                          node_size=1200,
                          label='Movies',
                          alpha=0.9)
    
    # Draw edges
    # Separate ACTED_IN and DIRECTED relationships
    acted_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'ACTED_IN']
    directed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'DIRECTED']
    
    nx.draw_networkx_edges(G, pos, 
                          edgelist=acted_edges,
                          edge_color='#95a5a6',  # Gray
                          alpha=0.5,
                          width=1.5)
    
    nx.draw_networkx_edges(G, pos, 
                          edgelist=directed_edges,
                          edge_color='#f39c12',  # Orange
                          alpha=0.7,
                          width=2.5,
                          style='dashed')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Add legend
    plt.legend(loc='upper left', fontsize=12)
    
    # Add title
    plt.title("Movie Knowledge Graph Visualization", fontsize=20, fontweight='bold', pad=20)
    
    # Add subtitle with graph stats
    stats_text = f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | " \
                 f"People: {len(person_nodes)} | Movies: {len(movie_nodes)}"
    plt.text(0.5, 0.95, stats_text, 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Graph visualization saved to: {output_path}")
    
    return output_path

def create_genre_subgraph(G, genre, output_path):
    """Create a subgraph for a specific genre"""
    print(f"\nCreating {genre} subgraph...")
    
    # Filter nodes
    genre_movies = [node for node, attr in G.nodes(data=True) 
                   if attr.get('node_type') == 'movie' and attr.get('genre') == genre]
    
    if not genre_movies:
        print(f"  No {genre} movies found")
        return None
    
    # Get all people connected to these movies
    subgraph_nodes = set(genre_movies)
    for movie in genre_movies:
        subgraph_nodes.update(G.neighbors(movie))
    
    # Create subgraph
    subG = G.subgraph(subgraph_nodes)
    
    # Visualize
    plt.figure(figsize=(12, 9))
    
    person_nodes = [node for node, attr in subG.nodes(data=True) if attr.get('node_type') == 'person']
    movie_nodes = [node for node, attr in subG.nodes(data=True) if attr.get('node_type') == 'movie']
    
    pos = nx.spring_layout(subG, k=2, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(subG, pos, nodelist=person_nodes, 
                          node_color='#3498db', node_size=1500, alpha=0.9, label='People')
    nx.draw_networkx_nodes(subG, pos, nodelist=movie_nodes, 
                          node_color='#e74c3c', node_size=1500, alpha=0.9, label='Movies')
    nx.draw_networkx_edges(subG, pos, edge_color='#95a5a6', alpha=0.6, width=2)
    nx.draw_networkx_labels(subG, pos, font_size=10, font_weight='bold')
    
    plt.legend(loc='upper left', fontsize=12)
    plt.title(f"{genre} Movies Knowledge Graph", fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {genre} subgraph saved to: {output_path}")
    
    return output_path

def print_graph_statistics(G):
    """Print detailed graph statistics"""
    print("\nGraph Statistics:")
    print("-" * 60)
    
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    
    person_nodes = [node for node, attr in G.nodes(data=True) if attr.get('node_type') == 'person']
    movie_nodes = [node for node, attr in G.nodes(data=True) if attr.get('node_type') == 'movie']
    
    print(f"  Person nodes: {len(person_nodes)}")
    print(f"  Movie nodes: {len(movie_nodes)}")
    
    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\n  Most connected nodes:")
    for node, centrality in top_nodes:
        node_type = G.nodes[node].get('node_type', 'unknown')
        print(f"    • {node} ({node_type}): {centrality:.3f}")
    
    # Check if graph is connected
    if nx.is_connected(G):
        print(f"\n  Graph is connected")
        print(f"  Average shortest path length: {nx.average_shortest_path_length(G):.2f}")
    else:
        print(f"\n  Graph has {nx.number_connected_components(G)} connected components")

def main():
    print("=" * 60)
    print("STEP 6: Visualizing the Knowledge Graph")
    print("=" * 60)
    
    # Connection parameters
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = "password123"
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to Neo4j
    conn = Neo4jConnection(URI, USER, PASSWORD)
    
    try:
        # Fetch graph data
        graph_data = fetch_graph_data(conn)
        
        # Create NetworkX graph
        G = create_networkx_graph(graph_data)
        
        # Print statistics
        print_graph_statistics(G)
        
        # Visualize full graph
        main_output = os.path.join(output_dir, 'knowledge_graph.png')
        visualize_graph(G, main_output)
        
        # Create genre-specific subgraphs
        for genre in ['Drama', 'Sci-Fi', 'Action']:
            genre_output = os.path.join(output_dir, f'knowledge_graph_{genre.lower()}.png')
            create_genre_subgraph(G, genre, genre_output)
        
        print("\n" + "=" * 60)
        print("Step 6 completed successfully!")
        print(f"Visualizations saved in: {output_dir}")
        print("=" * 60)
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()
