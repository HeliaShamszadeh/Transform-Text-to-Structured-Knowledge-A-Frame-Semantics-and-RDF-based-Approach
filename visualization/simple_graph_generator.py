#!/usr/bin/env python3
"""
Simple Graph Generator
Creates clean PNG visualizations of author knowledge graphs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_author_graph(author_name, data_dir="evaluation_outputs", output_dir="graphs_2"):
    """Create a single author graph visualization."""
    
    # Set up paths
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    triples_file = data_path / author_name / "rdf" / f"{author_name}_triples.csv"
    
    if not triples_file.exists():
        print(f"❌ Missing triples file for {author_name}")
        return False
    
    try:
        triples_df = pd.read_csv(triples_file)
        
        print(f"📊 {author_name}: {len(triples_df)} triples")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes and edges from triples
        for _, row in triples_df.iterrows():
            subject = row['subject']
            predicate = row['predicate']
            object_val = row['object']
            
            # Clean node names for display
            subject_clean = subject.split('/')[-1].replace('_', ' ') if subject.startswith('http://') else subject
            object_clean = object_val.split('/')[-1].replace('_', ' ') if object_val.startswith('http://') else object_val
            
            G.add_node(subject, name=subject_clean, 
                      is_entity=subject.startswith('http://'))
            G.add_node(object_val, name=object_clean, 
                      is_entity=object_val.startswith('http://'))
            
            # Use predicate as edge label
            G.add_edge(subject, object_val, label=predicate)
        
        # Limit graph size for readability
        if G.number_of_nodes() > 30:
            # Get top nodes by degree
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:30]
            G = G.subgraph([node for node, _ in top_nodes])
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Use spring layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Separate nodes by type
        entity_nodes = [n for n, d in G.nodes(data=True) if d.get('is_entity', False)]
        object_nodes = [n for n, d in G.nodes(data=True) if not d.get('is_entity', False)]
        
        # Draw nodes
        if entity_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, 
                                 node_color='#FF6B6B', node_size=400, 
                                 alpha=0.8)
        
        if object_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=object_nodes, 
                                 node_color='#4ECDC4', node_size=300, 
                                 alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='#95A5A6', 
                             alpha=0.6, width=1.5)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        # Only show edge labels for important edges to avoid clutter
        important_edges = [(u, v) for u, v in G.edges() if G.degree(u) > 1 or G.degree(v) > 1]
        filtered_edge_labels = {edge: edge_labels[edge] for edge in important_edges if edge in edge_labels}
        nx.draw_networkx_edge_labels(G, pos, filtered_edge_labels, font_size=6, alpha=0.8)
        
        # Draw labels for all nodes
        labels = {node: G.nodes[node]['name'][:12] + "..." if len(G.nodes[node]['name']) > 12 
                 else G.nodes[node]['name'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold')
        
        # Add title
        plt.title(f"{author_name.replace('_', ' ')} Knowledge Graph\n"
                 f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}", 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        import matplotlib.patches as mpatches
        entity_patch = mpatches.Patch(color='#FF6B6B', label='Entities')
        object_patch = mpatches.Patch(color='#4ECDC4', label='Objects')
        plt.legend(handles=[entity_patch, object_patch], loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        output_file = output_path / f"{author_name}_graph.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Saved: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {author_name}: {e}")
        return False

def main():
    """Generate graphs for all authors."""
    print("🚀 Simple Graph Generator")
    print("=" * 40)
    
    # Get all authors
    data_dir = Path("evaluation_outputs")
    authors = []
    
    for author_dir in data_dir.iterdir():
        if author_dir.is_dir():
            author_name = author_dir.name
            triples_file = author_dir / "rdf" / f"{author_name}_triples.csv"
            
            if triples_file.exists():
                authors.append(author_name)
    
    print(f"Found {len(authors)} authors with complete data")
    
    # Generate graphs
    success_count = 0
    for i, author in enumerate(authors, 1):
        print(f"\n[{i}/{len(authors)}] Processing {author}...")
        if create_author_graph(author):
            success_count += 1
    
    print(f"\n🎉 Generated {success_count} graph visualizations!")
    print("📁 Check the 'graphs_2' folder for PNG files")

if __name__ == "__main__":
    main()
