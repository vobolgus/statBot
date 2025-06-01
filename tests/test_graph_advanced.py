#!/usr/bin/env python3
"""
Advanced test script for graph visualization with different layout algorithms.
Usage: python test_graph_advanced.py [min_interactions] [layout_algorithm]

Layout algorithms:
- fruchterman (default)
- spring
- kamada
- circular
- shell
"""

import sys
import logging
import networkx as nx
import plotly.graph_objects as go
from analytics import load_and_prepare_data
from social_graph import build_interaction_graph, calculate_graph_metrics
from config import DB_FILE
import math

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_graph_with_layout(G, title="User Interaction Network", layout_type="fruchterman"):
    """Create graph visualization with specified layout algorithm."""
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes to visualize")
    
    # Choose layout algorithm
    logger.info(f"Using {layout_type} layout algorithm")
    
    if layout_type == "fruchterman":
        pos = nx.fruchterman_reingold_layout(G, k=0.8, iterations=150, weight='weight', seed=42, scale=2)
    elif layout_type == "spring":
        pos = nx.spring_layout(G, k=1.5, iterations=100, weight='weight', seed=42, scale=2)
    elif layout_type == "kamada":
        pos = nx.kamada_kawai_layout(G, weight='weight', scale=2)
    elif layout_type == "circular":
        pos = nx.circular_layout(G, scale=2)
    elif layout_type == "shell":
        # Create shells based on node degree
        shells = []
        nodes_by_degree = {}
        for node in G.nodes():
            degree = G.degree(node, weight='weight')
            if degree not in nodes_by_degree:
                nodes_by_degree[degree] = []
            nodes_by_degree[degree].append(node)
        
        # Sort by degree and create shells
        for degree in sorted(nodes_by_degree.keys(), reverse=True):
            shells.append(nodes_by_degree[degree])
        
        pos = nx.shell_layout(G, shells, scale=2)
    else:
        pos = nx.fruchterman_reingold_layout(G, k=0.8, iterations=150, weight='weight', seed=42, scale=2)
    
    # Create edge traces
    edge_traces = []
    annotations = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=math.log(2*weight+1) * 0.2,  # Reduced: was 0.5, max was 10
                color='rgba(125,125,125,0.4)'  # Slightly more transparent
            ),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
        
        # Arrow annotations
        annotations.append(
            dict(
                ax=x0, ay=y0, axref='x', ayref='y',
                x=x1, y=y1, xref='x', yref='y',
                showarrow=True, arrowhead=2, arrowsize=0.7,  # Smaller arrows
                arrowwidth=min(weight * 0.15, 2),  # Thinner arrows
                arrowcolor='rgba(125,125,125,0.4)',
                opacity=0.4  # More transparent
            )
        )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Calculate metrics
        in_degree = G.in_degree(node, weight='weight')
        out_degree = G.out_degree(node, weight='weight')
        total_degree = in_degree + out_degree
        
        # Node size (logarithmic scale)
        # Reduced sizes: base 15 (was 30), multiplier 8 (was 15), max 40 (was 70)
        node_size = math.log(total_degree + 1) * 8
        node_sizes.append(node_size)
        node_colors.append(total_degree)
        
        # Hover text
        hover_text = f"<b>{node}</b><br>"
        hover_text += f"Получено ответов: {in_degree}<br>"
        hover_text += f"Отправлено ответов: {out_degree}<br>"
        hover_text += f"Всего взаимодействий: {total_degree}"
        node_text.append(hover_text)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node for node in G.nodes()],
        hovertext=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_sizes,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                title=dict(text='Взаимодействия', side='right'),
                xanchor='left'
            ),
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Add legend
    legend_annotations = annotations + [
        dict(
            text=f"<b>Алгоритм: {layout_type}</b><br>" +
                 "<b>Размер узла:</b> Общее количество взаимодействий (лог. шкала)<br>" +
                 "<b>Цвет узла:</b> Интенсивность взаимодействий<br>" +
                 "<b>Стрелки:</b> Направление ответов",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1, borderpad=4,
            xanchor="left", yanchor="top"
        )
    ]
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=legend_annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        width=1200,
        height=800
    )
    
    # Save image
    filename = f"interaction_graph_{layout_type}.png"
    fig.write_image(filename, scale=2)
    
    return filename


def test_graph_advanced(min_interactions=3, layout_type="fruchterman"):
    """Test graph generation with different layouts."""
    logger.info(f"Loading data from {DB_FILE}")
    
    # Load data
    df = load_and_prepare_data(DB_FILE)
    
    if df.empty:
        logger.error("No messages in the database.")
        return
    
    logger.info(f"Loaded {len(df)} messages")
    
    # Build interaction graph
    logger.info(f"Building interaction graph with min_interactions={min_interactions}")
    G = build_interaction_graph(df, min_interactions)
    
    if G.number_of_nodes() == 0:
        logger.error(f"No interactions found with minimum threshold of {min_interactions} replies.")
        return
    
    logger.info(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate metrics (including h-index)
    metrics = calculate_graph_metrics(G, df)
    
    # Print metrics
    print(f"\nГраф взаимодействий:")
    print(f"- Узлов: {G.number_of_nodes()}")
    print(f"- Связей: {G.number_of_edges()}")
    print(f"- Плотность: {metrics.get('density', 0):.3f}")
    print(f"- Сообществ: {metrics.get('communities', 0)}")
    
    # Create visualization
    logger.info(f"Creating graph visualization with {layout_type} layout...")
    try:
        graph_file = create_graph_with_layout(
            G, 
            f"Граф взаимодействий - {layout_type} layout (мин. {min_interactions} ответов)",
            layout_type
        )
        logger.info(f"Graph saved to: {graph_file}")
        print(f"\n✅ График сохранен в файл: {graph_file}")
        
    except Exception as e:
        logger.error(f"Error creating graph: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Parse command line arguments
    min_interactions = 3
    layout_type = "fruchterman"
    
    if len(sys.argv) > 1:
        try:
            min_interactions = int(sys.argv[1])
        except ValueError:
            print(f"Invalid min_interactions value: {sys.argv[1]}")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        layout_type = sys.argv[2].lower()
        if layout_type not in ["fruchterman", "spring", "kamada", "circular", "shell"]:
            print(f"Unknown layout type: {layout_type}")
            print("Available layouts: fruchterman, spring, kamada, circular, shell")
            sys.exit(1)
    
    print(f"Testing with min_interactions={min_interactions}, layout={layout_type}")
    test_graph_advanced(min_interactions, layout_type)