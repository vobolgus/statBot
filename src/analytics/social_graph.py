"""
Social interaction graph analysis and visualization.
"""

import logging
import math
import fix_pillow  # Fix for Pillow 10+ compatibility
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def build_interaction_graph(df: pd.DataFrame, min_interactions: int = 30) -> nx.DiGraph:
    """
    Build a directed graph of user interactions based on reply patterns.
    
    Args:
        df: DataFrame with messages containing reply_to_message_id
        min_interactions: Minimum number of interactions to include edge
        
    Returns:
        NetworkX directed graph with weighted edges
    """
    G = nx.DiGraph()
    
    # Count interactions between users
    interaction_counts = {}
    
    # Check if reply_to_message_id column exists
    if 'reply_to_message_id' not in df.columns:
        logger.warning("Column 'reply_to_message_id' not found in DataFrame")
        return G
    
    # Get messages with replies
    replies = df[df['reply_to_message_id'].notna()].copy()
    logger.info(f"Found {len(replies)} messages with replies out of {len(df)} total messages")
    
    for _, reply in replies.iterrows():
        # Find the original message
        reply_to_id = reply['reply_to_message_id']
        original_msgs = df[df['id'] == reply_to_id]
        
        if not original_msgs.empty:
            original_user = original_msgs.iloc[0]['username']
            reply_user = reply['username']
            
            # Skip self-replies
            if original_user == reply_user:
                continue
            
            # Create edge key (from replier to original author)
            edge_key = (reply_user, original_user)
            interaction_counts[edge_key] = interaction_counts.get(edge_key, 0) + 1
    
    logger.info(f"Found {len(interaction_counts)} unique interaction pairs")
    
    # Add edges to graph with weight
    for (from_user, to_user), count in interaction_counts.items():
        if count >= min_interactions:
            G.add_edge(from_user, to_user, weight=count)
    
    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    
    logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G


def calculate_h_index(df: pd.DataFrame, user: str) -> int:
    """
    Calculate h-index for a user based on replies they received.
    h-index means the user received at least h replies from at least h different people.
    
    Args:
        df: DataFrame with messages
        user: Username to calculate h-index for
        
    Returns:
        h-index value
    """
    if 'reply_to_message_id' not in df.columns:
        return 0
    
    # Get all replies to this user's messages
    user_messages = df[df['username'] == user]['id'].tolist()
    replies_to_user = df[
        (df['reply_to_message_id'].isin(user_messages)) & 
        (df['username'] != user)  # Exclude self-replies
    ]
    
    if replies_to_user.empty:
        return 0
    
    # Count replies from each person
    reply_counts = replies_to_user['username'].value_counts()
    
    # Sort reply counts in descending order
    sorted_counts = sorted(reply_counts.values, reverse=True)
    
    # Calculate h-index
    h_index = 0
    for i, count in enumerate(sorted_counts):
        if count >= i + 1:
            h_index = i + 1
        else:
            break
    
    return h_index


def calculate_graph_metrics(G: nx.DiGraph, df: pd.DataFrame = None) -> Dict:
    """
    Calculate various metrics for the social graph.
    
    Args:
        G: NetworkX directed graph
        df: Original DataFrame for h-index calculation
        
    Returns:
        Dictionary with graph metrics
    """
    metrics = {}
    
    if G.number_of_nodes() == 0:
        return metrics
    
    # Basic metrics
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Centrality measures
    metrics['in_degree_centrality'] = nx.in_degree_centrality(G)
    metrics['out_degree_centrality'] = nx.out_degree_centrality(G)
    metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
    
    # Find most interactive users
    in_degrees = dict(G.in_degree(weight='weight'))
    out_degrees = dict(G.out_degree(weight='weight'))
    
    metrics['most_replied_to'] = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    metrics['most_active_repliers'] = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Calculate h-index for each user if DataFrame is provided
    if df is not None:
        h_indices = {}
        for user in G.nodes():
            h_indices[user] = calculate_h_index(df, user)
        
        metrics['h_indices'] = h_indices
        metrics['top_h_index'] = sorted(h_indices.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Community detection (for undirected version)
    G_undirected = G.to_undirected()
    try:
        communities = list(nx.community.greedy_modularity_communities(G_undirected))
        metrics['communities'] = len(communities)
        metrics['community_sizes'] = [len(c) for c in communities]
    except:
        metrics['communities'] = 0
        metrics['community_sizes'] = []
    
    return metrics


def create_interactive_graph(G: nx.DiGraph, title: str = "User Interaction Network") -> str:
    """
    Create an interactive Plotly visualization of the social graph.
    
    Args:
        G: NetworkX directed graph
        title: Title for the graph
        
    Returns:
        Path to saved image file
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes to visualize")
    
    # Use spring layout with better parameters for readability
    # Weight edges to pull connected nodes closer
    # Use node degrees as weights to cluster active users better
    node_weights = {}
    for node in G.nodes():
        # Active users (high degree) get higher weight
        degree = G.degree(node, weight='weight')
        node_weights[node] = 1.0 + degree / 10.0
    
    # Try to detect communities for better layout
    G_undirected = G.to_undirected()
    try:
        communities = list(nx.community.greedy_modularity_communities(G_undirected))
        # Create a mapping of nodes to communities
        node_to_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_to_community[node] = i
    except:
        node_to_community = {node: 0 for node in G.nodes()}
    
    # Create position dictionary with Kamada-Kawai algorithm
    # This algorithm is good for weighted graphs and creates stable layouts
    pos = nx.kamada_kawai_layout(
        G, 
        weight='weight',  # Use edge weights for better positioning
        scale=2
    )
    
    # Adjust positions to group communities together
    # Calculate community centers
    community_centers = {}
    for node, (x, y) in pos.items():
        comm = node_to_community.get(node, 0)
        if comm not in community_centers:
            community_centers[comm] = {'x': [], 'y': []}
        community_centers[comm]['x'].append(x)
        community_centers[comm]['y'].append(y)
    
    # Calculate actual centers
    for comm in community_centers:
        community_centers[comm] = {
            'x': sum(community_centers[comm]['x']) / len(community_centers[comm]['x']),
            'y': sum(community_centers[comm]['y']) / len(community_centers[comm]['y'])
        }
    
    # Slightly pull nodes towards their community centers
    for node in pos:
        comm = node_to_community.get(node, 0)
        if comm in community_centers and len(community_centers) > 1:
            center = community_centers[comm]
            x, y = pos[node]
            # Move 20% towards community center
            pos[node] = (
                x * 0.8 + center['x'] * 0.2,
                y * 0.8 + center['y'] * 0.2
            )
    
    # Create edge traces
    edge_traces = []
    annotations = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        
        # Edge trace
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=math.log(2*weight+1) * 0.2,  # Logarithmic scale for edge width
                color='rgba(125,125,125,0.4)'  # Slightly more transparent
            ),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
        
        # Add arrow annotation for direction
        annotations.append(
            dict(
                ax=x0,
                ay=y0,
                axref='x',
                ayref='y',
                x=x1,
                y=y1,
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=0.7,  # Smaller arrows (was 1)
                arrowwidth=min(weight * 0.15, 2),  # Thinner arrows (was 0.3, max was 3)
                arrowcolor='rgba(125,125,125,0.4)',
                opacity=0.4  # More transparent (was 0.5)
            )
        )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Calculate node size based on total degree
        in_degree = G.in_degree(node, weight='weight')
        out_degree = G.out_degree(node, weight='weight')
        total_degree = in_degree + out_degree
        # Use logarithmic scale for better visualization
        # Reduced sizes: base 15 (was 30), multiplier 8 (was 15), max 40 (was 70)
        node_size = 15 + min(math.log(total_degree + 1) * 8, 40)
        node_sizes.append(node_size)
        
        # Create hover text
        hover_text = f"<b>{node}</b><br>"
        hover_text += f"Получено ответов: {in_degree}<br>"
        hover_text += f"Отправлено ответов: {out_degree}<br>"
        hover_text += f"Всего взаимодействий: {total_degree}"
        node_text.append(hover_text)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node for node in G.nodes()],
        hovertext=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_sizes,
            color=[G.degree(node, weight='weight') for node in G.nodes()],
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
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=annotations,  # Only arrow annotations, no legend
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        width=1200,
        height=800
    )
    
    # Save image
    filename = "interaction_graph.png"
    fig.write_image(filename, scale=2)
    
    return filename


def generate_graph_statistics(G: nx.DiGraph, metrics: Dict) -> str:
    """
    Generate text statistics about the interaction graph.
    
    Args:
        G: NetworkX directed graph
        metrics: Dictionary with graph metrics
        
    Returns:
        Formatted statistics text
    """
    if not metrics or metrics.get('nodes', 0) == 0:
        return "Недостаточно данных для построения графа взаимодействий."
    
    text = "📊 **Статистика графа взаимодействий**\n\n"
    
    text += f"**Основные метрики:**\n"
    text += f"• Участников: {metrics['nodes']}\n"
    text += f"• Связей: {metrics['edges']}\n"
    text += f"• Плотность графа: {metrics['density']:.3f}\n"
    
    if metrics.get('communities', 0) > 0:
        text += f"• Сообществ: {metrics['communities']}\n"
        if metrics['community_sizes']:
            text += f"• Размеры сообществ: {', '.join(map(str, sorted(metrics['community_sizes'], reverse=True)))}\n"
    
    text += f"\n**🎯 Самые популярные (получают больше всего ответов):**\n"
    for i, (user, score) in enumerate(metrics.get('most_replied_to', [])[:5], 1):
        text += f"{i}. {user} — {score} ответов\n"
    
    text += f"\n**💬 Самые активные (отвечают чаще всех):**\n"
    for i, (user, score) in enumerate(metrics.get('most_active_repliers', [])[:5], 1):
        text += f"{i}. {user} — {score} ответов\n"
    
    # H-index rankings
    if 'top_h_index' in metrics and metrics['top_h_index']:
        text += f"\n**🎓 H-index (научная продуктивность):**\n"
        for i, (user, h_index) in enumerate(metrics['top_h_index'], 1):
            if h_index > 0:
                text += f"{i}. {user} — h-index: {h_index}\n"
        text += f"*h-index показывает что пользователь получил минимум h ответов от минимум h разных людей*\n"
    
    # Find hubs (high betweenness centrality)
    betweenness = metrics.get('betweenness_centrality', {})
    if betweenness:
        hubs = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
        if any(score > 0 for _, score in hubs):
            text += f"\n**🌟 Ключевые связующие (хабы):**\n"
            for i, (user, score) in enumerate(hubs, 1):
                if score > 0:
                    text += f"{i}. {user} (коэффициент: {score:.3f})\n"
    
    return text