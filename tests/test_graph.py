#!/usr/bin/env python3
"""
Test script for graph visualization without running the bot.
Usage: python test_graph.py [min_interactions]
"""

import sys
import logging
from analytics import load_and_prepare_data
from social_graph import (
    build_interaction_graph,
    calculate_graph_metrics,
    create_interactive_graph,
    generate_graph_statistics
)
from config import DB_FILE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_graph(min_interactions=3):
    """Test graph generation locally."""
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
        logger.error(
            f"No interactions found with minimum threshold of {min_interactions} replies.\n"
            f"Try reducing the threshold."
        )
        return
    
    logger.info(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate metrics (including h-index)
    logger.info("Calculating graph metrics...")
    metrics = calculate_graph_metrics(G, df)
    
    # Generate text statistics
    stats_text = generate_graph_statistics(G, metrics)
    print("\n" + stats_text)
    
    # Create visualization
    logger.info("Creating graph visualization...")
    try:
        graph_file = create_interactive_graph(
            G, 
            f"Граф взаимодействий (мин. {min_interactions} ответов)"
        )
        logger.info(f"Graph saved to: {graph_file}")
        print(f"\n✅ График сохранен в файл: {graph_file}")
        
        # Show some debug info
        print(f"\nИнформация о графе:")
        print(f"- Узлов: {G.number_of_nodes()}")
        print(f"- Связей: {G.number_of_edges()}")
        print(f"- Плотность: {metrics.get('density', 0):.3f}")
        print(f"- Сообществ: {metrics.get('communities', 0)}")
        
    except Exception as e:
        logger.error(f"Error creating graph: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Get min_interactions from command line argument if provided
    min_interactions = 3
    if len(sys.argv) > 1:
        try:
            min_interactions = int(sys.argv[1])
        except ValueError:
            print(f"Invalid min_interactions value: {sys.argv[1]}")
            print("Usage: python test_graph.py [min_interactions]")
            sys.exit(1)
    
    test_graph(min_interactions)