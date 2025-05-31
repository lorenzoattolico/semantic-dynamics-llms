"""
Spreading activation module.

This module handles:
1. Loading the semantic network
2. Generating concept pairs for analysis
3. Calculating spreading activation for different parameter configurations
"""

import random
import pandas as pd
import networkx as nx
from itertools import combinations
from tqdm import tqdm
from SpreadPy.Models.models import BaseSpreading

from src.utils import setup_logger
from config import (
    EDGES_FILE, NUM_PAIRS, RETENTION_LEVELS, TIME_STEPS, 
    DECAY, SUPPRESS, SPREADING_RESULTS_FILE
)

# Setup logger
logger = setup_logger("spreading_activation")

def load_semantic_network():
    """
    Load the semantic network from edge list file.
    
    Returns:
        nx.Graph: The semantic network graph
    """
    logger.info(f"Loading semantic network from {EDGES_FILE}")
    try:
        df_edges = pd.read_csv(EDGES_FILE)
        G = nx.from_pandas_edgelist(df_edges, source='source', target='target')
        logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        raise

def generate_concept_pairs(G, num_pairs=NUM_PAIRS):
    """
    Generate random concept pairs for analysis, stratified by path length.
    
    Args:
        G (nx.Graph): The semantic network graph
        num_pairs (int): Number of pairs to generate
        
    Returns:
        list: List of tuples representing concept pairs
    """
    logger.info(f"Generating {num_pairs} concept pairs")
    
    # Get all possible pairs
    all_pairs = list(combinations(G.nodes(), 2))
    
    # Stratify by path length to ensure representation of closely and distantly related concepts
    pairs_by_distance = {}
    
    # Sample a subset for path length calculation (for efficiency)
    sample_size = min(len(all_pairs), num_pairs * 10)
    sample_pairs = random.sample(all_pairs, sample_size)
    
    for a, b in tqdm(sample_pairs, desc="Categorizing pairs by distance"):
        try:
            distance = nx.shortest_path_length(G, a, b)
        except nx.NetworkXNoPath:
            distance = -1  # No path
            
        if distance not in pairs_by_distance:
            pairs_by_distance[distance] = []
            
        pairs_by_distance[distance].append((a, b))
    
    # Determine how many to sample from each distance category
    total_sampled = 0
    pairs_per_category = num_pairs // len(pairs_by_distance)
    concept_pairs = []
    
    for distance, pairs in pairs_by_distance.items():
        to_sample = min(pairs_per_category, len(pairs))
        sampled = random.sample(pairs, to_sample)
        concept_pairs.extend(sampled)
        total_sampled += to_sample
    
    # If we need more pairs, sample randomly from all categories
    if total_sampled < num_pairs:
        remaining = num_pairs - total_sampled
        all_remaining = [p for pairs in pairs_by_distance.values() for p in pairs if p not in concept_pairs]
        if all_remaining:
            additional = random.sample(all_remaining, min(remaining, len(all_remaining)))
            concept_pairs.extend(additional)
    
    logger.info(f"Generated {len(concept_pairs)} concept pairs")
    return concept_pairs[:num_pairs]

def get_activation(G, source, target, retention, t_max):
    """
    Calculate spreading activation from source to target.
    
    Args:
        G (nx.Graph): The semantic network graph
        source (str): Source concept
        target (str): Target concept
        retention (float): Activation retention parameter
        t_max (int): Maximum time steps
        
    Returns:
        float: Activation value at target
    """
    try:
        model = BaseSpreading(G, retention=retention, decay=DECAY, suppress=SUPPRESS)
        
        # Set initial activation
        initial_status = {node: 100 if node == source else 0 for node in G.nodes}
        model.status = initial_status
        
        # Run simulation
        results = model.iteration_bunch(t_max + 1)
        
        # Get activation at target
        final_status = results[t_max]['status']
        activation = final_status.get(target, 0)
        
        return activation
    except Exception as e:
        logger.error(f"Error calculating activation from {source} to {target}: {e}")
        return 0.0

def calculate_spreading_activation(G, concept_pairs):
    """
    Calculate spreading activation for all concept pairs with different parameter configurations.
    
    Args:
        G (nx.Graph): The semantic network graph
        concept_pairs (list): List of concept pairs
        
    Returns:
        pd.DataFrame: DataFrame with spreading activation results
    """
    logger.info("Calculating spreading activation for all pairs")
    
    spreading_data = []
    
    for a, b in tqdm(concept_pairs, desc="Processing pairs"):
        row = {"Concept1": a, "Concept2": b}
        
        # Calculate path length
        try:
            row["PathLength"] = nx.shortest_path_length(G, a, b)
        except nx.NetworkXNoPath:
            row["PathLength"] = -1
        
        # Calculate spreading activation for all parameter configurations
        for r in RETENTION_LEVELS:
            for t in TIME_STEPS:
                key_prefix = f"r{int(r*10)}_t{t}"
                try:
                    act_ab = get_activation(G, a, b, r, t)
                    act_ba = get_activation(G, b, a, r, t)
                except Exception as e:
                    logger.error(f"Error calculating activation for {a}-{b} (r={r}, t={t}): {e}")
                    act_ab, act_ba = 0.0, 0.0
                    
                row[f"{key_prefix}_AtoB"] = act_ab
                row[f"{key_prefix}_BtoA"] = act_ba
                row[f"{key_prefix}_AVG"] = (act_ab + act_ba) / 2
        
        spreading_data.append(row)
    
    # Convert to DataFrame and save
    df_spreading = pd.DataFrame(spreading_data)
    df_spreading.to_csv(SPREADING_RESULTS_FILE, index=False)
    
    logger.info(f"Spreading activation results saved to {SPREADING_RESULTS_FILE}")
    return df_spreading

def main():
    """
    Main function to run the spreading activation analysis.
    """
    logger.info("Starting spreading activation analysis")
    
    # Load semantic network
    G = load_semantic_network()
    
    # Generate concept pairs
    concept_pairs = generate_concept_pairs(G)
    
    # Calculate spreading activation
    df_spreading = calculate_spreading_activation(G, concept_pairs)
    
    logger.info("Spreading activation analysis complete")
    logger.info(f"Generated {len(df_spreading)} pairs with activation values")

if __name__ == "__main__":
    main()
