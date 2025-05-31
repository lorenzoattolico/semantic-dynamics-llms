"""
Semantic network generation module.

This module handles the creation of a semantic network by:
1. Generating subgraphs for seed concepts using an LLM
2. Combining subgraphs into a complete semantic network
3. Verifying and adding semantic links to reach target density
"""

import time
import json
import random
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

from src.utils import setup_logger
from config import (
    API_KEY, LLM_URL, MODEL_NAME, SEED_CONCEPTS, NODES_PER_SEED, 
    INITIAL_EDGES, TARGET_DENSITY, BATCH_SIZE, RETRY_DELAY, MAX_RETRIES,
    EDGES_FILE, NODES_FILE
)

# Setup logger
logger = setup_logger("graph_generator")

def ask_llm(prompt, batch_id=None):
    """
    Query the LLM API with a given prompt.
    
    Args:
        prompt (str): The prompt to send to the LLM
        batch_id (int, optional): Batch identifier for logging
        
    Returns:
        str: The LLM's response
    """
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Request (batch {batch_id}, attempt {attempt + 1})")
            response = requests.post(LLM_URL, headers=headers, json=data, timeout=60)
            json_data = response.json()
            
            if "choices" not in json_data:
                logger.warning(f"Unexpected response (batch {batch_id}): {json.dumps(json_data, indent=2)}")
                return ""
                
            return json_data["choices"][0]["message"]["content"].strip().lower()
        except Exception as e:
            logger.error(f"LLM error: {e}")
            time.sleep(RETRY_DELAY)
            
    return ""

def generate_subgraph(seed):
    """
    Generate a semantic subgraph for a given seed concept.
    
    Args:
        seed (str): Seed concept
        
    Returns:
        tuple: (list of nodes, list of edges)
    """
    prompt = f"""
Create a dense semantic network based on the concept '{seed}'.

REQUIREMENTS:
- Include EXACTLY {INITIAL_EDGES} pairs.
- Use ONLY {NODES_PER_SEED} distinct words.
- Format: concept1, concept2 (one per line).
- Include the seed word '{seed}' in at least 5 of the pairs.
- Do NOT include explanations or extra text.
"""
    response = ask_llm(prompt)
    edges = []
    nodes = set()
    
    for line in response.split("\n"):
        if "," in line:
            a, b = [x.strip().lower() for x in line.split(",", 1)]
            edges.append((a, b))
            nodes.update([a, b])
            
    return list(nodes), edges

def parse_confirmed_pairs(response):
    """
    Parse the LLM's response to extract confirmed semantic pairs.
    
    Args:
        response (str): LLM response text
        
    Returns:
        list: List of tuples representing confirmed pairs
    """
    lines = [line.strip().lower() for line in response.split("\n") if "," in line]
    return [(a.strip(), b.strip()) for line in lines 
            for a, b in [line.split(",", 1)] 
            if a and b and a != b]

def check_semantic_links(graph):
    """
    Check semantic links between concept pairs using the LLM.
    
    Args:
        graph (nx.Graph): The semantic network graph
        
    Returns:
        None: Updates graph in-place
    """
    def build_prompt(pairs):
        formatted = "\n".join([f"- {a}, {b}" for a, b in pairs])
        return f"""
Below are concept pairs. Indicate only those that are very very strongly semantically related, formatted as:

concept1, concept2

Pairs:
{formatted}

Respond only with the list. No explanation.
"""

    non_edges = list(nx.non_edges(graph))
    random.shuffle(non_edges)
    total_pairs = len(non_edges)
    total_batches = total_pairs // BATCH_SIZE + (1 if total_pairs % BATCH_SIZE > 0 else 0)

    logger.info(f"Total pairs to check: {total_pairs}")
    logger.info(f"Total batches to process (up to density {TARGET_DENSITY}): {total_batches}")

    with tqdm(total=int(TARGET_DENSITY * total_pairs), 
              desc="Checking semantic links") as pbar:
        i = 0
        while nx.density(graph) < TARGET_DENSITY and i < total_pairs:
            batch = non_edges[i:i + BATCH_SIZE]
            i += BATCH_SIZE
            
            prompt = build_prompt(batch)
            response = ask_llm(prompt, batch_id=i // BATCH_SIZE)
            
            if not response:
                continue
                
            confirmed = parse_confirmed_pairs(response)
            for a, b in confirmed:
                if a in graph and b in graph:
                    graph.add_edge(a, b)
                    logger.info(f"Linked: {a} – {b}")
                    pbar.update(1)
                    
            time.sleep(2)  # Rate limiting

def export_graph(graph):
    """
    Export the graph to CSV files.
    
    Args:
        graph (nx.Graph): The semantic network graph
        
    Returns:
        None: Writes files to disk
    """
    edges = [{"source": u, "target": v} for u, v in graph.edges()]
    nodes = [{"node": n, "degree": graph.degree(n)} for n in graph.nodes()]
    
    pd.DataFrame(edges).to_csv(EDGES_FILE, index=False)
    pd.DataFrame(nodes).to_csv(NODES_FILE, index=False)
    
    logger.info(f"Exported edges to {EDGES_FILE}")
    logger.info(f"Exported nodes to {NODES_FILE}")

def draw_graph(graph, title="Semantic Network", save_path=None):
    """
    Visualize the semantic network graph.
    
    Args:
        graph (nx.Graph): The semantic network graph
        title (str): Plot title
        save_path (str, optional): Path to save the plot
        
    Returns:
        None: Displays or saves the plot
    """
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(graph, seed=42, k=0.2)
    
    node_colors = ['tomato' if n in SEED_CONCEPTS else 'skyblue' for n in graph.nodes()]
    node_sizes = [300 + 100 * graph.degree(n) for n in graph.nodes()]
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(graph, pos, alpha=0.4)
    nx.draw_networkx_labels(graph, pos, font_size=9)
    
    plt.title(f"{title} (Density: {nx.density(graph):.3f})")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graph visualization saved to {save_path}")
    
    plt.show()

def generate_semantic_network():
    """
    Generate a complete semantic network by:
    1. Creating subgraphs for each seed concept
    2. Combining subgraphs
    3. Adding semantic links to reach target density
    
    Returns:
        nx.Graph: The generated semantic network
    """
    G = nx.Graph()
    
    # Generate subgraphs for each seed concept
    for seed in tqdm(SEED_CONCEPTS, desc="Generating seed subgraphs"):
        logger.info(f"Generating subgraph for: {seed}")
        nodes, edges = generate_subgraph(seed)
        
        for node in nodes:
            G.add_node(node)
        for a, b in edges:
            G.add_edge(a, b)
            
        time.sleep(2)  # Rate limiting
    
    # Check and add semantic links to reach target density
    logger.info(f"Initial graph density: {nx.density(G):.4f}")
    check_semantic_links(G)
    logger.info(f"Final graph density: {nx.density(G):.4f}")
    
    # Export the graph
    export_graph(G)
    
    return G

def main():
    """
    Main function to generate and visualize the semantic network.
    """
    logger.info("Starting semantic network generation")
    G = generate_semantic_network()
    draw_graph(G, title="Semantic Network", 
              save_path="results/plots/semantic_network.png")
    logger.info("Semantic network generation complete")

if __name__ == "__main__":
    main()
