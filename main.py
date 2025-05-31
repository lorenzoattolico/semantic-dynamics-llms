"""
Main script for the Semantic Dynamics in LLMs project.

This script orchestrates the full experimental pipeline:
1. Generate semantic network from seed concepts
2. Calculate spreading activation for concept pairs
3. Evaluate similarity using LLM
4. Analyze and visualize results
"""

import argparse
import time
from pathlib import Path

from src.utils import setup_logger
from src import graph_generator
from src import spreading_activation
from src import similarity_evaluator
from src import analysis
from config import FINAL_RESULTS_FILE

# Setup logger
logger = setup_logger("main")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Semantic Dynamics in LLMs: Spreading Activation and Similarity Judgments"
    )
    
    parser.add_argument(
        "--steps", 
        type=str, 
        default="all",
        choices=["all", "generate", "spread", "evaluate", "analyze"],
        help="Steps to run (default: all)"
    )
    
    parser.add_argument(
        "--skip-existing", 
        action="store_true",
        help="Skip steps if output files already exist"
    )
    
    return parser.parse_args()

def run_graph_generation(skip_existing=False):
    """
    Run the graph generation step.
    
    Args:
        skip_existing (bool): Skip if output files exist
    """
    if skip_existing and Path(graph_generator.EDGES_FILE).exists():
        logger.info("Skipping graph generation (edge file exists)")
        return
        
    logger.info("=== STEP 1: GENERATING SEMANTIC NETWORK ===")
    graph_generator.main()
    logger.info("Graph generation complete")

def run_spreading_activation(skip_existing=False):
    """
    Run the spreading activation step.
    
    Args:
        skip_existing (bool): Skip if output files exist
    """
    if skip_existing and Path(spreading_activation.SPREADING_RESULTS_FILE).exists():
        logger.info("Skipping spreading activation (results file exists)")
        return
        
    logger.info("=== STEP 2: CALCULATING SPREADING ACTIVATION ===")
    spreading_activation.main()
    logger.info("Spreading activation calculation complete")

def run_similarity_evaluation(skip_existing=False):
    """
    Run the similarity evaluation step.
    
    Args:
        skip_existing (bool): Skip if output files exist
    """
    if skip_existing and Path(FINAL_RESULTS_FILE).exists():
        logger.info("Skipping similarity evaluation (final results file exists)")
        return
        
    logger.info("=== STEP 3: EVALUATING SIMILARITY ===")
    similarity_evaluator.main()
    logger.info("Similarity evaluation complete")

def run_analysis():
    """
    Run the analysis step.
    """
    logger.info("=== STEP 4: ANALYZING RESULTS ===")
    analysis.main()
    logger.info("Analysis complete")

def main():
    """
    Main function to run the full experimental pipeline.
    """
    args = parse_arguments()
    start_time = time.time()
    
    logger.info("Starting Semantic Dynamics in LLMs experiment")
    
    try:
        # Run selected steps
        if args.steps in ["all", "generate"]:
            run_graph_generation(args.skip_existing)
        
        if args.steps in ["all", "spread"]:
            run_spreading_activation(args.skip_existing)
        
        if args.steps in ["all", "evaluate"]:
            run_similarity_evaluation(args.skip_existing)
        
        if args.steps in ["all", "analyze"]:
            run_analysis()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Experiment completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in experiment: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
