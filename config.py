"""
Configuration parameters for the Semantic Dynamics in LLMs project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# File paths
EDGES_FILE = DATA_DIR / "semantic_edges.csv"
NODES_FILE = DATA_DIR / "semantic_nodes.csv"
SPREADING_RESULTS_FILE = RESULTS_DIR / "spreading_results.csv"
FINAL_RESULTS_FILE = RESULTS_DIR / "semantic_analysis_results.csv"

# LLM API configuration
# Note: Use environment variables for sensitive information
API_KEY = os.getenv("MISTRAL_API_KEY", "")
LLM_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_NAME = "mistral-medium"

# Graph generation parameters
SEED_CONCEPTS = [
    "dog", "fruit", "car", "computer", "teacher",
    "school", "fear", "feeling", "computer", "device",
    "bicycle", "sport", "music", "animal", "vehicle",
    "apple", "football", "chair", "office", "job"
]
NODES_PER_SEED = 10
INITIAL_EDGES = 15
TARGET_DENSITY = 0.04

# Spreading activation parameters
RETENTION_LEVELS = [0.1, 0.5, 0.9]
TIME_STEPS = [1, 10, 100]
DECAY = 0
SUPPRESS = 0

# Similarity evaluation parameters
NUM_PAIRS = 1500

# API request parameters
BATCH_SIZE = 25
RETRY_DELAY = 10
MAX_RETRIES = 5

# Analysis parameters
LIKERT_SCALE_MAX = 7
