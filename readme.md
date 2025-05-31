# Semantic Dynamics in LLMs: Spreading Activation and Similarity Judgments

This repository contains the code to replicate the experiments from the paper "Semantic Dynamics in LLMs: Spreading Activation and Similarity Judgments" by Lorenzo Domenico Attolico, University of Trento (May 15, 2025).

## Overview

This study investigates whether spreading activation, a classical cognitive theory, can explain semantic relationships in Large Language Models (LLMs). Using Mistral as the reference model, we:

1. Construct a 200-node semantic network through seed-based prompting
2. Collect similarity judgments for 1,500 concept pairs
3. Simulate activation propagation across the network with varying parameters
4. Analyze the correlation between activation patterns and explicit similarity judgments

## Key Findings

- Significant correlation between activation patterns and explicit similarity judgments
- Correlation peaks at intermediate parameter values (retention = 0.5, steps = 10)
- Network properties (path length and clustering) systematically relate to similarity judgments
- Spreading activation reveals implicit structural properties in LLM representations

## Project Structure

```
semantic-dynamics-llm/
├── config.py                 # Configuration parameters
├── data/                     # Data directory
│   └── README.md             # Data description
├── main.py                   # Main script to run the experiment pipeline
├── notebooks/                # Jupyter notebooks for exploration and visualization
│   ├── analisi.ipynb         # Analysis notebook
│   ├── coppie.ipynb          # Pair generation and spreading activation notebook
│   └── generazione_grafo.ipynb  # Graph generation notebook
├── README.md                 # This file
├── requirements.txt          # Project dependencies
└── src/                      # Source code
    ├── __init__.py           # Package initialization
    ├── analysis.py           # Analysis and visualization code
    ├── graph_generator.py    # Semantic network generation
    ├── similarity_evaluator.py  # Likert score evaluation using LLM
    ├── spreading_activation.py  # Spreading activation algorithm
    └── utils.py              # Utility functions
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/semantic-dynamics-llm.git
cd semantic-dynamics-llm

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The experiment can be run end-to-end using the main script:

```bash
python main.py
```

Or you can run individual components:

```bash
# Generate semantic network
python -m src.graph_generator

# Calculate spreading activation and collect similarity judgments
python -m src.spreading_activation
python -m src.similarity_evaluator

# Run analysis
python -m src.analysis
```

## Configuration

Modify `config.py` to adjust experiment parameters:

- `SEED_CONCEPTS`: List of seed words for network generation
- `RETENTION_LEVELS`: Retention rates for spreading activation (0.1, 0.5, 0.9)
- `TIME_STEPS`: Propagation steps for spreading activation (1, 10, 100)
- `MODEL_NAME`: LLM model name ("mistral-medium")
- Other parameters for network construction and API configuration

## Results

The analysis produces various visualizations:

1. Correlation heatmap between spreading activation and similarity judgments
2. Distribution of activation values across Likert scores
3. Effect of retention rate and time steps on activation patterns
4. Relationship between network path length and similarity ratings

## Citation

If you use this code in your research, please cite:

```
@article{attolico2025semantic,
  title={Semantic Dynamics in LLMs: Spreading Activation and Similarity Judgments},
  author={Attolico, Lorenzo Domenico},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
