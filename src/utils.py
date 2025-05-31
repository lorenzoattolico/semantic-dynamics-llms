"""
Utility functions for the Semantic Dynamics in LLMs project.
"""

import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set default plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with a specific name and configuration.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file
        level (int, optional): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / log_file
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def ensure_directory(directory_path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path (str or Path): Directory path
        
    Returns:
        Path: Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_figure(fig, filename, directory="results/plots", dpi=300, **kwargs):
    """
    Save a figure to disk.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        filename (str): Filename
        directory (str): Directory to save in
        dpi (int): DPI for saved figure
        **kwargs: Additional arguments for plt.savefig
        
    Returns:
        Path: Path to saved figure
    """
    dir_path = ensure_directory(directory)
    path = dir_path / filename
    fig.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)
    return path
