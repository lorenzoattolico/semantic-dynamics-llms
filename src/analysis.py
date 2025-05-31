"""
Analysis module.

This module handles:
1. Loading the final dataset with spreading activation and similarity scores
2. Analyzing correlations between activation and similarity
3. Visualizing results with various plots
4. Examining effects of parameter configurations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

from src.utils import setup_logger
from config import (
    FINAL_RESULTS_FILE, RETENTION_LEVELS, TIME_STEPS, 
    PLOTS_DIR, LIKERT_SCALE_MAX
)

# Setup logger
logger = setup_logger("analysis")

def load_final_results():
    """
    Load the final dataset with spreading activation and similarity scores.
    
    Returns:
        pd.DataFrame: The final dataset
    """
    logger.info(f"Loading final results from {FINAL_RESULTS_FILE}")
    try:
        df = pd.read_csv(FINAL_RESULTS_FILE)
        logger.info(f"Loaded {len(df)} concept pairs with activation and similarity scores")
        
        # Calculate average activation if not already present
        for r in [int(r*10) for r in RETENTION_LEVELS]:
            for t in TIME_STEPS:
                config = f"r{r}_t{t}_AVG"
                atob_col = f"r{r}_t{t}_AtoB"
                btoa_col = f"r{r}_t{t}_BtoA"
                
                if config not in df.columns and atob_col in df.columns and btoa_col in df.columns:
                    df[config] = df[[atob_col, btoa_col]].mean(axis=1)
                    logger.info(f"Calculated mean between {atob_col} and {btoa_col} as {config}")
        
        # Add Likert group category
        df['LikertGroup'] = 'Medium (4)'
        df.loc[df['LikertScore'] <= 3, 'LikertGroup'] = 'Low (1-3)'
        df.loc[df['LikertScore'] >= 5, 'LikertGroup'] = 'High (5-7)'
        
        return df
    except Exception as e:
        logger.error(f"Error loading final results: {e}")
        raise

def analyze_correlations(df):
    """
    Analyze correlations between spreading activation and similarity judgments.
    
    Args:
        df (pd.DataFrame): The final dataset
        
    Returns:
        pd.DataFrame: Correlation results
    """
    logger.info("Analyzing correlations between activation and similarity")
    
    correlation_data = []
    
    for r in [int(r*10) for r in RETENTION_LEVELS]:
        for t in TIME_STEPS:
            config = f"r{r}_t{t}_AVG"
            
            if config in df.columns:
                corr_spearman, p_spearman = stats.spearmanr(df['LikertScore'], df[config])
                
                correlation_data.append({
                    'Retention': r/10,  # convert back to 0.1, 0.5, 0.9
                    'Time': t,
                    'Config': config,
                    'SpearmanCorr': corr_spearman,
                    'SpearmanP': p_spearman
                })
    
    corr_df = pd.DataFrame(correlation_data)
    
    # Find configuration with highest correlation
    max_spearman = corr_df.loc[corr_df['SpearmanCorr'].idxmax()]
    
    logger.info("\nCorrelation analysis results:")
    logger.info(f"Highest Spearman correlation: rho = {max_spearman['SpearmanCorr']:.4f}")
    logger.info(f"Best configuration: Retention = {max_spearman['Retention']}, Time = {max_spearman['Time']}")
    
    return corr_df, max_spearman

def plot_correlation_heatmap(corr_df):
    """
    Plot correlation heatmap for all parameter configurations.
    
    Args:
        corr_df (pd.DataFrame): Correlation results
        
    Returns:
        None: Saves plot to disk
    """
    logger.info("Generating correlation heatmap")
    
    pivot_spearman = corr_df.pivot_table(
        index='Time', 
        columns='Retention', 
        values='SpearmanCorr'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_spearman, annot=True, cmap='coolwarm', fmt='.3f')
    plt.title('Spearman Correlation between Likert Score and Spreading Activation', fontsize=14)
    plt.xlabel('Retention')
    plt.ylabel('Time')
    plt.tight_layout()
    
    save_path = PLOTS_DIR / 'heatmap_spearman_correlations.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Heatmap saved to {save_path}")
    
    plt.close()

def plot_optimal_configuration(df, optimal_config):
    """
    Plot distribution of activation values across Likert scores for optimal configuration.
    
    Args:
        df (pd.DataFrame): The final dataset
        optimal_config (str): Optimal configuration name
        
    Returns:
        None: Saves plot to disk
    """
    logger.info(f"Analyzing optimal configuration: {optimal_config}")
    
    # Extract parameters from config name
    r_val = float(optimal_config.split('_')[0].replace('r', '')) / 10
    t_val = int(optimal_config.split('_')[1].replace('t', ''))
    
    # Get statistics for optimal configuration by Likert score
    stats_by_likert = df.groupby('LikertScore')[optimal_config].describe()
    logger.info("Statistics for optimal configuration by Likert score:")
    logger.info(stats_by_likert.round(4))
    
    # Plot boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='LikertScore', y=optimal_config, data=df)
    
    # Add median line
    medians = df.groupby('LikertScore')[optimal_config].median()
    plt.plot(range(len(medians)), medians.values, 'r-', linewidth=2, label='Median')
    
    plt.title(f'Distribution of Spreading Activation (Retention={r_val}, Time={t_val}) by Likert Score', fontsize=14)
    plt.xlabel('Likert Score', fontsize=12)
    plt.ylabel('Spreading Activation Value (mean AtoB/BtoA)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path = PLOTS_DIR / 'boxplot_optimal_config.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Optimal configuration plot saved to {save_path}")
    
    plt.close()

def plot_parameter_effects(df):
    """
    Plot effects of retention and time parameters on activation patterns.
    
    Args:
        df (pd.DataFrame): The final dataset
        
    Returns:
        None: Saves plots to disk
    """
    logger.info("Analyzing parameter effects")
    
    # A. Effect of retention (fixed time = 10)
    t_fixed = 10
    r_values = [int(r*10) for r in RETENTION_LEVELS]
    retention_configs = [f"r{r}_t{t_fixed}_AVG" for r in r_values if f"r{r}_t{t_fixed}_AVG" in df.columns]
    
    plt.figure(figsize=(10, 8))
    for config in retention_configs:
        r_val = int(config.split('_')[0].replace('r', '')) / 10
        r_label = f"Retention {r_val}"
        grouped = df.groupby('LikertScore')[config].median()
        plt.plot(grouped.index, grouped.values, 'o-', linewidth=2, label=r_label)
    
    plt.title('Effect of Retention (Time = 10 fixed)', fontsize=14)
    plt.xlabel('Likert Score', fontsize=12)
    plt.ylabel('Spreading Activation (median)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path = PLOTS_DIR / 'effect_of_retention.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Retention effect plot saved to {save_path}")
    
    plt.close()
    
    # B. Effect of time (fixed retention = 0.5)
    r_fixed = 5  # corresponds to 0.5
    time_configs = [f"r{r_fixed}_t{t}_AVG" for t in TIME_STEPS if f"r{r_fixed}_t{t}_AVG" in df.columns]
    
    plt.figure(figsize=(10, 8))
    for config in time_configs:
        t_val = int(config.split('_')[1].replace('t', ''))
        t_label = f"Time {t_val}"
        grouped = df.groupby('LikertScore')[config].median()
        plt.plot(grouped.index, grouped.values, 'o-', linewidth=2, label=t_label)
    
    plt.title('Effect of Time (Retention = 0.5 fixed)', fontsize=14)
    plt.xlabel('Likert Score', fontsize=12)
    plt.ylabel('Spreading Activation (median)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path = PLOTS_DIR / 'effect_of_time.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Time effect plot saved to {save_path}")
    
    plt.close()

def analyze_path_length(df):
    """
    Analyze the relationship between network path length and similarity ratings.
    
    Args:
        df (pd.DataFrame): The final dataset
        
    Returns:
        None: Saves plot to disk
    """
    if 'PathLength' not in df.columns:
        logger.warning("PathLength column not found, skipping path length analysis")
        return
    
    logger.info("Analyzing relationship between path length and similarity")
    
    # Create path length categories
    df['PathLengthCat'] = pd.cut(
        df['PathLength'],
        bins=[-2, 0, 1, 2, 3, float('inf')],
        labels=['Not connected', 'Adjacent', 'Distance 2', 'Distance 3', 'Distance 4+']
    )
    
    # Calculate statistics by path length
    path_stats = df.groupby('PathLengthCat')['LikertScore'].describe()
    logger.info("Likert score statistics by path length category:")
    logger.info(path_stats.round(4))
    
    # Calculate correlation between path length and Likert score
    corr_path, p_path = stats.spearmanr(df['PathLength'], df['LikertScore'])
    logger.info(f"Spearman correlation between PathLength and LikertScore: rho = {corr_path:.4f} (p-value = {p_path:.6f})")
    
    # Plot boxplot of Likert scores by path length
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='PathLengthCat', y='LikertScore', data=df)
    plt.title('Distribution of Likert Scores by Path Length', fontsize=14)
    plt.xlabel('Path Length', fontsize=12)
    plt.ylabel('Likert Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = PLOTS_DIR / 'boxplot_path_length.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Path length plot saved to {save_path}")
    
    plt.close()

def main():
    """
    Main function to run the analysis.
    """
    logger.info("Starting analysis")
    
    # Load final results
    df = load_final_results()
    
    # Analyze correlations
    corr_df, max_spearman = analyze_correlations(df)
    
    # Plot correlation heatmap
    plot_correlation_heatmap(corr_df)
    
    # Plot optimal configuration
    optimal_config = max_spearman['Config']
    plot_optimal_configuration(df, optimal_config)
    
    # Plot parameter effects
    plot_parameter_effects(df)
    
    # Analyze path length
    analyze_path_length(df)
    
    logger.info("Analysis complete")

if __name__ == "__main__":
    main()
