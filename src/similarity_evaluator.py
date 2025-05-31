"""
Similarity evaluator module.

This module handles:
1. Loading spreading activation results
2. Obtaining Likert similarity scores from the LLM
3. Saving combined results for analysis
"""

import time
import requests
import pandas as pd
from tqdm import tqdm

from src.utils import setup_logger
from config import (
    API_KEY, LLM_URL, MODEL_NAME, SPREADING_RESULTS_FILE, 
    FINAL_RESULTS_FILE, MAX_RETRIES, LIKERT_SCALE_MAX
)

# Setup logger
logger = setup_logger("similarity_evaluator")

def load_spreading_results():
    """
    Load spreading activation results.
    
    Returns:
        pd.DataFrame: DataFrame with spreading activation results
    """
    logger.info(f"Loading spreading activation results from {SPREADING_RESULTS_FILE}")
    try:
        df = pd.read_csv(SPREADING_RESULTS_FILE)
        logger.info(f"Loaded {len(df)} concept pairs with activation values")
        return df
    except Exception as e:
        logger.error(f"Error loading spreading results: {e}")
        raise

def get_likert_score(concept1, concept2, retries=MAX_RETRIES):
    """
    Get Likert similarity score for a concept pair using the LLM.
    
    Args:
        concept1 (str): First concept
        concept2 (str): Second concept
        retries (int): Number of retry attempts
        
    Returns:
        int: Likert score (1-7) or None if failed
    """
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    prompt = f"""
Please rate the semantic relatedness between '{concept1}' and '{concept2}'
on a 1–{LIKERT_SCALE_MAX} Likert scale, where:
1 = No relationship
{LIKERT_SCALE_MAX} = Very strong relationship
Respond with only the number (e.g. 5). No additional text or explanation.
"""
    
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 10
    }

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Evaluating: {concept1} – {concept2} (try {attempt}/{retries})")
            resp = requests.post(LLM_URL, headers=headers, json=data, timeout=30)
            resp.raise_for_status()

            raw = resp.json()["choices"][0]["message"]["content"].strip()
            # Extract the first numeric digit
            digits = ''.join(c for c in raw if c.isdigit())
            score = int(digits[0]) if digits else None

            if score and 1 <= score <= LIKERT_SCALE_MAX:
                logger.info(f"LLM answered: {raw!r} → {score}")
                return score
                
            logger.warning(f"Invalid answer: {raw!r}")
        except Exception as e:
            logger.error(f"Error {attempt} for {concept1}-{concept2}: {e}")
            
        time.sleep(2)  # Rate limiting

    logger.error(f"No valid response for: {concept1} – {concept2}")
    return None

def evaluate_similarity(df_spreading):
    """
    Evaluate similarity for all concept pairs and add to DataFrame.
    
    Args:
        df_spreading (pd.DataFrame): DataFrame with spreading activation results
        
    Returns:
        pd.DataFrame: DataFrame with spreading activation and similarity results
    """
    logger.info("Starting similarity evaluation")
    
    # Copy DataFrame to avoid modifying the original
    df = df_spreading.copy()
    df["LikertScore"] = None
    
    total = len(df)
    for idx, row in tqdm(df.iterrows(), total=total, desc="Evaluating similarity"):
        concept1 = row["Concept1"]
        concept2 = row["Concept2"]
        
        likert_score = get_likert_score(concept1, concept2)
        df.at[idx, "LikertScore"] = likert_score
        
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            # Save intermediate results
            temp_file = FINAL_RESULTS_FILE.with_name(f"temp_{FINAL_RESULTS_FILE.name}")
            df.to_csv(temp_file, index=False)
            logger.info(f"Progress: {idx+1}/{total} pairs ({(idx+1)/total*100:.1f}%)")
        
        time.sleep(2)  # Rate limiting
    
    # Remove pairs without valid scores
    df_clean = df.dropna(subset=["LikertScore"])
    logger.info(f"Completed with {len(df_clean)}/{total} valid pairs")
    
    # Save final results
    df_clean.to_csv(FINAL_RESULTS_FILE, index=False)
    logger.info(f"Results saved to {FINAL_RESULTS_FILE}")
    
    return df_clean

def main():
    """
    Main function to run the similarity evaluation.
    """
    logger.info("Starting similarity evaluation process")
    
    # Load spreading activation results
    df_spreading = load_spreading_results()
    
    # Evaluate similarity
    df_final = evaluate_similarity(df_spreading)
    
    logger.info("Similarity evaluation complete")
    logger.info(f"Final dataset: {len(df_final)} concept pairs with activation and similarity scores")

if __name__ == "__main__":
    main()
