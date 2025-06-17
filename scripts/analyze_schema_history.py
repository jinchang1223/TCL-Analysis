import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def calculate_entropy(probs):
    """Calculate entropy given a list of probabilities"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zero probabilities
    return -np.sum(probs * np.log2(probs))

def calculate_rank_shift(before_tokens, after_tokens):
    """Calculate average rank shift for tokens that appear in both lists"""
    before_ranks = {token["token_id"]: i for i, token in enumerate(before_tokens)}
    after_ranks = {token["token_id"]: i for i, token in enumerate(after_tokens)}
    
    common_tokens = set(before_ranks.keys()) & set(after_ranks.keys())
    if not common_tokens:
        return 0
    
    rank_shifts = []
    for token_id in common_tokens:
        before_rank = before_ranks[token_id]
        after_rank = after_ranks[token_id]
        rank_shifts.append(abs(before_rank - after_rank))
    
    return np.mean(rank_shifts)

def analyze_history_file(history_file):
    """Analyze a single history file and return statistics"""
    with open(history_file, 'r') as f:
        record = json.load(f)
    
    # Handle both old and new format
    if isinstance(record, list):
        history = record
        file_name = os.path.basename(history_file)
    else:
        history = record.get("decoding_history", [])
        file_name = record.get("schema", os.path.basename(history_file))
    
    # Initialize statistics
    stats = {
        "file_name": file_name,
        "total_steps": len(history),
        "total_interventions": 0,
        "intervention_rate": 0,
        "mean_accepted_ratio": 0,
        "mean_top1_prob": 0,
        "mean_top1_prob_masked": 0,
        "min_accepted_ratio": float('inf'),
        "max_accepted_ratio": 0,
        "mean_rank_shift": 0
    }
    
    # Collect data for each step
    accepted_ratios = []
    top1_probs = []
    top1_probs_masked = []
    rank_shifts = []
    
    for steps in history:
        step = steps[0]
        # Skip if step is not in expected format
        if not isinstance(step, dict) or "metrics" not in step:
            continue
            
        # Count interventions
        if step["metrics"].get("is_intervened", False):
            stats["total_interventions"] += 1
        
        # Collect accepted ratio
        accepted_ratio = step["metrics"].get("acceptance_ratio", 0)
        accepted_ratios.append(accepted_ratio)
        stats["min_accepted_ratio"] = min(stats["min_accepted_ratio"], accepted_ratio)
        stats["max_accepted_ratio"] = max(stats["max_accepted_ratio"], accepted_ratio)
        
        # Collect top-1 probabilities
        top1_probs.append(step["metrics"].get("top_token_prob", 0))
        if step.get("top_20_after_constraint"):
            top1_probs_masked.append(step["top_20_after_constraint"][0].get("raw_likelihood", 0))
        
        # Calculate rank shift
        if step.get("top_20_before_constraint") and step.get("top_20_after_constraint"):
            rank_shift = calculate_rank_shift(
                step["top_20_before_constraint"],
                step["top_20_after_constraint"]
            )
            rank_shifts.append(rank_shift)
    
    # Calculate final statistics
    if stats["total_steps"] > 0:
        stats["intervention_rate"] = stats["total_interventions"] / stats["total_steps"]
        stats["mean_accepted_ratio"] = np.mean(accepted_ratios) if accepted_ratios else 0
        stats["mean_top1_prob"] = np.mean(top1_probs) if top1_probs else 0
        stats["mean_top1_prob_masked"] = np.mean(top1_probs_masked) if top1_probs_masked else 0
        stats["mean_rank_shift"] = np.mean(rank_shifts) if rank_shifts else 0
    
    return stats

def main():
    # Define paths
    HISTORY_PATH = "tries/Glaiveai2K"
    OUTPUT_FILE = "tries/history_statistics_glaiveai2k.json"
    
    # Get all history files
    history_files = list(Path(HISTORY_PATH).glob("*.json"))
    
    # Analyze each file
    all_stats = []
    for history_file in tqdm(history_files, desc="Analyzing history files"):
        try:
            stats = analyze_history_file(history_file)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error analyzing {history_file.name}: {str(e)}")
            continue
    
    # Save all statistics
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_FILE}")
    

if __name__ == "__main__":
    main() 