#!/usr/bin/env python3
"""
Compare llama31_8b and mistral7b models based on experiment results.
Generates detailed comparison metrics for model selection.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def load_model_results(csv_path: str) -> pd.DataFrame:
    """Load model results from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive metrics from model results."""
    return {
        "avg_latency": df["latency_sec"].mean(),
        "median_latency": df["latency_sec"].median(),
        "min_latency": df["latency_sec"].min(),
        "max_latency": df["latency_sec"].max(),
        "std_latency": df["latency_sec"].std(),
        "avg_words": df["answer_words"].mean(),
        "median_words": df["answer_words"].median(),
        "min_words": df["answer_words"].min(),
        "max_words": df["answer_words"].max(),
        "total_questions": len(df),
        "errors": 0  # Assuming no errors based on experiment cards
    }


def compare_models(llama_path: str, mistral_path: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
    """Compare two models and return metrics and comparison."""
    llama_df = load_model_results(llama_path)
    mistral_df = load_model_results(mistral_path)
    
    llama_metrics = calculate_metrics(llama_df)
    mistral_metrics = calculate_metrics(mistral_df)
    
    # Calculate improvements
    comparison = {
        "latency_improvement_pct": ((mistral_metrics["avg_latency"] - llama_metrics["avg_latency"]) / mistral_metrics["avg_latency"]) * 100,
        "median_latency_improvement_pct": ((mistral_metrics["median_latency"] - llama_metrics["median_latency"]) / mistral_metrics["median_latency"]) * 100,
        "word_count_reduction_pct": ((mistral_metrics["avg_words"] - llama_metrics["avg_words"]) / mistral_metrics["avg_words"]) * 100,
        "speedup_factor": mistral_metrics["avg_latency"] / llama_metrics["avg_latency"],
    }
    
    return llama_metrics, mistral_metrics, comparison


def print_comparison(llama_metrics: Dict[str, float], 
                     mistral_metrics: Dict[str, float], 
                     comparison: Dict[str, str]):
    """Print formatted comparison table."""
    print("\n" + "="*80)
    print("MODEL COMPARISON: llama31_8b vs mistral7b")
    print("="*80)
    
    print("\n📊 Performance Metrics:")
    print("-" * 80)
    print(f"{'Metric':<30} {'llama31_8b':<20} {'mistral7b':<20} {'Winner':<10}")
    print("-" * 80)
    
    # Latency comparison
    llama_lat = llama_metrics["avg_latency"]
    mistral_lat = mistral_metrics["avg_latency"]
    latency_winner = "llama31_8b" if llama_lat < mistral_lat else "mistral7b"
    print(f"{'Average Latency (s)':<30} {llama_lat:<20.3f} {mistral_lat:<20.3f} {latency_winner:<10}")
    
    llama_med = llama_metrics["median_latency"]
    mistral_med = mistral_metrics["median_latency"]
    median_winner = "llama31_8b" if llama_med < mistral_med else "mistral7b"
    print(f"{'Median Latency (s)':<30} {llama_med:<20.3f} {mistral_med:<20.3f} {median_winner:<10}")
    
    # Word count comparison (lower is better for concise answers)
    llama_words = llama_metrics["avg_words"]
    mistral_words = mistral_metrics["avg_words"]
    words_winner = "llama31_8b" if llama_words < mistral_words else "mistral7b"
    print(f"{'Average Answer Length (words)':<30} {llama_words:<20.1f} {mistral_words:<20.1f} {words_winner:<10}")
    
    print("-" * 80)
    
    print("\n🚀 Key Improvements (llama31_8b vs mistral7b):")
    print("-" * 80)
    print(f"  • Latency improvement: {comparison['latency_improvement_pct']:.1f}% faster")
    print(f"  • Median latency improvement: {comparison['median_latency_improvement_pct']:.1f}% faster")
    print(f"  • Answer conciseness: {comparison['word_count_reduction_pct']:.1f}% shorter")
    print(f"  • Overall speedup: {comparison['speedup_factor']:.2f}x faster")
    
    print("\n📈 Detailed Statistics:")
    print("-" * 80)
    print(f"\nllama31_8b:")
    print(f"  Latency: {llama_metrics['min_latency']:.3f}s - {llama_metrics['max_latency']:.3f}s (std: {llama_metrics['std_latency']:.3f}s)")
    print(f"  Words: {llama_metrics['min_words']:.0f} - {llama_metrics['max_words']:.0f} (std: {llama_metrics.get('std_words', 0):.1f})")
    
    print(f"\nmistral7b:")
    print(f"  Latency: {mistral_metrics['min_latency']:.3f}s - {mistral_metrics['max_latency']:.3f}s (std: {mistral_metrics['std_latency']:.3f}s)")
    print(f"  Words: {mistral_metrics['min_words']:.0f} - {mistral_metrics['max_words']:.0f} (std: {mistral_metrics.get('std_words', 0):.1f})")
    
    print("\n" + "="*80)
    
    return {
        "llama_metrics": llama_metrics,
        "mistral_metrics": mistral_metrics,
        "comparison": comparison
    }


def main():
    """Main function to compare models."""
    outputs_dir = Path("outputs")
    llama_path = outputs_dir / "llama31_8b_llm_only_run.csv"
    mistral_path = outputs_dir / "mistral7b_llm_only_run.csv"
    
    if not llama_path.exists():
        print(f"Error: {llama_path} not found")
        return
    
    if not mistral_path.exists():
        print(f"Error: {mistral_path} not found")
        return
    
    llama_metrics, mistral_metrics, comparison = compare_models(str(llama_path), str(mistral_path))
    results = print_comparison(llama_metrics, mistral_metrics, comparison)
    
    print("\n✅ Comparison complete!")
    return results


if __name__ == "__main__":
    main()




