import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def analyze_embedding_distances_distribution(per_token_heatmaps, output_dir, metric="cos"):
    """Analyze the distribution of embedding distances."""
    all_distances = []

    for token_id, data in per_token_heatmaps.items():
        if data["count"] > 10:  # Filter tokens with reasonable frequency
            mean_distances = data[f"mean_{metric}"].cpu().numpy()
            all_distances.extend(mean_distances)

    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_distances, bins=50, alpha=0.75)
    plt.xlabel(f"{metric.upper()} Distance")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {metric.upper()} Distances Across All Tokens and Layers")
    plt.axvline(np.mean(all_distances), color='red', linestyle='dashed', linewidth=1,
                label=f'Mean: {np.mean(all_distances):.4f}')
    plt.axvline(np.median(all_distances), color='green', linestyle='dashed', linewidth=1,
                label=f'Median: {np.median(all_distances):.4f}')
    plt.legend()
    plt.tight_layout()
    file_name = f"distance_distribution_{metric}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    print(f"saved to {file_path}")
    plt.close()

    return {
        "mean": np.mean(all_distances),
        "median": np.median(all_distances),
        "std": np.std(all_distances),
        "min": np.min(all_distances),
        "max": np.max(all_distances),
    }

def analyze_token_consistency(per_token_heatmaps, output_dir, metric="cos", suffix="", top_k_tokens=10, tokenizer=None):
    """Analyze the consistency of changes across tokens."""
    num_layers = len(next(iter(per_token_heatmaps.values()))[f"mean_{metric}"])

    print(f"num_layers: {num_layers}")

    if suffix != "":
        suffix = "_" + suffix

    # Calculate correlation matrix between layers
    layer_data = {}
    for i in range(num_layers):
        layer_data[f"Layer_{i}"] = []

    sorted_by_counts_tokens = sorted(per_token_heatmaps.items(), key=lambda x: x[1]["count"], reverse=True)

    for i, (token_id, data) in enumerate(sorted_by_counts_tokens):
        if data["count"] < 10:  # Only consider tokens with sufficient occurrences
            break

        distances = data[f"mean_{metric}"].cpu().numpy()
        for layer_idx in range(num_layers):
            layer_data[f"Layer_{layer_idx}"].append(distances[layer_idx])

        # if top_k_tokens > 0 and i < top_k_tokens:
        #     current_token_layer_data = {}
        #     for layer_idx in range(num_layers):
        #         current_token_layer_data[f"Layer_{layer_idx}"] = [distances[layer_idx]]

        #     df = pd.DataFrame(current_token_layer_data)
        #     corr_matrix = df.corr(method='spearman')
        #     plt.figure(figsize=(12, 10))
        #     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
        #     token_str = ""
        #     if tokenizer is not None:
        #         token_str = tokenizer.decode([token_id])

        #     plt.title(f"Spearman Correlation of {metric.upper()} Distances Between Layers for Token {token_id} [{token_str}]")
        #     plt.tight_layout()
        #     file_name = f"layer_correlation_{metric}_token_{token_id}{suffix}.png"
        #     file_path = os.path.join(output_dir, file_name)
        #     plt.savefig(file_path)
        #     print(f"saved to {file_path}")
        #     plt.close()


    df = pd.DataFrame(layer_data)
    corr_matrix = df.corr(method='spearman')
    breakpoint()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title(f"Spearman Correlation of {metric.upper()} Distances Between Layers")
    plt.tight_layout()
    file_name = f"layer_correlation_{metric}{suffix}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    print(f"saved to {file_path}")
    plt.close()

    return corr_matrix

def analyze_frequency_impact(per_token_heatmaps, output_dir, metric="cos"):
    """Analyze how token frequency impacts embedding changes."""
    token_freqs = []
    avg_distances = []
    std_distances = []

    for token_id, data in per_token_heatmaps.items():
        freq = data["count"]
        distances = data[f"mean_{metric}"].cpu().numpy()

        token_freqs.append(freq)
        avg_distances.append(np.mean(distances))
        std_distances.append(np.std(distances))

    # Calculate correlation
    corr, p_value = spearmanr(token_freqs, avg_distances)

    # Bin tokens by frequency for box plots
    df = pd.DataFrame({
        'Frequency': token_freqs,
        'Average Distance': avg_distances,
        'Std Distance': std_distances
    })

    # Create frequency bins (log scale)
    bins = [1, 10, 100, 1000, 10000, float('inf')]
    labels = ['1-10', '11-100', '101-1000', '1001-10000', '10000+']
    df['Frequency Bin'] = pd.cut(df['Frequency'], bins=bins, labels=labels, right=False)

    # Box plot of distances by frequency bin
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Frequency Bin', y='Average Distance', data=df)
    plt.title(f"Average {metric.upper()} Distance by Token Frequency\nSpearman Correlation: {corr:.4f} (p={p_value:.4f})")
    plt.xlabel("Token Frequency")
    plt.ylabel(f"Average {metric.upper()} Distance")
    plt.tight_layout()
    file_name = f"frequency_boxplot_{metric}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    print(f"saved to {file_path}")
    plt.close()

    return {
        "correlation": corr,
        "p_value": p_value,
        "binned_data": df
    }

def identify_extreme_tokens(per_token_heatmaps, tokenizer, output_dir, metric="cos", n=20):
    """Identify tokens with extreme embedding changes."""
    token_stats = []

    for token_id, data in per_token_heatmaps.items():
        if data["count"] < 5:  # Skip tokens with very few occurrences
            continue

        distances = data[f"mean_{metric}"].cpu().numpy()
        token_str = tokenizer.decode([token_id])

        token_stats.append({
            "token_id": token_id,
            "token_str": token_str,
            "frequency": data["count"],
            "avg_distance": np.mean(distances),
            "max_distance": np.max(distances),
            "min_distance": np.min(distances),
            "std_distance": np.std(distances),
        })

    # Sort by average distance (highest first)
    token_stats_sorted = sorted(token_stats, key=lambda x: x["avg_distance"], reverse=True)

    # Extract highest and lowest tokens
    highest_tokens = token_stats_sorted[:n]
    lowest_tokens = token_stats_sorted[-n:]

    # Create a table for highest tokens
    highest_df = pd.DataFrame(highest_tokens)
    highest_df.to_csv(os.path.join(output_dir, f"highest_{metric}_tokens.csv"), index=False)

    # Create a table for lowest tokens
    lowest_df = pd.DataFrame(lowest_tokens)
    lowest_df.to_csv(os.path.join(output_dir, f"lowest_{metric}_tokens.csv"), index=False)

    # Plot bar charts
    plt.figure(figsize=(12, 8))
    plt.barh([f"{t['token_str']} ({t['frequency']})" for t in highest_tokens[:20]],
             [t["avg_distance"] for t in highest_tokens[:20]])
    plt.xlabel(f"Average {metric.upper()} Distance")
    plt.title(f"Top 20 Tokens with Highest {metric.upper()} Distances")
    plt.gca().invert_yaxis()  # To have highest at the top
    plt.tight_layout()
    file_name = f"highest_{metric}_tokens.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    print(f"saved to {file_path}")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.barh([f"{t['token_str']} ({t['frequency']})" for t in lowest_tokens[:20]],
             [t["avg_distance"] for t in lowest_tokens[:20]])
    plt.xlabel(f"Average {metric.upper()} Distance")
    plt.title(f"Top 20 Tokens with Lowest {metric.upper()} Distances")
    plt.gca().invert_yaxis()  # To have lowest at the top
    plt.tight_layout()
    file_name = f"lowest_{metric}_tokens.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    print(f"saved to {file_path}")
    plt.close()

    return {
        "highest": highest_tokens,
        "lowest": lowest_tokens
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap_file", type=str, required=True, help="Path to the token heatmaps file (.pt)")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the file name")

    args = parser.parse_args()

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load heatmap data
    logger.info(f"Loading token heatmaps from {args.heatmap_file}")
    per_token_heatmaps = torch.load(args.heatmap_file)

    # Create output directory
    model_name = os.path.basename(args.heatmap_file).replace("token_heatmaps_", "").replace(".pt", "")
    output_dir = os.path.join("results", "token_stats", model_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Analyzing statistics for {len(per_token_heatmaps)} unique tokens")

    # Generate visualizations for both metrics
    results = {}
    for metric in ["cos", "l1"]:
        logger.info(f"Generating statistics for {metric} distance")

        results[metric] = {}

        # Analyze distribution
        results[metric]["distribution"] = analyze_embedding_distances_distribution(per_token_heatmaps, output_dir, metric)

        # Analyze layer correlations
        results[metric]["correlations"] = analyze_token_consistency(per_token_heatmaps, output_dir, metric, suffix=args.suffix, tokenizer=tokenizer)

        # Analyze frequency impact
        results[metric]["frequency_impact"] = analyze_frequency_impact(per_token_heatmaps, output_dir, metric)

        # Identify extreme tokens
        results[metric]["extreme_tokens"] = identify_extreme_tokens(per_token_heatmaps, tokenizer, output_dir, metric)

    # Create summary report
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Token Embedding Change Analysis Summary\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total tokens analyzed: {len(per_token_heatmaps)}\n\n")

        for metric in ["cos", "l1"]:
            f.write(f"--- {metric.upper()} Distance Statistics ---\n")
            f.write(f"Mean distance: {results[metric]['distribution']['mean']:.6f}\n")
            f.write(f"Median distance: {results[metric]['distribution']['median']:.6f}\n")
            f.write(f"Standard deviation: {results[metric]['distribution']['std']:.6f}\n")
            f.write(f"Min distance: {results[metric]['distribution']['min']:.6f}\n")
            f.write(f"Max distance: {results[metric]['distribution']['max']:.6f}\n")
            f.write(f"Frequency-Distance correlation: {results[metric]['frequency_impact']['correlation']:.6f} ")
            f.write(f"(p={results[metric]['frequency_impact']['p_value']:.6f})\n\n")

            f.write(f"Top 5 tokens with highest {metric.upper()} distances:\n")
            for i, token in enumerate(results[metric]['extreme_tokens']['highest'][:5]):
                f.write(f"{i+1}. '{token['token_str']}' (ID: {token['token_id']}, Freq: {token['frequency']}): {token['avg_distance']:.6f}\n")

            f.write(f"\nTop 5 tokens with lowest {metric.upper()} distances:\n")
            for i, token in enumerate(results[metric]['extreme_tokens']['lowest'][:5]):
                f.write(f"{i+1}. '{token['token_str']}' (ID: {token['token_id']}, Freq: {token['frequency']}): {token['avg_distance']:.6f}\n")

            f.write("\n")

    logger.info(f"All statistics and visualizations saved to {output_dir}")