import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from collections import Counter
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def plot_token_heatmap(token_id, token_data, tokenizer, output_dir, metric="cos"):
    """Plot heatmap for a specific token."""
    token_str = tokenizer.decode([token_id])
    plt.figure(figsize=(10, 6))

    mean_data = token_data[f"mean_{metric}"].cpu().numpy()
    num_layers = len(mean_data)

    plt.bar(range(num_layers), mean_data)
    plt.xlabel("Layer")
    plt.ylabel(f"{metric.upper()} Distance")
    plt.title(f"Token '{token_str}' (ID: {token_id}, Frequency: {token_data['count']}) - Mean {metric.upper()} Distance Across Layers")

    # Add error bars if standard deviation is available
    if f"std_{metric}" in token_data:
        std_data = token_data[f"std_{metric}"].cpu().numpy()
        plt.errorbar(range(num_layers), mean_data, yerr=std_data, fmt='none', capsize=5, color='red', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"token_{token_id}_{metric}.png"))
    plt.close()

def plot_token_frequency_vs_change(per_token_heatmaps, tokenizer, output_dir, metric="cos"):
    """Plot relationship between token frequency and mean change."""
    token_frequencies = [data["count"] for _, data in per_token_heatmaps.items()]

    # Calculate average change across all layers for each token
    token_changes = []
    for _, data in per_token_heatmaps.items():
        mean_changes = data[f"mean_{metric}"].cpu().numpy()
        token_changes.append(np.mean(mean_changes))

    plt.figure(figsize=(12, 8))
    plt.scatter(token_frequencies, token_changes, alpha=0.5)
    plt.xscale('log')
    plt.xlabel("Token Frequency (log scale)")
    plt.ylabel(f"Average {metric.upper()} Distance")
    plt.title(f"Token Frequency vs. Average Embedding Change ({metric.upper()} distance)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"frequency_vs_change_{metric}.png"))
    plt.close()

def plot_heatmap_by_layer(per_token_heatmaps, tokenizer, output_dir, top_n=100, metric="cos"):
    """Plot heatmap of top-N most frequent tokens across layers."""
    # Get the most frequent tokens
    token_counts = {token_id: data["count"] for token_id, data in per_token_heatmaps.items()}
    most_common_tokens = sorted(token_counts.keys(), key=lambda x: token_counts[x], reverse=True)[:top_n]

    # Create matrix of layer-wise distances for the most common tokens
    num_layers = len(next(iter(per_token_heatmaps.values()))[f"mean_{metric}"])
    heatmap_data = np.zeros((top_n, num_layers))

    token_texts = []
    for i, token_id in enumerate(most_common_tokens):
        token_data = per_token_heatmaps[token_id]
        heatmap_data[i, :] = token_data[f"mean_{metric}"].cpu().numpy()
        token_texts.append(f"{tokenizer.decode([token_id])} ({token_counts[token_id]})")

    plt.figure(figsize=(15, 20))
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(label=f"{metric.upper()} Distance")
    plt.xlabel("Layer")
    plt.ylabel("Token")
    plt.title(f"Top {top_n} Most Frequent Tokens: {metric.upper()} Distance Across Layers")
    plt.yticks(range(top_n), token_texts, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"top{top_n}_tokens_heatmap_{metric}.png"))
    plt.close()

def plot_average_layer_change(per_token_heatmaps, output_dir, metric="cos"):
    """Plot average change at each layer across all tokens."""
    num_layers = len(next(iter(per_token_heatmaps.values()))[f"mean_{metric}"])
    total_changes = np.zeros(num_layers)
    total_tokens = 0

    for token_id, data in per_token_heatmaps.items():
        count = data["count"]
        mean_changes = data[f"mean_{metric}"].cpu().numpy()
        total_changes += mean_changes * count
        total_tokens += count

    avg_changes = total_changes / total_tokens

    plt.figure(figsize=(10, 6))
    plt.bar(range(num_layers), avg_changes)
    plt.xlabel("Layer")
    plt.ylabel(f"Average {metric.upper()} Distance")
    plt.title(f"Average Token Embedding Change by Layer ({metric.upper()} distance)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"avg_layer_change_{metric}.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap_file", type=str, required=True, help="Path to the token heatmaps file (.pt)")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--top_n", type=int, default=100, help="Number of top tokens to visualize in heatmap")
    parser.add_argument("--visualize_tokens", type=int, default=20, help="Number of most frequent tokens to individually visualize")

    args = parser.parse_args()

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load heatmap data
    logger.info(f"Loading token heatmaps from {args.heatmap_file}")
    per_token_heatmaps = torch.load(args.heatmap_file)

    # Create output directory
    model_name = os.path.basename(args.heatmap_file).replace("token_heatmaps_", "").replace(".pt", "")
    output_dir = os.path.join("results", "token_viz", model_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Analyzing data for {len(per_token_heatmaps)} unique tokens")

    # Generate visualizations for both metrics
    for metric in ["cos", "l1"]:
        logger.info(f"Generating visualizations for {metric} distance")

        # Plot frequency vs. change
        plot_token_frequency_vs_change(per_token_heatmaps, tokenizer, output_dir, metric)

        # Plot heatmap of top tokens
        plot_heatmap_by_layer(per_token_heatmaps, tokenizer, output_dir, args.top_n, metric)

        # Plot average change by layer
        plot_average_layer_change(per_token_heatmaps, output_dir, metric)

        # Visualize most frequent tokens individually
        token_counts = {token_id: data["count"] for token_id, data in per_token_heatmaps.items()}
        most_common_tokens = sorted(token_counts.keys(), key=lambda x: token_counts[x], reverse=True)[:args.visualize_tokens]

        for token_id in tqdm(most_common_tokens, desc=f"Plotting individual token visualizations ({metric})"):
            plot_token_heatmap(token_id, per_token_heatmaps[token_id], tokenizer, output_dir, metric)

    logger.info(f"All visualizations saved to {output_dir}")