import pandas as pd
import seaborn as sns
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

@torch.no_grad()
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


@torch.no_grad()
def plot_token_heatmap_by_occurrence(token_id, token_data, tokenizer, output_dir):

    plt.clf()

    token_occurrences = len(token_data['layer_embeddings'])
    # Calculate correlation matrix between layers
    # oX layer number
    # oY token occurrence
    # heatmap - cosine similarity between current embedding and mean embedding value

    layer_data = {}
    for i in range(token_occurrences):
        layer_data[f"Occurrence_{i}"] = []

    all_occurrences = []
    for layer_embedding_occurrence in token_data['layer_embeddings']:
        layer_embedding_occurrence_t = torch.vstack(layer_embedding_occurrence)
        all_occurrences.append(layer_embedding_occurrence_t)

    # [ n_occurrences, n_layers, hidden_size ]
    all_occurrences_t = torch.vstack([ x.unsqueeze(0) for x in all_occurrences ])
    # [ 1, n_layers, hidden_size ]
    all_occurrences_mean_t = all_occurrences_t.mean(dim=0, keepdim=True)
    distances = torch.nn.functional.cosine_similarity(all_occurrences_t, all_occurrences_mean_t, dim=2)
    # [ n_occurrences, n_layers ]
    # distances = distances.permute(1, 0)

    df = pd.DataFrame(distances.float().detach().cpu().numpy())
    corr_matrix = df.corr(method='spearman')

    # Plot correlation heatmap
    plt.figure(figsize=(24, 20))
    # sns.heatmap(distances.float().detach().cpu().numpy(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title(f"Spearman Correlation Distances Between Occurrences")
    plt.tight_layout()
    file_name = f"occurrence_correlation_with_mean_layer_embedding_{token_id}_{metric}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close()

    print(f"saved to {file_path}")

    # [ n_occurrences, n_layers, hidden_size ]
    # all_occurrences_t
    metrics = {
        "mean": [],
        "std": [],
        "max": [],
        "median": [],
    }

    for layer_i in range(all_occurrences_t.shape[1]):
        plt.clf()
        plt.figure(figsize=(24, 20))

        # [ n_occurrences, hidden_size ]
        current_layer_embeddings = all_occurrences_t[:, layer_i, :]

        normalized_layer_embeddings = torch.nn.functional.normalize(current_layer_embeddings, p=2, dim=1)

        # pairwise cosine similarity
        pairwise_cosine_similarity = normalized_layer_embeddings @ normalized_layer_embeddings.t()

        # [ n_occurrences, n_occurrences ]
        distances = 1 - pairwise_cosine_similarity

        metrics["mean"].append(distances.mean().item())
        metrics["std"].append(distances.std().item())
        metrics["max"].append(distances.max().item())
        metrics["median"].append(distances.median().item())

    plt.clf()
    plt.figure(figsize=(24, 20))
    plt.plot(metrics["mean"], label="mean")
    # plt.plot(metrics["std"], label="std")
    plt.plot(metrics["max"], label="max")
    plt.plot(metrics["median"], label="median")
    plt.title(f"Tokens Metrics Distance Between Occurrences")
    plt.legend()
    metric_file_name = f"metrics_{token_id}.png"
    metric_file_path = os.path.join(output_dir, metric_file_name)
    plt.savefig(metric_file_path)
    plt.close()
    print(f"saved to {metric_file_path}")


    # =====
    # Per token layer difference distances
    

    return corr_matrix


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
    parser.add_argument("--embeddings_file", type=str, required=False, help="Path to the token embeddings file (.pt)")
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

    token_counts = {token_id: data["count"] for token_id, data in per_token_heatmaps.items()}
    most_common_tokens = sorted(token_counts.keys(), key=lambda x: token_counts[x], reverse=True)[:args.visualize_tokens]


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
        for token_id in tqdm(most_common_tokens, desc=f"Plotting individual token visualizations ({metric})"):
            plot_token_heatmap(token_id, per_token_heatmaps[token_id], tokenizer, output_dir, metric)

    # =====

    if args.embeddings_file:
        per_token_embeddings = torch.load(args.embeddings_file)

        for token_id in tqdm(most_common_tokens, desc=f"plot_token_heatmap_by_occurrence"):
            plot_token_heatmap_by_occurrence(token_id, per_token_embeddings[token_id], tokenizer, output_dir)

    logger.info(f"All visualizations saved to {output_dir}")