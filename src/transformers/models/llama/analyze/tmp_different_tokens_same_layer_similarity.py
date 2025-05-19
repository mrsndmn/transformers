import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import imageio.v2 as imageio
import tempfile

import random

from transformers.models.llama.analyze.embeddings_change import compute_distances, norm_compute_cosine_distance, compute_l1_distance
from transformers.models.llama.analyze.mean_per_token_embeddings_change import trim_embeddings

from sklearn.metrics.pairwise import cosine_distances


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def pairwise_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет попарные косинусные расстояния между строками x и y.
    Возвращает матрицу расстояний shape (x.shape[0], y.shape[0]).
    """
    # Нормализация по строкам
    x_norm = x / (x.norm(dim=1, keepdim=True)+1e-6)
    y_norm = y / (y.norm(dim=1, keepdim=True)+1e-6)

    # Косинусное сходство: матричное произведение
    cosine_sim = x_norm @ y_norm.T

    return cosine_sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_file", type=str, required=True, help="Path to the embeddings file (.pt)")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--tok_k_tokens", type=int, default=10)
    parser.add_argument("--num_hop_layers", type=int, default=1)
    parser.add_argument("--min_occurrencies", type=int, default=20)
    parser.add_argument("--max_process_occurrences", type=int, default=100)
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the file name")
    parser.add_argument("--output_dir", type=str, default="results/most_common_tokens_occurrences", help="Output directory")
    parser.add_argument("--least_common", type=bool, default=False, help="Least common tokens")
    parser.add_argument("--trim_quantile", type=float, default=0.0, help="Trim quantile")
    parser.add_argument("--trim_mode", type=str, default="both", help="Trim mode")
    parser.add_argument("--normalize_embeddings", type=bool, default=False, help="Normalize embeddings")
    parser.add_argument("--plot_per_layer_distances", type=int, default=0, help="Plot per layer distances")
    parser.add_argument("--plot_per_occurence_distances", type=int, default=0, help="Plot per occurence distances")
    parser.add_argument("--diff_analyse", type=int, default=0, help="Analyse diffs")
    parser.add_argument("--analyze_outliers_indices_and_per_layer_overlap", type=int, default=0, help="Analyze outliers indices and per layer overlap")
    parser.add_argument("--analyze_outliers_indices_and_per_layer_overlap_quantile", type=float, default=0.1, help="Analyze outliers indices and per layer overlap quantile")
    parser.add_argument("--analyze_outliers_between_occurrences", type=int, default=0, help="Analyze outliers overlap between different occurrences")

    args = parser.parse_args()
    plot_per_layer_distances = bool(args.plot_per_layer_distances)
    plot_per_occurence_distances = bool(args.plot_per_occurence_distances)
    diff_analyse = bool(args.diff_analyse)
    analyze_outliers_indices_and_per_layer_overlap = bool(args.analyze_outliers_indices_and_per_layer_overlap)
    analyze_outliers_between_occurrences = bool(args.analyze_outliers_between_occurrences)
    print("plot_per_layer_distances", plot_per_layer_distances)
    print("plot_per_occurence_distances", plot_per_occurence_distances)
    print("diff_analyse", diff_analyse)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load heatmap data
    logger.info(f"Loading embeddings from {args.embeddings_file}")
    per_token_embeddings = torch.load(args.embeddings_file, map_location=torch.device('cpu'))

    most_common_tokens = sorted(per_token_embeddings.items(), key=lambda x: x[1]["count"], reverse=not args.least_common)
    most_common_tokens = [ x[0] for x in most_common_tokens ]
    # most_common_tokens = [ x[0] for x in most_common_tokens if x[1]["count"] >= args.min_occurrencies ]
    most_common_tokens = random.sample(most_common_tokens, args.tok_k_tokens)
    # most_common_tokens = most_common_tokens[:args.tok_k_tokens]

    tokens_chars = list(map(tokenizer.decode, most_common_tokens))
    print(f"most_common_tokens: {most_common_tokens}: {tokens_chars}")

    token_occurrences = [ per_token_embeddings[token_id]["count"] for token_id in most_common_tokens]

    print(f"most_common_tokens:", "\t".join(tokens_chars))
    print(f"token_occurrences:", "\t".join(map(str, token_occurrences)))

    num_layers = len(per_token_embeddings[most_common_tokens[0]]["layer_embeddings"][0])
    print(f"num_layers: {num_layers}")

    metric = 'cos'
    suffix = f"_{args.suffix}"

    # Create output directory for plots
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Dictionary to store outliers overlap data for plotting
    outliers_overlap_data = {}

    layer_id = 22

    embeddings = []

    for token_id in most_common_tokens:
        token_data = per_token_embeddings[token_id]
        embeddings.append(token_data["layer_embeddings"][0][layer_id])

    print(f"embeddings: {len(embeddings)}")

    embeddings_t = torch.stack(embeddings).to(torch.float64)

    cosine_sim =pairwise_cosine_similarity(embeddings_t, embeddings_t)

    # plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cosine_sim, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title(f"Pairwise Cosine Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"different_tokens_same_layer_similarity_{token_id}{suffix}.png"))
    print(f"saved to {os.path.join(plots_dir, f'different_tokens_same_layer_similarity_{token_id}{suffix}.png')}")
    plt.close()
