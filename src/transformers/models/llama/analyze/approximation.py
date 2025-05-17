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

from transformers.models.llama.analyze.embeddings_changes_diff_correlation import pairwise_cosine_similarity

def plot_heatmap(data, file_path, title, cmap='coolwarm', vmin=None, vmax=None, fmt=".2f"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(data, annot=True, cmap=cmap, vmax=vmax, vmin=vmin, fmt=fmt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path)
    print(f"saved {file_path}")
    plt.close()


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
    # parser.add_argument("--analyze_outliers_between_occurrences", type=int, default=0, help="Analyze outliers overlap between different occurrences")

    args = parser.parse_args()
    # analyze_outliers_between_occurrences = bool(args.analyze_outliers_between_occurrences)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load heatmap data
    logger.info(f"Loading embeddings from {args.embeddings_file}")
    per_token_embeddings = torch.load(args.embeddings_file, map_location=torch.device('cpu'))

    most_common_tokens = sorted(per_token_embeddings.items(), key=lambda x: x[1]["count"], reverse=False)
    most_common_tokens = [ x[0] for x in most_common_tokens if x[1]["count"] >= args.min_occurrencies ]
    most_common_tokens = most_common_tokens[:args.tok_k_tokens]

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

    for token_id in most_common_tokens:
        token_str = tokenizer.decode(token_id)
        print(f"token: {token_id} ({token_str})")

        token_data = per_token_embeddings[token_id]

        # [ num samples, num layers, embedding ]
        random_token_embeddings = random.sample(
            token_data["layer_embeddings"],
            min(args.max_process_occurrences, len(token_data["layer_embeddings"]))
        )

        # Calculate correlation matrix between layers
        layer_data = {}
        for i in range(num_layers-1):
            layer_data[f"Layer_{i}"] = []

        # Store all differences for analysis
        all_diffs = []
        layer_diffs = {i: [] for i in range(num_layers-1)}

        num_occurrences = len(random_token_embeddings)

        overall_additive_l1_dist = []
        overall_additive_cosine_sim = []
        overall_noop_l1_dist = []
        overall_noop_cosine_sim = []
        overall_scale_l1_dist = []
        overall_scale_cosine_sim = []

        # Create temporary directory for storing frames
        with tempfile.TemporaryDirectory() as temp_dir:
            cosine_frames = []
            l1_frames = []

            occurence_frames = []

            for layer_idx in range(num_layers - args.num_hop_layers):

                # [ num_occurrences, hidden_size ]
                hs_is = torch.stack([ x[layer_idx] for x in random_token_embeddings]).float()
                hs_js = torch.stack([ x[layer_idx + args.num_hop_layers] for x in random_token_embeddings]).float()


                #
                # Noop
                #
                # [num_occurrences, num_occurrences]
                noop_cosine_sim = pairwise_cosine_similarity(hs_js, hs_is)
                noop_l1_dist = torch.cdist(hs_js, hs_is, p=1)

                # plot heatmaps
                # file_name = f"noop_cosine_sim_layer_{token_id}_{layer_idx}.png"
                # file_path = os.path.join(plots_dir, file_name)
                # plot_heatmap(noop_cosine_sim, file_path, f"Cosine Similarity Between Layers Differences. Noop. Layer {layer_idx}. Token {token_id} ({token_str})")

                # file_name = f"noop_l1_dist_layer_{token_id}_{layer_idx}.png"
                # file_path = os.path.join(plots_dir, file_name)
                # plot_heatmap(noop_l1_dist, file_path, f"L1 Distance Between Layers Differences. Noop. Layer {layer_idx}. Token {token_id} ({token_str})", vmax=5000)


                #
                # Mean additive approximation
                #
                diff_mean_approx = (hs_js - hs_is).mean(dim=0)
                hs_js_hat = hs_is + diff_mean_approx

                # [num_occurrences, num_occurrences]
                cosine_sim = pairwise_cosine_similarity(hs_js_hat, hs_js)
                l1_dist = torch.cdist(hs_js_hat, hs_js, p=1)

                # plot heatmaps
                # file_name = f"cosine_sim_layer_{token_id}_{layer_idx}.png"
                # file_path = os.path.join(plots_dir, file_name)
                # plot_heatmap(cosine_sim, file_path, f"Cosine Similarity Between Layers Differences. Layer {layer_idx}. Token {token_id} ({token_str})")

                # file_name = f"l1_dist_layer_{token_id}_{layer_idx}.png"
                # file_path = os.path.join(plots_dir, file_name)
                # plot_heatmap(l1_dist, file_path, f"L1 Distance Between Layers Differences. Layer {layer_idx}. Token {token_id} ({token_str})", vmax=5000)

                #
                # Scale multiplicative approximation
                #
                scale_approx =  ((hs_js - hs_is) / hs_is).mean(dim=0).nan_to_num(0, 0, 0)
                hs_js_hat = hs_is + (hs_is * scale_approx)

                # [num_occurrences, num_occurrences]
                scale_cosine_sim = pairwise_cosine_similarity(hs_js_hat, hs_js)
                scale_l1_dist = torch.cdist(hs_js_hat, hs_js, p=1)

                # plot heatmaps
                # file_name = f"cosine_sim_layer_{token_id}_{layer_idx}_scale_multiplicative.png"
                # file_path = os.path.join(plots_dir, file_name)
                # plot_heatmap(scale_cosine_sim, file_path, f"Cosine Similarity Between Layers Differences. Scale Multiplicative Approximation. Layer {layer_idx}. Token {token_id} ({token_str})")

                # file_name = f"l1_dist_layer_{token_id}_{layer_idx}_scale_multiplicative.png"
                # file_path = os.path.join(plots_dir, file_name)
                # plot_heatmap(scale_l1_dist, file_path, f"L1 Distance Between Layers Differences. Scale Multiplicative Approximation. Layer {layer_idx}. Token {token_id} ({token_str})", vmax=5000)

                # ================================
                # Overall
                # ================================

                overall_additive_l1_dist.append(l1_dist.mean(-1))
                overall_additive_cosine_sim.append(cosine_sim.mean(-1))
                overall_noop_l1_dist.append(noop_l1_dist.mean(-1))
                overall_noop_cosine_sim.append(noop_cosine_sim.mean(-1))
                overall_scale_l1_dist.append(scale_l1_dist.mean(-1))
                overall_scale_cosine_sim.append(scale_cosine_sim.mean(-1))

        # ================================
        # Plot overall
        # ================================
        file_path = os.path.join(plots_dir, f"overall_additive_l1_dist_layer_{token_id}.png")
        plot_heatmap(overall_additive_l1_dist, file_path, f"Overall L1 Distance. Additive Approximation. Token {token_id} ({token_str})", vmax=5000)
        file_path = os.path.join(plots_dir, f"overall_additive_cosine_sim_layer_{token_id}.png")
        plot_heatmap(overall_additive_cosine_sim, file_path, f"Overall Cosine Similarity. Additive Approximation. Token {token_id} ({token_str})", vmin=0, vmax=2)
        file_path = os.path.join(plots_dir, f"overall_noop_l1_dist_layer_{token_id}.png")
        plot_heatmap(overall_noop_l1_dist, file_path, f"Overall L1 Distance. Noop. Token {token_id} ({token_str})", vmax=5000)
        file_path = os.path.join(plots_dir, f"overall_noop_cosine_sim_layer_{token_id}.png")
        plot_heatmap(overall_noop_cosine_sim, file_path, f"Overall Cosine Similarity. Noop. Token {token_id} ({token_str})", vmin=0, vmax=2)
        file_path = os.path.join(plots_dir, f"overall_scale_l1_dist_layer_{token_id}.png")
        plot_heatmap(overall_scale_l1_dist, file_path, f"Overall L1 Distance. Scale Multiplicative Approximation. Token {token_id} ({token_str})", vmax=5000)
        file_path = os.path.join(plots_dir, f"overall_scale_cosine_sim_layer_{token_id}.png")
        plot_heatmap(overall_scale_cosine_sim, file_path, f"Overall Cosine Similarity. Scale Multiplicative Approximation. Token {token_id} ({token_str})", vmin=0, vmax=2)
