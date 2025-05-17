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

        # Create temporary directory for storing frames
        with tempfile.TemporaryDirectory() as temp_dir:
            cosine_frames = []
            l1_frames = []

            occurence_frames = []

            for occurence_idx in range(num_occurrences):
                occurence_diffs = []
                for layer_idx in range(num_layers - 1):
                    # hs_i ~ [ hidden_size ]
                    hs_i = random_token_embeddings[occurence_idx][layer_idx]
                    # hs_j ~ [ hidden_size ]
                    hs_j = random_token_embeddings[occurence_idx][layer_idx + args.num_hop_layers]

                    if args.trim_quantile > 0:
                        hs_i, hs_j = trim_embeddings(hs_i, hs_j, args.trim_quantile, trim_mode=args.trim_mode)

                    if args.normalize_embeddings:
                        hs_i = hs_i - hs_i.mean()
                        hs_j = hs_j - hs_j.mean()

                    if diff_analyse:
                        hs_diff = (hs_j - hs_i).float()
                    else:
                        hs_diff = hs_j.float()

                    all_diffs.append(hs_diff)
                    layer_diffs[layer_idx].append(hs_diff)
                    occurence_diffs.append(hs_diff)

                if analyze_outliers_between_occurrences:
                    # For each layer, compute outliers for current occurrence

                    plt.figure(figsize=(12, 8))

                    for layer_idx in range(num_layers):
                        current_embeddings = random_token_embeddings[occurence_idx][layer_idx].float()

                        quantile = args.analyze_outliers_indices_and_per_layer_overlap_quantile
                        lower_bound = torch.quantile(current_embeddings, quantile, dim=-1, keepdim=True)
                        upper_bound = torch.quantile(current_embeddings, 1.0 - quantile, dim=-1, keepdim=True)


                        # Find outliers for current occurrence
                        current_outliers = set(torch.where((current_embeddings < lower_bound) |
                                                         (current_embeddings > upper_bound))[0].numpy().tolist())

                        # Compute overlap with all other occurrences
                        overlaps = []
                        layers_indexes = list(range(occurence_idx+1, num_occurrences))
                        for other_occurence_idx in layers_indexes:

                            other_embeddings = random_token_embeddings[other_occurence_idx][layer_idx].float()
                            other_lower_bound = torch.quantile(other_embeddings, quantile, dim=-1, keepdim=True)
                            other_upper_bound = torch.quantile(other_embeddings, 1.0 - quantile, dim=-1, keepdim=True)

                            other_outliers = set(torch.where((other_embeddings < other_lower_bound) |
                                                            (other_embeddings > other_upper_bound))[0].numpy().tolist())

                            overlap = len(current_outliers.intersection(other_outliers)) / len(current_outliers)
                            overlaps.append(overlap)

                        plt.plot(layers_indexes,
                                overlaps,
                                marker='o',
                                label=f'Layer {layer_idx}')

                    plt.title(f'Outliers Overlap Between Occurrences for Token "{token_str}". Quantile {args.analyze_outliers_indices_and_per_layer_overlap_quantile}')
                    plt.xlabel('Occurrence Index')
                    plt.ylabel('Average Outliers Overlap with Other Occurrences')
                    plt.grid(True)
                    plt.ylim(0, 1)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()

                    file_name = f'occurrence_outliers_overlap_token_occurence_{occurence_idx}_tok_{token_id}_{token_str}_quantile_q{args.analyze_outliers_indices_and_per_layer_overlap_quantile}{suffix}.png'
                    plt.savefig(os.path.join(plots_dir, file_name))
                    print(f"saved to {os.path.join(plots_dir, file_name)}")
                    plt.close()


                if plot_per_occurence_distances:
                    occurence_diffs_t = torch.stack(occurence_diffs).to(torch.float64)

                    occurence_diffs_cosine = pairwise_cosine_similarity(occurence_diffs_t, occurence_diffs_t)
                    occurence_diffs_cosine = occurence_diffs_cosine.numpy()

                    plt.figure(figsize=(12, 10))
                    sns.heatmap(occurence_diffs_cosine, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
                    plt.title(f"Pairwise Cosine Similarity Between Embeddings from different Layers. Occurence {occurence_idx}. Token {token_id} ({token_str})")
                    plt.tight_layout()

                    # Save frame for occurrence GIF
                    occurence_frame_path = os.path.join(temp_dir, f"occurence_frame_{occurence_idx}.png")
                    plt.savefig(occurence_frame_path)
                    occurence_frames.append(imageio.imread(occurence_frame_path))

                    plt.savefig(os.path.join(plots_dir, f"occurence_diffs_{occurence_idx}_{token_id}_{token_str}{suffix}.png"))
                    plt.close()

            if plot_per_layer_distances or analyze_outliers_indices_and_per_layer_overlap:

                prev_outliers = set()
                for layer_idx in range(num_layers-1):
                    current_layer_diffs = layer_diffs[layer_idx]
                    # [ num_occurrences, hidden_size ]
                    layer_diffs_t = torch.stack(current_layer_diffs).to(torch.float64)

                    if analyze_outliers_indices_and_per_layer_overlap:
                        # [ hidden_size ]
                        layer_diffs_t_mean = layer_diffs_t.mean(dim=0)
                        quantile = args.analyze_outliers_indices_and_per_layer_overlap_quantile
                        assert quantile > 0 and quantile < 0.5

                        lower_bound = torch.quantile(layer_diffs_t_mean, quantile, dim=-1, keepdim=True)
                        upper_bound = torch.quantile(layer_diffs_t_mean, 1.0 - quantile, dim=-1, keepdim=True)
                        print(f"Layer {layer_idx} lower_bound {lower_bound} upper_bound {upper_bound}")

                        current_outliers = set(torch.where((layer_diffs_t_mean < lower_bound) | (layer_diffs_t_mean > upper_bound))[0].numpy().tolist())
                        outliers_overlap = prev_outliers.intersection(current_outliers)

                        outliers_overlap_percent = len(outliers_overlap) / len(current_outliers)
                        print(f"Layer {layer_idx} outliers_overlap% {outliers_overlap_percent:.2f}")

                        # Store data for plotting
                        if token_str not in outliers_overlap_data:
                            outliers_overlap_data[token_str] = []
                        outliers_overlap_data[token_str].append(outliers_overlap_percent)

                        prev_outliers = current_outliers

                    if plot_per_layer_distances:
                        # Heatmap of pairwise distances of layers differences
                        print("occurence_diffs_t min max mean", layer_diffs_t.min(), layer_diffs_t.max(), layer_diffs_t.mean())

                        print("occurence_diffs_t", layer_diffs_t.shape)
                        occurence_diffs_cosine = pairwise_cosine_similarity(layer_diffs_t, layer_diffs_t)
                        occurence_diffs_cosine = occurence_diffs_cosine.numpy()

                        # Plot pairwise cosine similarity heatmap
                        plt.figure(figsize=(12, 10))
                        sns.heatmap(occurence_diffs_cosine, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
                        plt.title(f"Pairwise Cosine Similarity Between Layers Differences. Layer {layer_idx}. Token {token_id} ({token_str})")
                        plt.tight_layout()

                        # Save frame for cosine GIF
                        cosine_frame_path = os.path.join(temp_dir, f"cosine_frame_{layer_idx}.png")
                        plt.savefig(cosine_frame_path)
                        cosine_frames.append(imageio.imread(cosine_frame_path))
                        plt.close()

                        layer_diffs_t_norm = layer_diffs_t / layer_diffs_t.norm(2, dim=1, keepdim=True)
                        occurence_diffs_l1 = torch.cdist(layer_diffs_t_norm, layer_diffs_t_norm, p=1)
                        occurence_diffs_l1 = occurence_diffs_l1.numpy()

                        # Plot pairwise l1 similarity heatmap
                        plt.figure(figsize=(12, 10))
                        sns.heatmap(occurence_diffs_l1, annot=True, cmap='coolwarm', vmin=0, vmax=100, fmt=".0f")
                        plt.title(f"Pairwise Normalized L1 Distance Between Layers Differences. Layer {layer_idx}. Token {token_id} ({token_str})")
                        plt.tight_layout()

                        # Save frame for L1 GIF
                        l1_frame_path = os.path.join(temp_dir, f"l1_frame_{layer_idx}.png")
                        plt.savefig(l1_frame_path)
                        l1_frames.append(imageio.imread(l1_frame_path))
                        plt.close()

            # Save GIFs

            if len(cosine_frames) > 0:
                cosine_gif_path = os.path.join(args.output_dir, f"layer_diff_pairwise_cosine_similarity_token_{token_id}{suffix}.gif")
                imageio.mimsave(cosine_gif_path, cosine_frames, duration=1.0, loop=1000)
                print(f"Saved cosine GIF to {cosine_gif_path}")
            if len(l1_frames) > 0:
                l1_gif_path = os.path.join(args.output_dir, f"layer_diff_pairwise_l1_token_{token_id}{suffix}.gif")
                imageio.mimsave(l1_gif_path, l1_frames, duration=1.0, loop=1000)
                print(f"Saved L1 GIF to {l1_gif_path}")

            # Save occurrence GIF if frames were collected
            if len(occurence_frames) > 0:
                occurence_gif_path = os.path.join(args.output_dir, f"occurence_pairwise_cosine_similarity_token_{token_id}{suffix}.gif")
                imageio.mimsave(occurence_gif_path, occurence_frames, duration=1.0, loop=1000)
                print(f"Saved occurrence GIF to {occurence_gif_path}")

        # 0. Histogram of all differences
        all_diffs_v = torch.stack(all_diffs).flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(all_diffs_v, bins=100)
        plt.title(f'Distribution of All Differences for token "{token_str}"')
        plt.xlabel('Difference')
        plt.ylabel('Frequency')
        file_name = f'all_diffs_hist_{token_id}_{token_str}{suffix}.png'
        plt.savefig(os.path.join(plots_dir, file_name))
        print(f"saved to {os.path.join(plots_dir, file_name)}")
        plt.close()


        # Convert to numpy arrays for analysis
        all_diffs = torch.stack(all_diffs).numpy()
        layer_diffs = {k: torch.stack(v).numpy() for k, v in layer_diffs.items()}

        # 1. Magnitude of changes across layers
        diff_magnitudes = np.linalg.norm(all_diffs, axis=1)
        plt.figure(figsize=(10, 6))
        plt.hist(diff_magnitudes, bins=50)
        plt.title(f'Distribution of Embedding Change Magnitudes for token "{token_str}"')
        plt.xlabel('Magnitude of Change')
        plt.ylabel('Frequency')
        file_name = f'diff_magnitudes_{token_str}{suffix}.png'
        plt.savefig(os.path.join(plots_dir, file_name))
        print(f"saved to {os.path.join(plots_dir, file_name)}")
        plt.close()

        # 2. Layer-wise correlation of changes
        layer_correlations = np.zeros((num_layers-1, num_layers-1))
        for i in range(num_layers-1):
            for j in range(num_layers-1):
                corr, _ = spearmanr(layer_diffs[i].mean(axis=0), layer_diffs[j].mean(axis=0) )
                layer_correlations[i, j] = corr

        plt.figure(figsize=(12, 10))
        sns.heatmap(layer_correlations, annot=True, cmap='coolwarm', center=0,
                xticklabels=[f'L{i}' for i in range(num_layers-1)],
                yticklabels=[f'L{i}' for i in range(num_layers-1)])
        plt.title(f'Layer-wise Correlation of Embedding Changes for token "{token_str}"')
        file_name = f'layer_correlations_{token_id}_{token_str}{suffix}.png'
        plt.savefig(os.path.join(plots_dir, file_name))
        print(f"saved to {os.path.join(plots_dir, file_name)}")
        plt.close()

        # 3. PCA visualization of changes
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(all_diffs)

        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        plt.title(f'PCA of Embedding Changes for token "{token_str}"')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        file_name = f'pca_changes_{token_str}{suffix}.png'
        plt.savefig(os.path.join(plots_dir, file_name))
        print(f"saved to {os.path.join(plots_dir, file_name)}")
        plt.close()

        # 4. Layer-wise statistics
        layer_stats = {
            'mean_magnitude': [],
            'std_magnitude': [],
            'max_magnitude': [],
            'min_magnitude': []
        }

        for layer_idx in range(num_layers-1):
            layer_magnitudes = np.linalg.norm(layer_diffs[layer_idx], axis=1)
            layer_stats['mean_magnitude'].append(np.mean(layer_magnitudes))
            layer_stats['std_magnitude'].append(np.std(layer_magnitudes))
            layer_stats['max_magnitude'].append(np.max(layer_magnitudes))
            layer_stats['min_magnitude'].append(np.min(layer_magnitudes))

        # Plot layer-wise statistics
        plt.figure(figsize=(12, 8))
        x = range(num_layers-1)
        plt.plot(x, layer_stats['mean_magnitude'], 'b-', label='Mean')
        plt.fill_between(x,
                        np.array(layer_stats['mean_magnitude']) - np.array(layer_stats['std_magnitude']),
                        np.array(layer_stats['mean_magnitude']) + np.array(layer_stats['std_magnitude']),
                        alpha=0.2)
        plt.plot(x, layer_stats['max_magnitude'], 'r--', label='Max')
        plt.plot(x, layer_stats['min_magnitude'], 'g--', label='Min')
        plt.title(f'Layer-wise Statistics of Embedding Changes for token "{token_str}"')
        plt.xlabel('Layer')
        plt.ylabel('Magnitude of Change')
        plt.legend()
        file_name = f'layer_stats_{token_str}{suffix}.png'
        plt.savefig(os.path.join(plots_dir, file_name))
        print(f"saved to {os.path.join(plots_dir, file_name)}")
        plt.close()

        # 5. Save numerical results
        results = {
            'token': token_str,
            'token_id': token_id,
            'layer_correlations': layer_correlations,
            'layer_stats': layer_stats,
            'pca_explained_variance': pca.explained_variance_ratio_
        }

        file_name = f'analysis_results_{token_str}{suffix}.pt'
        torch.save(results, os.path.join(args.output_dir, file_name))
        print(f"saved to {os.path.join(args.output_dir, file_name)}")

    # Plot outliers overlap percentage for all tokens
    if analyze_outliers_indices_and_per_layer_overlap and outliers_overlap_data:
        plt.figure(figsize=(12, 8))
        for token_str, overlap_percents in outliers_overlap_data.items():
            plt.plot(range(len(overlap_percents)), overlap_percents, marker='o', label=token_str)

        plt.title(f'Outliers Overlap Percentage Across Layers. Quantile {args.analyze_outliers_indices_and_per_layer_overlap_quantile}')
        plt.xlabel('Layer Index')
        plt.ylabel('Outliers Overlap Percentage')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        file_name = f'outliers_overlap_percentage_quantile_q{args.analyze_outliers_indices_and_per_layer_overlap_quantile}{suffix}.png'
        plt.savefig(os.path.join(plots_dir, file_name))
        print(f"saved to {os.path.join(plots_dir, file_name)}")
        plt.close()
