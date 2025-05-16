import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import imageio
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

    args = parser.parse_args()

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

                    hs_diff = (hs_j - hs_i).float()

                    all_diffs.append(hs_diff)
                    layer_diffs[layer_idx].append(hs_diff)
                    occurence_diffs.append(hs_diff)
            
            for layer_idx in range(num_layers - 1):
                current_layer_diffs = layer_diffs[layer_idx]
                # Heatmap of pairwise distances of layers differences
                layer_diffs_t = torch.stack(current_layer_diffs).to(torch.float64)
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

                occurence_diffs_l1 = torch.cdist(layer_diffs_t, layer_diffs_t, p=1)
                occurence_diffs_l1 = occurence_diffs_l1.numpy()

                # Plot pairwise l1 similarity heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(occurence_diffs_l1, annot=True, cmap='coolwarm', vmax=100, fmt=".0f")
                plt.title(f"Pairwise L1 Distance Between Layers Differences. Layer {layer_idx}. Token {token_id} ({token_str})")
                plt.tight_layout()
                
                # Save frame for L1 GIF
                l1_frame_path = os.path.join(temp_dir, f"l1_frame_{layer_idx}.png")
                plt.savefig(l1_frame_path)
                l1_frames.append(imageio.imread(l1_frame_path))
                plt.close()

            # Save GIFs
            cosine_gif_path = os.path.join(args.output_dir, f"layer_diff_pairwise_cosine_similarity_token_{token_id}{suffix}.gif")
            l1_gif_path = os.path.join(args.output_dir, f"layer_diff_pairwise_l1_token_{token_id}{suffix}.gif")
            
            imageio.mimsave(cosine_gif_path, cosine_frames, duration=1.0)
            imageio.mimsave(l1_gif_path, l1_frames, duration=1.0)
            print(f"Saved cosine GIF to {cosine_gif_path}")
            print(f"Saved L1 GIF to {l1_gif_path}")

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
