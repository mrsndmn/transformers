import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns

import random

from transformers.models.llama.analyze.embeddings_change import compute_distances, norm_compute_cosine_distance, compute_l1_distance


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_file", type=str, required=True, help="Path to the embeddings file (.pt)")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--tok_k_tokens", type=int, default=10)
    parser.add_argument("--num_hop_layers", type=int, default=1)
    parser.add_argument("--min_occurrencies", type=int, default=10)
    parser.add_argument("--max_process_occurrences", type=int, default=100)
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the file name")
    parser.add_argument("--output_dir", type=str, default="results/most_common_tokens_occurrences", help="Output directory")
    parser.add_argument("--least_common", type=bool, default=False, help="Least common tokens")
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

    assert args.suffix == ""

    for suffix in [ '1', '2', '3', '4' ]:
        suffix = f"_{suffix}"

        # suffix = args.suffix
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

            for occurence_idx in range(len(random_token_embeddings)):
                for layer_idx in range(num_layers - 1):
                    hs_i = random_token_embeddings[occurence_idx][layer_idx]

                    hs_j = random_token_embeddings[occurence_idx][layer_idx + args.num_hop_layers]

                    cos_distance = norm_compute_cosine_distance(hs_i, hs_j)
                    assert metric == 'cos'

                    cos_distance = cos_distance.cpu().float().numpy().item()
                    layer_data[f"Layer_{layer_idx}"].append(cos_distance)

            df = pd.DataFrame(layer_data)

            corr_method = 'pearson'
            corr_matrix = df.corr(method=corr_method)

            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
            plt.title(f"{corr_method.upper()} Correlation of {metric.upper()} Distances Between Layers for token {token_id} ({token_str})")
            plt.tight_layout()
            file_name = f"layer_correlation_{metric}_token_{token_id}{suffix}.png"
            file_path = os.path.join(args.output_dir, file_name)
            plt.savefig(file_path)
            print(f"saved to {file_path}")
            plt.close()

            # Plot distances heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
            plt.title(f"{metric.upper()} Distances Between Layers for token {token_id} ({token_str})")
            plt.tight_layout()
            file_name = f"layer_distances_{metric}_token_{token_id}{suffix}.png"
            file_path = os.path.join(args.output_dir, file_name)
            plt.savefig(file_path)
            print(f"saved to {file_path}")
            plt.close()





