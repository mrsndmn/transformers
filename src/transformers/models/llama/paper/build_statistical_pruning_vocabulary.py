import numpy as np
import pickle
from tqdm import tqdm
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import matplotlib.pyplot as plt
import argparse
import logging
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns
import imageio.v2 as imageio
import tempfile
from collections import Counter
import random
from sklearn.metrics.pairwise import cosine_distances

from transformers.models.llama.analyze.embeddings_change import compute_distances, norm_compute_cosine_distance, compute_l1_distance
from transformers.models.llama.analyze.mean_per_token_embeddings_change import trim_embeddings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--token_potential_min_df_csv_path", type=str, required=True)

    args = parser.parse_args()
    model_name = args.llama_checkpoint.split("/")[-1]

    token_potential_min_df = pd.read_csv(args.token_potential_min_df_csv_path)

    weighted_H_cols = [col for col in token_potential_min_df.columns if col.startswith("weighted_H")]

    # Find max 50 quantile for all weighted layers
    start_pruning_layer = token_potential_min_df[ weighted_H_cols ].quantile(q=0.5)
    weighted_H_to_start_pruning = start_pruning_layer.idxmax()

    # Calculate automatically
    H_to_start_pruning = weighted_H_to_start_pruning.removeprefix("weighted_")

    num_hop_layers_df = token_potential_min_df[[weighted_H_to_start_pruning, H_to_start_pruning]].groupby(by=H_to_start_pruning).sum()
    num_hop_layers = num_hop_layers_df.idxmax()[0]

    print("Start pruning layer weights:\n", start_pruning_layer)
    print("\n\nNumber of layers to be pruned:\n", num_hop_layers_df)

    print("\n\n\n=====\n")
    print(f"Pruning hidden state index: {weighted_H_to_start_pruning}")
    print(f"Recommended number of hop layers: {num_hop_layers}")

    filtered_df = token_potential_min_df[ token_potential_min_df[weighted_H_to_start_pruning] == num_hop_layers ]

    for quantile in [0.25, 0.5, 0.75, 1]:
        vocab_quantile = filtered_df.sort_values(by=weighted_H_to_start_pruning, ascending=False).head(int(quantile * len(filtered_df)))
        file_path = os.path.join(args.output_dir, f"pruning_vocab_q{quantile}_{model_name}.csv")
        vocab_quantile['token_id'].to_csv(file_path, index=False)
        print(f"Saved pruning vocab q{quantile} to {file_path} - Vocab size: {len(vocab_quantile)}")


if __name__ == "__main__":
    main()
