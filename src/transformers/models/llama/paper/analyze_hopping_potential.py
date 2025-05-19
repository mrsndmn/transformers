import numpy as np
import pickle
from tqdm import tqdm
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_checkpoint", type=str, required=True)
    parser.add_argument("--per_token_hopping_potentials_path", type=str, required=True)
    parser.add_argument("--token_occurences_path", type=str, required=True)
    parser.add_argument("--min_frequency", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def load_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.llama_checkpoint)

    with open(args.per_token_hopping_potentials_path, "rb") as f:
        per_token_hopping_potentials = pickle.load(f)

    with open(args.token_occurences_path, "rb") as f:
        token_occurences = pickle.load(f)

    return tokenizer, per_token_hopping_potentials, token_occurences

def analyze_token_potentials(tokenizer, per_token_hopping_potentials, token_occurences, max_tokens):
    top_k_tokens = [token_id for token_id, _ in token_occurences.most_common(max_tokens)]
    token_potential_min = []

    for token_id in tqdm(top_k_tokens):
        hopping_potentials = per_token_hopping_potentials[token_id]
        hopping_potentials_t = torch.tensor(hopping_potentials)
        hopping_potentials_t_min = hopping_potentials_t.min(dim=0).values
        hopping_potential_min = hopping_potentials_t_min.numpy().tolist()

        token_freq = token_occurences[token_id]
        hopping_potentials_t_min_weighted = (np.log10(hopping_potentials_t_min.numpy() * token_freq)).tolist()

        token_potential_min.append({
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "token_freq": token_freq,
            **{f"H{i}": v for i, v in enumerate(hopping_potential_min)},
            **{f"weighted_H{i}": v for i, v in enumerate(hopping_potentials_t_min_weighted)},
        })

    token_potential_min_df = pd.DataFrame(token_potential_min)
    H_cols = [col for col in token_potential_min_df.columns if col.startswith("H")]
    token_potential_min_df['mean_hopping_potential'] = token_potential_min_df[H_cols].mean(axis=1)

    weighted_H_cols = [col for col in token_potential_min_df.columns if col.startswith("weighted_H")]
    token_potential_min_df['mean_weighted_hopping_potential'] = token_potential_min_df[weighted_H_cols].mean(axis=1)

    return token_potential_min_df

def plot_log_freq_vs_hopping_potential(token_potential_min_df, output_dir, model_name=""):
    log_freq = np.log10(token_potential_min_df['token_freq'])

    bin_step = 1
    bins = np.arange(np.floor(log_freq.min()),
                    np.ceil(log_freq.max()), step=bin_step)
    token_potential_min_df['log_freq_bin'] = pd.cut(log_freq, bins=bins)

    plt.figure(figsize=(12, 6))
    sns.violinplot(x='log_freq_bin', y='mean_hopping_potential', data=token_potential_min_df, inner='box')
    plt.xticks(rotation=45)
    plt.xlabel(f'log10(freq) bins (width = {bin_step})')
    plt.ylabel('mean hopping potential')
    plt.ylim(0, 4)
    plt.title('Mean Hopping Potential vs Log(Freq) Bins')
    plt.tight_layout()
    file_path = os.path.join(output_dir, f"boxplot_log_freq_bins_vs_mean_hopping_potential_{model_name}.png")
    plt.savefig(file_path)
    print(f"Saved boxplot to {file_path}")
    plt.close()
    return file_path

def plot_hopping_potential_distribution(token_potential_min_df, output_dir, model_name=""):
    H_cols = [col for col in token_potential_min_df.columns if col.startswith("H")]
    token_potential_min_df[H_cols].plot(kind='box', figsize=(15, 6))
    plt.xticks(rotation=90)
    plt.ylim(0, 10)
    plt.ylabel('Hop Layers')
    plt.xlabel('Hidden states to hop from')
    plt.grid(True)
    plt.tight_layout()
    file_path = os.path.join(output_dir, f"hopping_potential_min_boxplot_{model_name}.png")
    plt.savefig(file_path)
    print(f"Saved boxplot to {file_path}")
    plt.close()
    return file_path

def plot_weighted_hopping_potential_distribution(token_potential_min_df, output_dir, model_name=""):
    weighted_H_cols = [col for col in token_potential_min_df.columns if col.startswith("weighted_H")]
    token_potential_min_df[weighted_H_cols].plot(kind='box', figsize=(15, 6))
    plt.xticks(rotation=90)
    plt.ylabel('Log Hopping Potential')
    plt.xlabel('Hidden states to hop from')
    plt.grid(True)
    plt.tight_layout()
    file_path = os.path.join(output_dir, f"weighted_hopping_potential_min_boxplot_{model_name}.png")
    plt.savefig(file_path)
    print(f"Saved boxplot to {file_path}")
    plt.close()
    return file_path

def plot_token_occurrences_distribution(token_occurences, output_dir, model_name=""):
    token_occurences_values = np.array(list(token_occurences.values()))
    plt.hist(token_occurences_values, bins=100, edgecolor='black')
    plt.xlabel('Log token occurences')
    plt.ylabel('Number of tokens')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Token occurences distribution')
    file_path = os.path.join(output_dir, f"token_occurences_distribution_{model_name}.png")
    plt.savefig(file_path)
    print(f"Saved token occurences distribution to {file_path}")
    plt.close()
    return file_path

def analyze_hopping_potentials(token_potential_min_df):
    H_cols = [col for col in token_potential_min_df.columns if col.startswith("H")]
    max_h_per_token = token_potential_min_df[H_cols].idxmax(axis=1)
    ranked_h = max_h_per_token.value_counts().sort_values(ascending=False).head(10)

    log_freq = np.log10(token_potential_min_df['token_freq'])
    mean_hopping_potential_corr = token_potential_min_df['mean_hopping_potential'].corr(log_freq)

    return ranked_h, mean_hopping_potential_corr

def main():
    args = parse_args()
    tokenizer, per_token_hopping_potentials, token_occurences = load_data(args)

    model_name = args.llama_checkpoint.split("/")[-1]

    print("total tokens", len(token_occurences))
    print("total token occurences", sum(token_occurences.values()))

    max_tokens = len(token_occurences)
    if args.max_tokens is not None:
        max_tokens = min(args.max_tokens, len(token_occurences))

    token_potential_min_df = analyze_token_potentials(tokenizer, per_token_hopping_potentials, token_occurences, max_tokens)

    file_path = os.path.join(args.output_dir, f"token_potential_min_{model_name}.csv")
    token_potential_min_df.to_csv(file_path, index=False)
    print(f"Saved token potential min to {file_path}")

    plot_log_freq_vs_hopping_potential(token_potential_min_df, args.output_dir, model_name=model_name)

    plot_hopping_potential_distribution(token_potential_min_df, args.output_dir, model_name=model_name)
    plot_weighted_hopping_potential_distribution(token_potential_min_df, args.output_dir, model_name=model_name)

    plot_token_occurrences_distribution(token_occurences, args.output_dir, model_name=model_name)


if __name__ == "__main__":
    main()
