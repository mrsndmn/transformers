import numpy as np

import pickle
from tqdm import tqdm
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from collections import Counter
import seaborn as sns

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
    parser.add_argument("--llama_checkpoint", type=str, required=True)
    parser.add_argument("--per_token_hopping_potentials_path", type=str, required=True)
    parser.add_argument("--token_occurences_path", type=str, required=True)
    parser.add_argument("--min_frequency", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # Plot figures

    tokenizer = AutoTokenizer.from_pretrained(args.llama_checkpoint)

    with open(args.per_token_hopping_potentials_path, "rb") as f:
        per_token_hopping_potentials = pickle.load(f)

    with open(args.token_occurences_path, "rb") as f:
        token_occurences = pickle.load(f)

    print("total tokens", len(token_occurences))
    print("total token occurences", sum(token_occurences.values()))

    max_tokens = min(args.max_tokens, len(token_occurences))

    top_k_tokens = [token_id for token_id, _ in token_occurences.most_common(max_tokens)]

    token_potential_min = []

    for token_id in tqdm(top_k_tokens):
        hopping_potentials = per_token_hopping_potentials[token_id]
        # [ num_occurencies, num_hidden_states ]
        hopping_potentials_t = torch.tensor(hopping_potentials)
        hopping_potentials_t_min = hopping_potentials_t.min(dim=0).values
        hopping_potential_min = hopping_potentials_t_min.numpy().tolist()

        token_freq = token_occurences[token_id]
        hopping_potentials_t_min_weighted = (hopping_potentials_t_min.numpy() * token_freq).tolist()

        token_potential_min.append({
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "token_freq": token_freq,
            **{ f"H{i}": v for i, v in enumerate(hopping_potential_min) },
            **{ f"weighted_H{i}": v for i, v in enumerate(hopping_potentials_t_min_weighted) },
        })


    token_potential_min_df = pd.DataFrame(token_potential_min)
    token_potential_min_df.to_csv(os.path.join(args.output_dir, "token_potential_min.csv"), index=False)

    H_cols = [col for col in token_potential_min_df.columns if col.startswith("H")]

    # Для каждого токена найти, какая H* дала максимальное значение
    max_h_per_token = token_potential_min_df[H_cols].idxmax(axis=1)

    # Подсчёт количества, сколько раз каждая H* была максимальной
    ranked_h = max_h_per_token.value_counts().sort_values(ascending=False).head(10)

    print("Ранжирование H* по количеству максимумов:")
    print(ranked_h)

    token_potential_min_df['mean_hopping_potential'] = token_potential_min_df[H_cols].mean(axis=1)
    token_potential_min_df['max_hopping_potential'] = token_potential_min_df[H_cols].max(axis=1)

    log_freq = np.log10(token_potential_min_df['token_freq'])

    mean_hopping_potential_corr = token_potential_min_df['mean_hopping_potential'].corr( log_freq )
    print(f"mean_hopping_potential and log freq corr: {mean_hopping_potential_corr}")

    # Scatter plot log freq vs mean hopping potential
    bin_step = 1
    bins = np.arange(np.floor(log_freq.min()),
                 np.ceil(log_freq.max()), step=bin_step)

    token_potential_min_df['log_freq_bin'] = pd.cut(log_freq, bins=bins)

    plt.figure(figsize=(12, 6))
    # sns.boxplot(x='log_freq_bin', y='mean_hopping_potential', data=token_potential_min_df)
    sns.violinplot(x='log_freq_bin', y='mean_hopping_potential', data=token_potential_min_df, inner='box')
    plt.xticks(rotation=45)
    plt.xlabel(f'log10(freq) bins (width = {bin_step})')
    plt.ylabel('mean hopping potential')
    plt.ylim(0, 4)
    plt.title('Mean Hopping Potential vs Log(Freq) Bins')
    plt.tight_layout()
    file_path = os.path.join(args.output_dir, "boxplot_log_freq_bins_vs_mean_hopping_potential.png")
    plt.savefig(file_path)
    print(f"Saved boxplot for log freq vs mean hopping potential to {file_path}")
    plt.close()


    # Построить распределение значений для каждой H*
    token_potential_min_df[H_cols].plot(kind='box', figsize=(15, 6))
    # long_df = token_potential_min_df.melt(id_vars=['token_id'], value_vars=H_cols, var_name='H', value_name='value')
    # sns.violinplot(x='H', y='value', data=long_df, inner='box')  # inner='box' отображает медиану и квартиль
    # sns.violinplot(x='log_freq_bin', y='mean_hopping_potential', data=token_potential_min_df, inner='box')

    plt.xticks(rotation=90)
    plt.ylim(0, 10)
    plt.ylabel('Hop Layers')
    plt.xlabel('Hidden states to hop from')
    plt.grid(True)
    plt.tight_layout()
    file_path = os.path.join(args.output_dir, "hopping_potential_min_boxplot.png")
    plt.savefig(file_path)
    print(f"Saved hopping potential min boxplot to {file_path}")
    plt.close()

    # Plot token occurences distribution
    token_occurences_values = np.array(list(token_occurences.values()))
    plt.hist(token_occurences_values, bins=100, edgecolor='black')
    plt.xlabel('Log token occurences')
    plt.ylabel('Number of tokens')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Token occurences distribution')
    file_path = os.path.join(args.output_dir, "token_occurences_distribution.png")
    plt.savefig(file_path)
    print(f"Saved token occurences distribution to {file_path}")
    plt.close()

    breakpoint()

    # # plt.scatter(range(len(hopping_potential_min)), hopping_potential_min, label=f"{token_id} [{tokenizer.decode([token_id])}]")
    # plt.legend()
    # file_path = os.path.join(args.output_dir, "hopping_potential_min.png")
    # plt.savefig(file_path)
    # print(f"Saved hopping potential min to {file_path}")
    # plt.close()
