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
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers.models.qwen2.modeling_adaptive_qwen2 import AdaptiveQwen2ForCausalLM


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def min_swaps_to_sort(arr):
    n = len(arr)
    arr_pos = list(enumerate(arr))
    arr_pos.sort(key=lambda x: x[1])
    visited = [False] * n
    swaps = 0

    for i in range(n):
        if visited[i] or arr_pos[i][0] == i:
            continue

        cycle_size = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = arr_pos[j][0]
            cycle_size += 1

        if cycle_size > 0:
            swaps += (cycle_size - 1)

    return swaps

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_checkpoint1", type=str, required=True)
    parser.add_argument("--llama_checkpoint2", type=str, required=True)

    args = parser.parse_args()

    model_class1 = AdaptiveLlamaForCausalLM if "llama" in args.llama_checkpoint1 else AdaptiveQwen2ForCausalLM
    model_class2 = AdaptiveLlamaForCausalLM if "llama" in args.llama_checkpoint2 else AdaptiveQwen2ForCausalLM

    model1 = model_class1.from_pretrained(args.llama_checkpoint1)
    model2 = model_class2.from_pretrained(args.llama_checkpoint2)

    log_d_data1 = model1.model.fan_in.hcg.hcg_log_a.data
    log_d_data2 = model2.model.fan_in.hcg.hcg_log_a.data

    model1_sorted_tokens = torch.argsort(log_d_data1).numpy().tolist()
    model1_sorted_tokens_to_idx = {token_id: idx for idx, token_id in enumerate(model1_sorted_tokens)}

    model2_sorted_tokens = torch.argsort(log_d_data2).numpy().tolist()

    model2_to_model1_token_idxs = [ model1_sorted_tokens_to_idx[token_id] for token_id in model2_sorted_tokens ]

    min_swaps = min_swaps_to_sort(model2_to_model1_token_idxs)

    print(f"Min swaps to sort model2 tokens to model1 tokens: {min_swaps}")

    breakpoint()

if __name__ == "__main__":
    main()
