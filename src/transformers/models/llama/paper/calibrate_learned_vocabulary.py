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

@torch.no_grad()
def calibrate_vocabulary(model, tokens_frequency, expected_sparsity):
    log_d_data = model.model.fan_in.hcg.hcg_log_a.data

    # Ranked tokens importance
    ranked_token_idxs = torch.argsort(log_d_data).cpu().numpy().tolist()

    total_tokens = sum(tokens_frequency.values())

    print("Total tokens in vocabulary", total_tokens)

    new_log_a = torch.ones_like(log_d_data, device='cpu') * 10

    current_pruned_tokens = 0

    boarderline_token_idx = 0
    for token_id in ranked_token_idxs:
        if current_pruned_tokens / total_tokens * 100 >= expected_sparsity:
            break

        current_pruned_tokens += tokens_frequency.get(token_id, 0)
        boarderline_token_idx += 1
        new_log_a[token_id] = -10

    print("Boarderline token idx", boarderline_token_idx, "estimated sparsity", current_pruned_tokens / total_tokens * 100)

    new_log_a = new_log_a.to(log_d_data.device)

    model.model.fan_in.hcg.hcg_log_a.data = new_log_a

    return

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_checkpoint", type=str, required=True)
    # parser.add_argument("--tokens_frequency_csv_path", type=str, required=True, help="Pickle file with tokens frequency for calibration")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--expected_sparsity", type=int, required=True, help="Expected sparsity of the vocabulary")

    args = parser.parse_args()

    # tokens_frequency_df = pickle.load(open(args.tokens_frequency_csv_path, "rb"))

    tokens_frequency = {}

    tokenizer = AutoTokenizer.from_pretrained(args.llama_checkpoint)

    wikitext_103: datasets.Dataset = datasets.load_dataset("mrsndmn/wikitext-2-raw-v1-validation", split="validation")
    for item in wikitext_103:
        for token in tokenizer(item['text']).input_ids:
            tokens_frequency[token] = tokens_frequency.get(token, 0) + 1


    model_class = AdaptiveLlamaForCausalLM if "llama" in args.llama_checkpoint else AdaptiveQwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.llama_checkpoint)
    model = model_class.from_pretrained(args.llama_checkpoint, torch_dtype=torch.bfloat16)

    calibrate_vocabulary(model, tokens_frequency, args.expected_sparsity)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model and tokenizer to {args.output_dir} with calibrated vocabulary")

    breakpoint()


if __name__ == "__main__":
    main()
