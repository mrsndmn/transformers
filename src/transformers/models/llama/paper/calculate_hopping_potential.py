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

import torch

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
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--hop_threshold", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=16)
    # parser.add_argument("--torch_compile", type=bool, action="store_true", default=True)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {args.llama_checkpoint}")

    model_vanilla = AutoModelForCausalLM.from_pretrained(args.llama_checkpoint, torch_dtype=torch.bfloat16)
    model_vanilla.to(device)
    model_vanilla.requires_grad_(False)
    # if args.torch_compile:
    #     print("Torch compile model")
    #     model_vanilla = torch.compile(model_vanilla)

    tokenizer = AutoTokenizer.from_pretrained(args.llama_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    data_files = [ f"data/CC-MAIN-2024-10/000_{i:05}.parquet" for i in range(1) ]
    text_dataset = datasets.load_dataset("HuggingFaceFW/fineweb", split="train", data_files=data_files, num_proc=16)

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=1024, return_tensors='pt')

        return tokenized_inputs

    print("len text_dataset", len(text_dataset), "args.num_samples", args.num_samples)
    if args.num_samples is not None:
        text_dataset = text_dataset.select(range(args.num_samples))

    text_dataset = text_dataset.map(tokenize_function, batched=True, num_proc=32)

    batch_size = args.batch_size
    total_batches = len(text_dataset) // batch_size

    # Create output directories
    output_dir = os.path.join("results", "token_embeddings_hopping_potential")
    os.makedirs(output_dir, exist_ok=True)

    # token_id -> list[ list_{hopping_potentials} ]
    per_token_hopping_potentials = dict()
    token_occurencies = Counter()

    batch_count = 0
    for batch in tqdm(text_dataset.iter(batch_size=batch_size), total=total_batches):
        batch_count += 1
        texts = batch['text']

        model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
        model_inputs = model_inputs.to(device)

        seq_len = model_inputs['input_ids'].shape[1]

        # [ batch_size, seq_len ]
        model_inputs['input_ids'] = model_inputs['input_ids'][:, :seq_len]
        # [ batch_size, seq_len ]
        model_inputs['attention_mask'] = model_inputs['attention_mask'][:, :seq_len]

        outputs_vanilla = model_vanilla(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
        )

        # [ num_layers, batch_size, seq_len, hidden_size ]
        hidden_states_cat = torch.stack(outputs_vanilla.hidden_states, dim=0)
        # [ batch_size, seq_len, num_layers, hidden_size ]
        hidden_states_cat = hidden_states_cat.permute(1, 2, 0, 3)

        input_ids_cpu = model_inputs['input_ids'].cpu()
        current_batch_size = input_ids_cpu.shape[0]

        num_hidden_states = len(outputs_vanilla.hidden_states)

        for batch_idx in range(current_batch_size):
            for seq_idx in range(seq_len):
                if model_inputs['attention_mask'][batch_idx, seq_idx] == 0:
                    break

                token_id = input_ids_cpu[batch_idx, seq_idx].item()

                token_hidden_states = hidden_states_cat[batch_idx, seq_idx]
                cosine_similarities = pairwise_cosine_similarity(token_hidden_states, token_hidden_states)
                # [ num_hidden_states, num_hidden_states ]
                cosine_similarities_cpu = cosine_similarities.cpu()

                max_hop_layers = []
                for i in range(num_hidden_states-1):
                    max_hop_layers_i = 0
                    for j in range(i+1, num_hidden_states):
                        if cosine_similarities_cpu[i, j] < args.hop_threshold:
                            break
                        else:
                            max_hop_layers_i += 1

                    max_hop_layers.append(max_hop_layers_i)

                if token_id not in per_token_hopping_potentials:
                    per_token_hopping_potentials[token_id] = []

                per_token_hopping_potentials[token_id].append(max_hop_layers)
                token_occurencies[token_id] += 1

    # end iteration over dataset

    # Save per_token_hopping_potentials to file
    model_name = args.llama_checkpoint.split("/")[-1]

    file_path = os.path.join(output_dir, f"per_token_hopping_potentials_{model_name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(per_token_hopping_potentials, f)
    print(f"Saved per_token_hopping_potentials to {file_path}")

    # Save token_occurencies to file
    file_path = os.path.join(output_dir, f"token_occurencies_{model_name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(token_occurencies, f)
    print(f"Saved token_occurencies to {file_path}")
