import glob
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
from datasets import load_dataset

from transformers.models.llama.analyze.embeddings_change import compute_distances, norm_compute_cosine_distance, compute_l1_distance
from transformers.models.llama.analyze.mean_per_token_embeddings_change import trim_embeddings
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers.models.qwen2.modeling_adaptive_qwen2 import AdaptiveQwen2ForCausalLM

from transformers.models.llama.paper.calibrate_learned_vocabulary import calibrate_vocabulary

from transformers.models.llama.interpretation.explore_eval_hard_concrete_percent import evaluate_ppl_wikitext_103, evaluate_acc_hellaswag, compute_tokens_counts_hellaswag, compute_tokens_counts

from transformers.models.llama.paper.calibrate_learned_vocabulary_calc_ppl import evaluate_sparsity_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


# Single Layer Hopping
#
# python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --llama_checkpoint ./adaptive_hcg_llama31_8B_one_w_0.100_l_3-4_01U2EJA9/checkpoint-5000/ --output_suffix 

@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_idx", type=int, required=True)
    args = parser.parse_args()

    i = int(args.layer_idx)

    output_dir = 'results/calibrate_ppl_single_layer/'

    output_suffix = f"hcg_llama31_8B_L{i}-{i+1}_w_0.1"

    checkpoint_glob = sorted(glob.glob(f"./paper_checkpoints/single_layer_hopping/adaptive_hcg_llama31_8B_one_w_0.100_l_{i}-{i+1}_*/checkpoint-5*/"))

    if len(checkpoint_glob) == 0:
        logger.info(f"No checkpoint found for {output_suffix}")
        return

    checkpoint_path = checkpoint_glob[-1]
    print("use checkpoint checkpoint_path", checkpoint_path)

    if not os.path.exists(checkpoint_path):
        logger.info(f"Checkpoint {checkpoint_path} does not exist")
        return


    tokens_frequency = {}

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    wikitext_103: datasets.Dataset = datasets.load_dataset("mrsndmn/wikitext-2-raw-v1-validation", split="validation")
    for item in wikitext_103:
        for token in tokenizer(item['text']).input_ids:
            tokens_frequency[token] = tokens_frequency.get(token, 0) + 1

    model_class = AdaptiveLlamaForCausalLM if "llama" in checkpoint_path else AdaptiveQwen2ForCausalLM

    model = model_class.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)

    results = evaluate_sparsity_metrics(model, tokenizer, tokens_frequency)

    df = pd.DataFrame(results)
    result_file_path = os.path.join(output_dir, f"ppl_results_{output_suffix}.csv")
    df.to_csv(result_file_path, index=False)
    logger.info(f"Saved PPL results to {result_file_path}")


if __name__ == "__main__":
    main()
