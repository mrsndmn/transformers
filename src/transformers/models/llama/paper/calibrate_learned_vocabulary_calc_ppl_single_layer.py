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

from transformers.models.llama.analyze.embeddings_change import compute_distances, norm_compute_cosine_distance, compute_l1_distance
from transformers.models.llama.analyze.mean_per_token_embeddings_change import trim_embeddings
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers.models.qwen2.modeling_adaptive_qwen2 import AdaptiveQwen2ForCausalLM

from transformers.models.llama.paper.calibrate_learned_vocabulary import calibrate_vocabulary

from transformers.models.llama.interpretation.explore_eval_hard_concrete_percent import evaluate_ppl_wikitext_103

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

    output_dir = 'results/calibrate_ppl_single_layer/'

    for i in range(2, 30):
        output_suffix = f"hcg_llama31_8B_L{i}-{i+1}_w_0.1"

        checkpoint_glob = sorted(glob.glob(f"./adaptive_hcg_llama31_8B_one_w_0.100_l_{i}-{i+1}_*/checkpoint-5000/"))

        if len(checkpoint_glob) == 0:
            logger.info(f"No checkpoint found for {output_suffix}")
            continue

        checkpoint_path = checkpoint_glob[-1]
        print("use checkpoint checkpoint_path", checkpoint_path)

        if not os.path.exists(checkpoint_path):
            logger.info(f"Checkpoint {checkpoint_path} does not exist")
            continue

        result_file_path = os.path.join(output_dir, f"ppl_results_{output_suffix}.csv")

        if os.path.exists(result_file_path):
            continue


        tokens_frequency = {}

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        wikitext_103: datasets.Dataset = datasets.load_dataset("lighteval/wikitext_103", split="test")
        for item in wikitext_103:
            for token in tokenizer(item['text']).input_ids:
                tokens_frequency[token] = tokens_frequency.get(token, 0) + 1

        model_class = AdaptiveLlamaForCausalLM if "llama" in checkpoint_path else AdaptiveQwen2ForCausalLM

        model = model_class.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)

        results = []

        for expected_sparsity in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        # for expected_sparsity in [ 10 ]:
            calibrate_vocabulary(model, tokens_frequency, expected_sparsity)

            wikitext_results = evaluate_ppl_wikitext_103(model, max_samples=1000)
            ppl = wikitext_results['ppl']
            ppl_stderr = wikitext_results['ppl_stderr']

            results.append({
                "sparsity": expected_sparsity,
                "ppl": ppl,
                "ppl_stderr": ppl_stderr,
            })

        df = pd.DataFrame(results)
        df.to_csv(result_file_path, index=False)
        logger.info(f"Saved PPL results to {result_file_path}")


if __name__ == "__main__":
    main()
