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

# Llama instruct checkpoint
# python -m pdb -c continue src/transformers/models/llama/convert_hf_llama_to_adaptive_llama.py --model_type llama --from_llama unsloth/Meta-Llama-3.1-8B-Instruct --output_dir adaptive_hcg_llama31_8B_w_1.000_l_22-26_NTWKRP0G/calib60_instruct_checkpoint-5000 --hcg_fan_in_from adaptive_hcg_llama31_8B_w_1.000_l_22-26_NTWKRP0G/checkpoint-5000  --override_fan_in_layer_idx 22 --override_fan_out_layer_idx 26



# adaptive_hcg_llama31_8B_w_1.000_l_22-26_NTWKRP0G/checkpoint-5000
# adaptive_hcg_llama31_8B_w_0.100_l_22-26_HEHE7U06/checkpoint-5000
# adaptive_hcg_qwen25_7B_w_1.000_l_12-17_HXIGMJTI/checkpoint-5000
# adaptive_hcg_qwen25_7B_w_0.100_l_12-17_GQZARPC9/checkpoint-5000

# adaptive_hcg_llama31_8B_l22-26_analytical_pruning
# adaptive_hcg_qwen25_7B_l12-17_analytical_pruning

# python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --llama_checkpoint ./adaptive_hcg_llama31_8B_w_1.000_l_22-26_NTWKRP0G/checkpoint-5000 --output_dir results/calibrate_ppl/ --output_suffix hcg_llama31_8B_w_1.000
# python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --llama_checkpoint ./adaptive_hcg_llama31_8B_w_0.100_l_22-26_HEHE7U06/checkpoint-5000 --output_dir results/calibrate_ppl/ --output_suffix hcg_llama31_8B_w_0.100
# python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --llama_checkpoint ./adaptive_hcg_qwen25_7B_w_1.000_l_12-17_HXIGMJTI/checkpoint-5000 --output_dir results/calibrate_ppl/ --output_suffix hcg_qwen25_7B_w_1.000
# python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --llama_checkpoint ./adaptive_hcg_qwen25_7B_w_0.100_l_12-17_GQZARPC9/checkpoint-5000 --output_dir results/calibrate_ppl/ --output_suffix hcg_qwen25_7B_w_0.100


# Single Layer Hopping
#
# python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --llama_checkpoint ./adaptive_hcg_llama31_8B_one_w_0.100_l_3-4_01U2EJA9/checkpoint-5000/ --output_dir results/calibrate_ppl_single_layer/ --output_suffix hcg_llama31_8B_L3-4_w_0.1

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_suffix", type=str, required=True)

    args = parser.parse_args()

    tokens_frequency = {}

    tokenizer = AutoTokenizer.from_pretrained(args.llama_checkpoint)

    wikitext_103: datasets.Dataset = datasets.load_dataset("lighteval/wikitext_103", split="test")
    for item in wikitext_103:
        for token in tokenizer(item['text']).input_ids:
            tokens_frequency[token] = tokens_frequency.get(token, 0) + 1

    model_class = AdaptiveLlamaForCausalLM if "llama" in args.llama_checkpoint else AdaptiveQwen2ForCausalLM

    model = model_class.from_pretrained(args.llama_checkpoint, torch_dtype=torch.bfloat16)

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

    model_name = args.llama_checkpoint.split("/")[-1]

    df = pd.DataFrame(results)
    file_path = os.path.join(args.output_dir, f"ppl_results_{model_name}_{args.output_suffix}.csv")
    df.to_csv(file_path, index=False)
    logger.info(f"Saved PPL results to {file_path}")


if __name__ == "__main__":
    main()
