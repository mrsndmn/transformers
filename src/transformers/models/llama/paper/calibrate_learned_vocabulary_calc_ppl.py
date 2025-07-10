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
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_distances

from transformers.models.llama.analyze.embeddings_change import compute_distances, norm_compute_cosine_distance, compute_l1_distance
from transformers.models.llama.analyze.mean_per_token_embeddings_change import trim_embeddings
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers.models.qwen2.modeling_adaptive_qwen2 import AdaptiveQwen2ForCausalLM

from transformers.models.llama.paper.calibrate_learned_vocabulary import calibrate_vocabulary

from transformers.models.llama.interpretation.explore_eval_hard_concrete_percent import evaluate_ppl_wikitext_103, evaluate_acc_hellaswag, compute_tokens_counts_hellaswag, compute_tokens_counts, evaluate_tiny_stories

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

# python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --llama_checkpoint ./paper_checkpoints/base_thshld_0.6/adaptive_hcg_llama31_8B_learned_vocab_w_1.000_l_18-26_M0K275CH/checkpoint-5306 --output_dir results/calibrate_ppl/ --output_suffix hcg_llama31_8B_w_1.000_thshold_0.6
# python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --llama_checkpoint ./paper_checkpoints/base_thshld_0.6/adaptive_hcg_llama31_8B_learned_vocab_w_0.100_l_18-26_DAZ0UXGX/checkpoint-5306 --output_dir results/calibrate_ppl/ --output_suffix hcg_llama31_8B_w_0.100_thshold_0.6
# python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --llama_checkpoint ./paper_checkpoints/base_thshld_0.6/adaptive_hcg_qwen25_7B_learned_vocab_w_1.000_l_9-20_J3BTODV3/checkpoint-5306 --output_dir results/calibrate_ppl/ --output_suffix hcg_qwen25_7B_w_1.000_thshold_0.6
# python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --llama_checkpoint ./paper_checkpoints/base_thshld_0.6/adaptive_hcg_qwen25_7B_learned_vocab_w_0.100_l_9-20_3P72MPZU/checkpoint-5306 --output_dir results/calibrate_ppl/ --output_suffix hcg_qwen25_7B_w_0.100_thshold_0.6


def evaluate_sparsity_metrics(model, tokenizer, tokens_frequency, sparsity_only=False, model_name=None, with_wikitext=False):

    # print("Count wikitext_103 tokens frequency")
    # wikitext_103_dataset = load_dataset('lighteval/wikitext_103', 'default', split='test')
    # wikitext_103_bincount = compute_tokens_counts(model, tokenizer, wikitext_103_dataset)

    hellaswag_bincount_path = f"hellaswag_bincount_{model_name}.pt"
    if os.path.exists(hellaswag_bincount_path):
        print("Load hellaswag tokens frequency")
        hellaswag_bincount = torch.load(hellaswag_bincount_path)
    else:
        print("Count hellaswag tokens frequency")
        hellaswag_dataset = load_dataset('hellaswag', 'default', split='validation')
        hellaswag_bincount = compute_tokens_counts_hellaswag(model, tokenizer, hellaswag_dataset)
        torch.save(hellaswag_bincount, hellaswag_bincount_path)

    tiny_stories_bincount_path = f"tiny_stories_bincount_{model_name}.pt"
    if os.path.exists(tiny_stories_bincount_path):
        print("Load tiny_stories tokens frequency")
        tiny_stories_bincount = torch.load(tiny_stories_bincount_path)
    else:
        print("Count tiny_stories tokens frequency")
        tiny_stories_dataset = load_dataset('roneneldan/TinyStories', 'default', split='validation')
        tiny_stories_bincount = compute_tokens_counts(model, tokenizer, tiny_stories_dataset)
        torch.save(tiny_stories_bincount, tiny_stories_bincount_path)

    if with_wikitext:
        wikitext_103_bincount_path = f"wikitext_103_bincount_{model_name}.pt"
        if os.path.exists(wikitext_103_bincount_path):
            print("Load wikitext_103 tokens frequency")
            wikitext_103_bincount = torch.load(wikitext_103_bincount_path)
        else:
            print("Count wikitext_103 tokens frequency")
            wikitext_103_dataset = load_dataset('lighteval/wikitext_103', 'default', split='test')
            wikitext_103_bincount = compute_tokens_counts(model, tokenizer, wikitext_103_dataset)
            torch.save(wikitext_103_bincount, wikitext_103_bincount_path)

    results = []

    for expected_sparsity in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        calibrate_vocabulary(model, tokens_frequency, expected_sparsity)
        
        extra_results = {}
        if with_wikitext:
            wikitext_results = evaluate_ppl_wikitext_103(model, bincount=wikitext_103_bincount.clone(), sparsity_only=sparsity_only)
            extra_results['wikitext_ppl'] = wikitext_results['ppl']
            extra_results['wikitext_ppl_stderr'] = wikitext_results['ppl_stderr']
            extra_results['wikitext_pruned_percent'] = wikitext_results['pruned_percent']
            extra_results['wikitext_total_tokens_count'] = wikitext_results['total_tokens_count']

        tiny_stories_results = evaluate_tiny_stories(model, bincount=tiny_stories_bincount.clone(), sparsity_only=sparsity_only)
        tiny_stories_ppl = tiny_stories_results['ppl']
        tiny_stories_ppl_stderr = tiny_stories_results['ppl_stderr']
        tiny_stories_pruned_percent = tiny_stories_results['pruned_percent']
        tiny_stories_total_tokens_count = tiny_stories_results['total_tokens_count']

        hellaswag_results = evaluate_acc_hellaswag(model, bincount=hellaswag_bincount.clone(), sparsity_only=sparsity_only)
        acc_norm = hellaswag_results['acc_norm']
        acc_norm_stderr = hellaswag_results['acc_norm_stderr']
        hellaswag_pruned_percent = hellaswag_results['pruned_percent']
        hellaswag_total_tokens_count = hellaswag_results['total_tokens_count']

        result = {
            "sparsity": expected_sparsity,

            **extra_results,

            "hellaswag_acc_norm": acc_norm,
            "hellaswag_acc_norm_stderr": acc_norm_stderr,
            "hellaswag_pruned_percent": hellaswag_pruned_percent,
            "hellaswag_total_tokens_count": hellaswag_total_tokens_count,

            "tiny_stories_ppl": tiny_stories_ppl,
            "tiny_stories_ppl_stderr": tiny_stories_ppl_stderr,
            "tiny_stories_pruned_percent": tiny_stories_pruned_percent,
            "tiny_stories_total_tokens_count": tiny_stories_total_tokens_count,
        }

        print(result)

        results.append(result)

    return results

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_suffix", type=str, required=True)
    parser.add_argument("--sparsity_only", action="store_true", default=False)
    parser.add_argument("--calibration_dataset", type=str, default="wikitext_103")

    args = parser.parse_args()

    assert args.calibration_dataset in ["wikitext_103", "tiny_stories"]

    with_wikitext = args.calibration_dataset == "wikitext_103"

    sparsity_only = args.sparsity_only

    tokens_frequency = {}

    tokenizer = AutoTokenizer.from_pretrained(args.llama_checkpoint)

    if args.calibration_dataset == "wikitext_103":
        wikitext_103: datasets.Dataset = datasets.load_dataset("lighteval/wikitext_103", split="test")
        for item in wikitext_103:
            for token in tokenizer(item['text']).input_ids:
                tokens_frequency[token] = tokens_frequency.get(token, 0) + 1
    elif args.calibration_dataset == "tiny_stories":
        tiny_stories_dataset = load_dataset('roneneldan/TinyStories', 'default', split='validation')
        for item in tiny_stories_dataset:
            for token in tokenizer(item['text']).input_ids:
                tokens_frequency[token] = tokens_frequency.get(token, 0) + 1

    model_class = AdaptiveLlamaForCausalLM
    if "qwen" in args.llama_checkpoint.lower():
        model_class = AdaptiveQwen2ForCausalLM

    model = model_class.from_pretrained(args.llama_checkpoint, torch_dtype=torch.bfloat16)

    model_name = args.llama_checkpoint.replace("/", "_").replace(":", "_").replace(".", "_")

    results = evaluate_sparsity_metrics(model, tokenizer, tokens_frequency, sparsity_only=sparsity_only, model_name=model_name, with_wikitext=with_wikitext)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    file_path = os.path.join(args.output_dir, f"ppl_results_{model_name}_{args.output_suffix}.csv")

    file_path_old_backup = file_path.replace(".csv", "_old.csv")
    if os.path.exists(file_path_old_backup):
        old_df = pd.read_csv(file_path_old_backup)

        for col in ['wikitext_ppl','wikitext_ppl_stderr','wikitext_total_tokens_count','hellaswag_acc_norm','hellaswag_acc_norm_stderr','hellaswag_total_tokens_count']:
            df[col] = old_df[col]

    df.to_csv(file_path, index=False)
    logger.info(f"Saved PPL results to {file_path}")


if __name__ == "__main__":
    main()
