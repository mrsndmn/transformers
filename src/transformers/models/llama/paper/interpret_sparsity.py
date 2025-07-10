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

import torch.nn.functional as F

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

# Llama
# python src/transformers/models/llama/paper/interpret_sparsity.py --llama_checkpoint ./adaptive_hcg_llama31_8B_w_0.100_l_22-26_HEHE7U06/checkpoint-5000 --output_dir results/calibrate_ppl/ --output_suffix hcg_llama31_8B_w_0.100

# Qwen
# python src/transformers/models/llama/paper/interpret_sparsity.py --llama_checkpoint ./adaptive_hcg_qwen25_7B_w_0.100_l_12-17_GQZARPC9/checkpoint-5000 --output_dir results/calibrate_ppl/ --output_suffix hcg_qwen25_7B_w_0.100

def print_colored(tokens, color_mask, tokenizer):
    # tokens : List[int]
    # color_mask : List[bool]
    # print token by token and color masked tokens
    tokens = tokens[0]
    color_mask = color_mask[0]

    assert len(tokens) == len(color_mask)

    for i, token in enumerate(tokens):
        token_str = tokenizer.decode([token])
        if not color_mask[i]:
            print(f"\033[91m{token_str}\033[0m", end="")
        else:
            print(token_str, end="")



@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_checkpoint", type=str, required=True)

    args = parser.parse_args()

    tokens_frequency = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.llama_checkpoint)

    wikitext_103: datasets.Dataset = datasets.load_dataset("lighteval/wikitext_103", split="test")
    for item in wikitext_103:
        for token in tokenizer(item['text']).input_ids:
            tokens_frequency[token] = tokens_frequency.get(token, 0) + 1

    model_class = AdaptiveLlamaForCausalLM if "llama" in args.llama_checkpoint else AdaptiveQwen2ForCausalLM

    model = model_class.from_pretrained(args.llama_checkpoint, torch_dtype=torch.bfloat16).to(device)

    model_inputs = tokenizer( [ wikitext_103[2]['text'][:650] ], return_tensors="pt")
    model_inputs = model_inputs.to(device)

    prev_accumulated_ppl = None

    for expected_sparsity in [ 0, 10, 25, 50, 75, 90 ]:
    # for expected_sparsity in [ 90 ]:
        calibrate_vocabulary(model, tokens_frequency, expected_sparsity)

        outputs  = model(**model_inputs, labels=model_inputs["input_ids"], use_cache=False)

        merging_logits = [ x for x in  outputs.fan_in_merging_logits if x is not None ][0]
        merging_logits = merging_logits.squeeze(-1)
        merging_map = (merging_logits > 0)

        logits = outputs.logits
        input_ids = model_inputs["input_ids"]

        # Shift for next-token prediction
        shifted_logits = logits[:, :-1, :]
        shifted_labels = input_ids[:, 1:]

        # Convert to log-probabilities
        log_probs = F.log_softmax(shifted_logits, dim=-1)

        # Negative log-likelihood
        nll = -log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len - 1)

        # Optionally: Cumulative perplexity over time
        cumulative_nll = nll.cumsum(dim=1)
        token_positions = torch.arange(1, cumulative_nll.shape[1] + 1, device=logits.device)

        # Average negative log-likelihood and exponentiate to get perplexity
        accumulated_ppl_result = torch.exp(cumulative_nll / token_positions)
        accumulated_ppl_result = accumulated_ppl_result.to(torch.float32)

        if prev_accumulated_ppl is not None:
            per_token_ppl_increment = (accumulated_ppl_result - prev_accumulated_ppl)

            median_not_pruned_tokens_ppl_increment = per_token_ppl_increment[merging_map[:, :-1]].cpu().quantile(0.5).item()
            median_pruned_tokens_ppl_increment = per_token_ppl_increment[~merging_map[:, :-1]].cpu().quantile(0.5).item()

            print("\n")
            print(f"Expected sparsity: {expected_sparsity}")
            print(f"Median PPL increment for not pruned tokens: {median_not_pruned_tokens_ppl_increment}")
            print(f"Median PPL increment for pruned tokens: {median_pruned_tokens_ppl_increment}")


        prev_accumulated_ppl = accumulated_ppl_result

        print_colored(model_inputs["input_ids"].cpu().numpy().tolist(), merging_map.cpu().numpy().tolist(), tokenizer)

        # breakpoint()

if __name__ == "__main__":
    main()
