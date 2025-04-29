import sys
import argparse
import torch.nn as nn

from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import random # Added
import pytest
import safetensors
import matplotlib.pyplot as plt
import numpy as np # Added
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInOutput
from transformers.models.llama.train_adaptive_llama import freeze_lm_backbone
import imageio  # Add imageio import
import io # Add io import

from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig

# --- Global variables for hook ---
pruned_input_ids_counts = None
total_initial_tokens = 0
total_pruned_tokens = 0

# --- Hook function ---
def pruning_hook(module, input, output):
    global total_initial_tokens, total_pruned_tokens, pruned_input_ids_counts

    input_ids = output.input_ids
    attention_mask = output.input_ids_attention_mask
    pruned_input_ids = input_ids.flatten()[(((~output.full_merging_map.bool().flatten(1)) & attention_mask.bool()).flatten().bool())]

    pruned_input_ids_counts += torch.bincount(pruned_input_ids, minlength=module.config.vocab_size)

    # Assuming the first element of input tuple is hidden_states
    # and the second is the attention_mask we need. Adjust if structure differs.
    output
    total_initial_tokens += output.merged_embeddings_counts.sum().item()
    total_pruned_tokens += output.attention_mask.sum().item()

def evaluate_ppl_wikitext_103(model):
    evaluation_output_dir = "'/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/exps_evaluation'" # Removed extra quotes
    evaluation_tracker = EvaluationTracker(
        output_dir=evaluation_output_dir,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(
            cache_dir='/workspace-SR004.nfs2/.cache/huggingface',
        ),
        # env_config=env_config,
        custom_tasks_directory='/workspace-SR004.nfs2/d.tarasov/cosmopedia/evaluation/lighteval_tasks.py',
        override_batch_size=1,
        num_fewshot_seeds=1,
        max_samples=None,
        use_chat_template=False,
        system_prompt=None,
        load_responses_from_details_date_id=None,
    )

    tasks = "custom|wikitext_103|0|1"

    with torch.no_grad():
        pipeline = Pipeline(
            tasks=tasks,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model=model,
        )
        pipeline.evaluate()

        pipeline.show_results()
        results = pipeline.get_results()

        ppl = results['results']["custom:wikitext_103:0"]["ppl"]

    return {
        "ppl": ppl,
    }

@torch.no_grad()
def evaluate_different_percents(model, percent_step=10):
    global total_initial_tokens, total_pruned_tokens, pruned_input_ids_counts

    assert percent_step > 0

    pruned_input_ids_counts = torch.zeros(model.config.vocab_size, dtype=torch.long)

    all_results = []

    hook_handle = None # Variable to store the hook handle

    for eval_hard_concrete_percent in range(0, 100, percent_step):
        eval_hard_concrete_percent = eval_hard_concrete_percent / 100.0
        model.config.eval_hard_concrete_percent = eval_hard_concrete_percent
        print(f"--- Evaluating with eval_hard_concrete_percent = {eval_hard_concrete_percent} ---")

        # Reset counters for the new evaluation percentage
        total_initial_tokens = 0
        total_pruned_tokens = 0

        # Ensure the target layer exists
        target_layer = model.model.fan_in
        hook_handle = target_layer.register_forward_hook(pruning_hook)
        print("Registered forward hook on model.model.fan_in")

        ppl_results = evaluate_ppl_wikitext_103(model)
        ppl = ppl_results['ppl']

        # Calculate pruned percentage
        if total_initial_tokens > 0:
            pruned_percent = (total_initial_tokens - total_pruned_tokens) / total_initial_tokens * 100
        else:
            pruned_percent = 0.0
            print("Warning: total_initial_tokens is zero. Cannot calculate pruned percentage.")

        print(f"Total Initial Tokens: {total_initial_tokens}")
        print(f"Total Pruned Tokens (Remaining): {total_pruned_tokens}")
        print(f"Pruned Percentage: {pruned_percent:.2f}%")

        all_results.append({
            "eval_hard_concrete_percent": eval_hard_concrete_percent,
            "ppl": ppl,
            "pruned_percent": pruned_percent,
        })

        if hook_handle:
            hook_handle.remove()
            print("Removed forward hook.")
            hook_handle = None

        if ppl > 100: # Threshold increased slightly as per original code comment intent
            print(f"PPL ({ppl}) exceeded threshold, stopping evaluation early.")
            break

    df = pd.DataFrame(all_results)
    df = df.sort_values(by='eval_hard_concrete_percent')
    print("Pruning Percent", df)
    # df.to_csv("eval_hard_concrete_percent_results.csv", index=False)

    # Create figure and axes for plots
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot PPL
    color = 'tab:red'
    ax1.set_xlabel('Eval Hard Concrete Percent')
    ax1.set_ylabel('Perplexity (PPL)', color=color)
    ax1.plot(df['eval_hard_concrete_percent'], df['ppl'], color=color, marker='o', label='Perplexity (PPL)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    # Create a second y-axis for Pruned Percent
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Pruned Tokens (%)', color=color)
    ax2.plot(df['eval_hard_concrete_percent'], df['pruned_percent'], color=color, marker='x', linestyle='--', label='Pruned Tokens (%)')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add title and legend
    plt.title('Perplexity and Pruned Token Percentage vs. Eval Hard Concrete Percent')
    fig.tight_layout() # Adjust layout to prevent overlap
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.show()
    plt.savefig(os.path.join(checkpoint_base_path, f"eval_hard_concrete_percent_results_nomralize.png"))
    print("Saved plot to eval_hard_concrete_percent_results.png")

    print("pruned_input_ids_counts", pruned_input_ids_counts)
    print("max", pruned_input_ids_counts.max(), "argmax", pruned_input_ids_counts.argmax())

    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    print("Top 10 frequently pruned tokens:")

    for i in pruned_input_ids_counts.argsort()[-10:]:
        token_string = tokenizer.decode(i).replace('\n', '\\n')
        print(f"Token {i} [{token_string}]: {pruned_input_ids_counts[i]}")

    pruned_input_ids_counts_file = os.path.join(checkpoint_base_path, f"pruned_input_ids_counts.pt")
    torch.save(pruned_input_ids_counts, pruned_input_ids_counts_file)
    print("Saved pruned_input_ids_counts to", pruned_input_ids_counts_file)

    return df

@torch.no_grad()
def analyze_most_confident_pruned_tokens(model, checkpoint_base_path):

    hcg_log_a = model.model.fan_in.hcg.hcg_log_a
    all_inputs = torch.arange(model.config.vocab_size, device=hcg_log_a.device, dtype=torch.long)
    all_inputs = all_inputs.unsqueeze(0)
    attention_mask = torch.ones_like(all_inputs, dtype=torch.long)

    most_confident_pruned_tokens = model.model.fan_in.hcg(all_inputs, attention_mask).squeeze(-1)

    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)

    pruned_tokens = (all_inputs[ most_confident_pruned_tokens == 0.0 ]).flatten()
    print("pruned_tokens", pruned_tokens.shape[0], 'vocab_size', model.config.vocab_size, 'pruned_percent {:.2f}%'.format(pruned_tokens.shape[0] / model.config.vocab_size * 100))

    print("Sample from pruned tokens:")
    for token in random.sample(pruned_tokens.tolist(), 10):
        print(f"Token {token} [{tokenizer.decode(token)}]: {hcg_log_a[token]}")

    return most_confident_pruned_tokens


@torch.no_grad()
def evaluate_different_layers(model, fan_in_idxs=None, fan_out_idxs=None, exp_prefix=None):

    results = []
    assert len(fan_in_idxs) == len(fan_out_idxs)

    for fan_in_idx, fan_out_idx in tqdm(zip(fan_in_idxs, fan_out_idxs), total=len(fan_in_idxs)):
        model.model.fan_in_idx = fan_in_idx
        model.model.fan_out_idx = fan_out_idx
        ppl_results = evaluate_ppl_wikitext_103(model)
        ppl = ppl_results['ppl']
        print(f"PPL for fan_in_idx={fan_in_idx} and fan_out_idx={fan_out_idx}: {ppl}")

        results.append({
            "fan_in_idx": fan_in_idx,
            "fan_out_idx": fan_out_idx,
            "ppl": ppl,
        })

    df = pd.DataFrame(results)

    fan_in_idxs_str = ','.join(map(str, fan_in_idxs))
    fan_out_idxs_str = ','.join(map(str, fan_out_idxs))
    if exp_prefix is not None:
        result_file_name = f"{exp_prefix}_ppl_results"
    else:
        result_file_name = f"_ppl_results_fan_in_idx_{fan_in_idxs_str}_fan_out_idx_{fan_out_idxs_str}"

    plt.plot(df['fan_in_idx'], df['ppl'], marker='o', label='PPL')
    plt.xlabel('Fan In Index')
    plt.ylabel('Perplexity (PPL)')
    plt.title('Perplexity vs. Fan In Index')
    plt.ylim(0, 20)
    plt.legend()
    plt.show()

    plot_file_path = os.path.join(checkpoint_base_path, result_file_name + ".png")
    plt.savefig(plot_file_path)
    print("Saved PPL results to", plot_file_path)

    output_file = os.path.join(model.name_or_path, result_file_name + ".csv")
    df.to_csv(output_file, index=False)
    print("Saved PPL results to", output_file)
    print("df", df)

    return df

if __name__ == "__main__":
    torch.set_default_device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_base_path", type=str, required=True)
    parser.add_argument("--analyze_most_confident_pruned_tokens", action='store_true', default=False)
    parser.add_argument("--percent_step", type=int, default=0)
    parser.add_argument("--normalize_hcg_log_a", action='store_true', default=False)
    parser.add_argument("--fan_in_idxs", default=None)
    parser.add_argument("--fan_out_idxs", default=None)
    parser.add_argument("--concrete_random_mask_proba", default=None, type=float)
    parser.add_argument("--exp_prefix", default=None, type=str)
    args = parser.parse_args()

    normalize_hcg_log_a = args.normalize_hcg_log_a
    checkpoint_base_path = args.checkpoint_base_path

    print("checkpoint_base_path", checkpoint_base_path)
    checkpoints = os.listdir(checkpoint_base_path)
    checkpoints = [x for x in checkpoints if x.startswith('checkpoint')]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

    if len(checkpoints) == 0:
        print("No checkpoints found. Assuming the checkpoint_base_path is a single checkpoint.")
        last_checkpoint_path = checkpoint_base_path
    else:
        last_checkpoint_path = os.path.join(checkpoint_base_path, checkpoints[-1])

    # last_checkpoint_path = "adaptive_hcg_slm2_1.7B_w_0.010_l_10_no_self_attn_5XMH6AH4/checkpoint-100000/"

    # --- Initialization for Random Token Evolution ---
    hcg_log_a_key = 'model.fan_in.hcg.hcg_log_a'

    # Load tokenizer and select random tokens from the first checkpoint
    print(f"Loading tokenizer and state from: {last_checkpoint_path}")

    tokeniser = AutoTokenizer.from_pretrained(last_checkpoint_path)
    model = AdaptiveLlamaForCausalLM.from_pretrained(last_checkpoint_path, torch_dtype=torch.bfloat16)
    model.eval()

    print("Model fan in idx  ", model.model.fan_in_idx)
    print("Model fan out idx ", model.model.fan_out_idx)

    if args.concrete_random_mask_proba is not None:
        model.config.concrete_random_mask_proba = float(args.concrete_random_mask_proba)
        print(f"Setting concrete_random_mask_proba to {args.concrete_random_mask_proba}")

    if normalize_hcg_log_a:
        print("Normalizing hcg log a")
        log_a = model.model.fan_in.hcg.hcg_log_a.data
        log_a = log_a / log_a.abs().max() * 5 # 5 is for sigmoid at least 0 or at least 1
        model.model.fan_in.hcg.hcg_log_a.data = log_a

    if args.percent_step > 0:
        evaluate_different_percents(model, percent_step=args.percent_step)
    elif args.analyze_most_confident_pruned_tokens:
        most_confident_pruned_tokens = analyze_most_confident_pruned_tokens(model, checkpoint_base_path=checkpoint_base_path)
    else:
        fan_in_idxs =  list(map(int, args.fan_in_idxs.split(',')))
        fan_out_idxs = list(map(int, args.fan_out_idxs.split(',')))
        evaluate_different_layers(model, fan_in_idxs=fan_in_idxs, fan_out_idxs=fan_out_idxs, exp_prefix=args.exp_prefix)

