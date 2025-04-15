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
total_initial_tokens = 0
total_pruned_tokens = 0

# --- Hook function ---
def pruning_hook(module, input, output):
    global total_initial_tokens, total_pruned_tokens

    # Assuming the first element of input tuple is hidden_states
    # and the second is the attention_mask we need. Adjust if structure differs.
    output
    total_initial_tokens += output.merged_embeddings_counts.sum().item()
    total_pruned_tokens += output.attention_mask.sum().item()

if __name__ == "__main__":
    torch.set_default_device('cuda')

    import sys
    checkpoint_base_path = sys.argv[1]

    print("checkpoint_base_path", checkpoint_base_path)
    checkpoints = os.listdir(checkpoint_base_path)
    checkpoints = [x for x in checkpoints if x.startswith('checkpoint')]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

    if not checkpoints:
        print("No checkpoints found.")
        exit()

    # --- Initialization for Random Token Evolution ---
    num_random_tokens = 10
    random_token_indices = None
    random_token_strings = None
    token_prob_history = None # Will be {token_idx: []}
    evolution_images = [] # For the second animation
    # Define key upfront, try the standard one first
    hcg_log_a_key = 'model.fan_in.hcg.hcg_log_a'
    tokeniser = None
    vocab_size = None

    # Load tokenizer and select random tokens from the first checkpoint
    last_checkpoint_path = os.path.join(checkpoint_base_path, checkpoints[-1])
    print(f"Loading tokenizer and initial state from: {checkpoints[0]}")

    tokeniser = AutoTokenizer.from_pretrained(last_checkpoint_path)
    model = AdaptiveLlamaForCausalLM.from_pretrained(last_checkpoint_path, torch_dtype=torch.bfloat16)
    model.eval()

    all_results = []
    hook_handle = None # Variable to store the hook handle

    for eval_hard_concrete_percent in range(0, 11, 2): # Iterate up to 1.0
        eval_hard_concrete_percent = eval_hard_concrete_percent / 10.0
        model.config.eval_hard_concrete_percent = eval_hard_concrete_percent
        print(f"--- Evaluating with eval_hard_concrete_percent = {eval_hard_concrete_percent} ---")

        # Reset counters for the new evaluation percentage
        total_initial_tokens = 0
        total_pruned_tokens = 0

            # Ensure the target layer exists
        target_layer = model.model.fan_in
        hook_handle = target_layer.register_forward_hook(pruning_hook)
        print("Registered forward hook on model.model.fan_in")

        try:
            evaluation_output_dir = "'/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/exps_evaluation'" # Removed extra quotes
            evaluation_tracker = EvaluationTracker(
                output_dir=evaluation_output_dir,
            )
            pipeline_params = PipelineParameters(
                launcher_type=ParallelismManager.ACCELERATE,
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

                print("results", results)

                ppl = results['results']["custom:wikitext_103:0"]["ppl"]

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

                if ppl > 100: # Threshold increased slightly as per original code comment intent
                    print(f"PPL ({ppl}) exceeded threshold, stopping evaluation early.")
                    # Remove hook if loop breaks early
                    if hook_handle:
                        hook_handle.remove()
                        print("Removed forward hook.")
                        hook_handle = None
                    break
        except Exception as e:
            print(f"Error during evaluation for percentage {eval_hard_concrete_percent}: {e}")
        finally:
            # Ensure hook is removed after each percentage evaluation run
            if hook_handle:
                hook_handle.remove()
                print("Removed forward hook.")
                hook_handle = None


    df = pd.DataFrame(all_results)
    df = df.sort_values(by='eval_hard_concrete_percent')
    print("df", df)
    df.to_csv("eval_hard_concrete_percent_results.csv", index=False)

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
    plt.savefig("eval_hard_concrete_percent_results.png")
    print("Saved plot to eval_hard_concrete_percent_results.png")