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
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers.models.llama.train_adaptive_llama import freeze_lm_backbone
import imageio  # Add imageio import
import io # Add io import

from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig


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

    for eval_hard_concrete_percent in range(0, 10, 2):
        eval_hard_concrete_percent = eval_hard_concrete_percent / 10
        model.config.eval_hard_concrete_percent = eval_hard_concrete_percent
        try:
            evaluation_output_dir = "'/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/exps_evaluation'"
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

                all_results.append({
                    "eval_hard_concrete_percent": eval_hard_concrete_percent,
                    "ppl": results['results']["custom:wikitext_103:0"]["ppl"],
                    # todo pruned tokens percentage
                })

                if ppl > 100:
                    break
        except Exception as e:
            print("Error in evaluation of PPL", e)

    df = pd.DataFrame(all_results)
    df = df.sort_values(by='eval_hard_concrete_percent')
    print("df", df)
    df.to_csv("eval_hard_concrete_percent_results.csv", index=False)
    plt.plot( df['eval_hard_concrete_percent'], df['ppl'] )
    plt.gcf().set_size_inches(5, 5)
    plt.show()
    plt.savefig("eval_hard_concrete_percent_results.png")