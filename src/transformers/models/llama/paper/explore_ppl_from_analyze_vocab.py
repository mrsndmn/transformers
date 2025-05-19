
import pandas as pd
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
from transformers.models.qwen2.modeling_adaptive_qwen2 import AdaptiveQwen2ForCausalLM
import imageio  # Add imageio import
import io # Add io import

from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig

from transformers.models.llama.interpretation.explore_eval_hard_concrete_percent import evaluate_different_percents, analyze_most_confident_pruned_tokens, evaluate_different_layers, evaluate_ppl_wikitext_103, pruning_hook, reset_counters, get_counters


if __name__ == "__main__":
    torch.set_default_device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_base_path", type=str, required=True)
    parser.add_argument("--fan_in_idx", required=True, type=int)
    parser.add_argument("--fan_out_idx", required=True, type=int)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--fan_out_projection", default=None, type=int)
    # Multiple vocab csv files
    parser.add_argument("--vocab_csv_path", nargs="+", default=None, required=True)


    args = parser.parse_args()

    checkpoint_base_path = args.checkpoint_base_path
    print(f"Loading tokenizer and state from: {checkpoint_base_path}")

    tokeniser = AutoTokenizer.from_pretrained(checkpoint_base_path)

    device_map = None

    model = build_adaptive_llama_from_llama_checkpoint(
        checkpoint_base_path,
        hcg_log_a=10.0,
    )

    model.model.fan_in_idx = args.fan_in_idx
    model.model.fan_out_idx = args.fan_out_idx
    model.config.fan_out_projection = False

    fan_out_projection = None
    if args.fan_out_projection is not None:
        print("\n\nSetting fan_out_projection to", args.fan_out_projection)
        fan_out_projection = args.fan_out_projection > 0
        print("\n\nSetting fan_out_projection to", fan_out_projection)
        model.config.fan_out_projection = fan_out_projection

    print("Model fan in idx  ", model.model.fan_in_idx)
    print("Model fan out idx ", model.model.fan_out_idx)

    target_layer = model.model.fan_in
    hook_handle = target_layer.register_forward_hook(pruning_hook)

    for vocab_csv_path in args.vocab_csv_path:
        vocab_df = pd.read_csv(vocab_csv_path)
        vocab_t = torch.tensor(vocab_df['token_id'].tolist())

        print("Vocab", vocab_csv_path)
        print("Vocab_t", vocab_t.shape)

        model.model.fan_in.hcg.hcg_log_a.data[:] = 10.0
        model.model.fan_in.hcg.hcg_log_a.data[vocab_t] = -10.0

        reset_counters()

        metric_results = evaluate_ppl_wikitext_103(model, max_samples=args.max_samples)

        total_initial_tokens, total_pruned_tokens, pruned_input_ids_counts = get_counters()

        # TODO count pruned tokens
        print("metric_results", metric_results)
        print("total_initial_tokens", total_initial_tokens)
        print("total_pruned_tokens", total_pruned_tokens)
        print("pruned percent", (total_initial_tokens - total_pruned_tokens) / total_initial_tokens)

    # End loop
    breakpoint()


