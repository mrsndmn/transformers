
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
    parser.add_argument("--vocab_csv_path", default=None, required=True)
    parser.add_argument("--output_dir", default=None, required=True)

    args = parser.parse_args()

    checkpoint_base_path = args.checkpoint_base_path
    print(f"Loading tokenizer and state from: {checkpoint_base_path}")

    tokeniser = AutoTokenizer.from_pretrained(checkpoint_base_path)

    device_map = None

    model = build_adaptive_llama_from_llama_checkpoint(
        checkpoint_base_path,
        hcg_log_a=10.0,
    )

    model.config.fan_in_idx = args.fan_in_idx
    model.config.fan_out_idx = args.fan_out_idx
    model.config.fan_out_projection = False
    model.model.recalc_fan_in_fan_out_idx()

    print("Model fan in idx  ", model.model.fan_in_idx)
    print("Model fan out idx ", model.model.fan_out_idx)

    print("Model fan in idx  ", model.model.fan_in_idx)
    print("Model fan out idx ", model.model.fan_out_idx)

    print("Vocab", args.vocab_csv_path)
    vocab_df = pd.read_csv(args.vocab_csv_path)
    vocab_t = torch.tensor(vocab_df['token_id'].tolist(), dtype=torch.long)
    print("Vocab_t", vocab_t.shape)

    model.model.fan_in.hcg.hcg_log_a.data[:] = 10.0
    model.model.fan_in.hcg.hcg_log_a.data[vocab_t] = -10.0

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=False)

    tokeniser.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


