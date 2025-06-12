# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gc
import json
import os
import shutil
import warnings
from typing import List

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast, AutoConfig

from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInHCG, AdaptiveLlamaForCausalLMWithEachLayerPruning
from transformers.models.qwen2.modeling_adaptive_qwen2 import AdaptiveQwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config

def build_adaptive_llama_from_llama_checkpoint(
        llama_checkpoint,
        dummy_adaptive_fan_in=None,
        generate_merges_transform_impl='python', # cuda_kernel
        fan_out_projection=True,
        merging_type='hcg',
        hcg_temperature=1.0,
        hcg_log_a=1.0,
        single_layer_hopping=False,
        learnt_temperature=False,
        flash_attention=True,
        scale_not_pruned_gradients=0.0,
        concrete_random_mask_proba=None,
        concrete_uniform_pruning=None,
        concrete_stop_word_pruning=None,
        pretrain_fan_out_projection=False,
        hcg_fan_in_from=None,
        adaptive_model_class=None,
        each_layer_pruning=False,
        fan_out_projection_mlp_intermediate_size=None,
    ):

    if adaptive_model_class is None:
        adaptive_model_class = AdaptiveLlamaForCausalLM
        if each_layer_pruning:
            adaptive_model_class = AdaptiveLlamaForCausalLMWithEachLayerPruning

        if "qwen" in llama_checkpoint.lower():
            adaptive_model_class = AdaptiveQwen2ForCausalLM

    torch_dtype = torch.bfloat16
    llama_model = AutoModelForCausalLM.from_pretrained(llama_checkpoint, torch_dtype=torch_dtype, device_map='cpu')
    llama_model_state_dict = llama_model.state_dict()

    config_kwargs = {}
    if flash_attention:
        config_kwargs["attn_implementation"] = 'flash_attention_2'

    config = AutoConfig.from_pretrained(llama_checkpoint, **config_kwargs)

    if dummy_adaptive_fan_in is not None:
        assert len(dummy_adaptive_fan_in) == config.num_hidden_layers // 2
    else:
        dummy_adaptive_fan_in = [ True ] * (config.num_hidden_layers // 2)
        dummy_adaptive_fan_in[-1] = False

    config.dummy_adaptive_fan_in = dummy_adaptive_fan_in
    config.generate_merges_transform_impl = generate_merges_transform_impl
    config.fan_out_projection = fan_out_projection
    config.merging_type = merging_type
    config.hcg_temperature = hcg_temperature
    config.hcg_log_a = hcg_log_a
    config.learnt_temperature = learnt_temperature
    config.scale_not_pruned_gradients = scale_not_pruned_gradients
    config.concrete_random_mask_proba = concrete_random_mask_proba
    config.pretrain_fan_out_projection = pretrain_fan_out_projection
    config.concrete_uniform_pruning = concrete_uniform_pruning
    config.concrete_stop_word_pruning = concrete_stop_word_pruning
    config.single_layer_hopping = single_layer_hopping

    if fan_out_projection_mlp_intermediate_size is not None:
        config.fan_out_projection_mlp_intermediate_size = fan_out_projection_mlp_intermediate_size

    if flash_attention:
        config._attn_implementation = 'flash_attention_2'

    num_hidden_layers = config.num_hidden_layers
    assert num_hidden_layers % 2 == 0

    dtype_orig = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    adaptive_llama_model = adaptive_model_class(config)
    torch.set_default_dtype(dtype_orig)

    adaptive_llama_model_state_dict = adaptive_llama_model.state_dict()

    for param_name, param_value in llama_model_state_dict.items():
        adaptive_llama_model_state_dict[param_name] = param_value

    if hcg_fan_in_from is not None:
        hcg_fan_in_from_model = adaptive_model_class.from_pretrained(hcg_fan_in_from, device_map='cpu')
        hcg_fan_in_from_model_state_dict = hcg_fan_in_from_model.state_dict()

        hcg_log_a_param_name = 'model.fan_in.hcg.hcg_log_a'
        hcg_log_a_param_value = hcg_fan_in_from_model_state_dict[hcg_log_a_param_name]
        adaptive_llama_model_state_dict[hcg_log_a_param_name] = hcg_log_a_param_value

    adaptive_llama_model.load_state_dict(adaptive_llama_model_state_dict)
    print("total parameters:", sum(p.numel() for p in adaptive_llama_model.parameters()))

    return adaptive_llama_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        help="Model type",
        default="llama",
    )

    parser.add_argument(
        "--from_llama",
        help="HF Llama checkpoint for weights conversion",
        default="HuggingFaceTB/SmolLM2-135M"
    )
    parser.add_argument(
        "--hcg_fan_in_from",
        help="Checkpoint from which to extract weights for fan-in hcg",
        default=None,
    )
    parser.add_argument(
        "--override_fan_in_layer_idx",
        help="Override fan in layer index",
        default=None,
    )
    parser.add_argument(
        "--override_fan_out_layer_idx",
        help="Override fan in layer index",
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--fan_in_layer_idx",
        help="Fan in layer index",
        type=int,
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )

    args = parser.parse_args()

    model_type = args.model_type

    if model_type == "llama":
        adaptive_model_class = AdaptiveLlamaForCausalLM
    elif model_type == "qwen2":
        adaptive_model_class = AdaptiveQwen2ForCausalLM
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    llama_config = AutoConfig.from_pretrained(args.from_llama)
    num_layers = llama_config.num_hidden_layers
    assert num_layers % 2 == 0

    dummy_adaptive_fan_in = [ True ] * (num_layers // 2)

    if args.fan_in_layer_idx is not None:
        dummy_adaptive_fan_in[args.fan_in_layer_idx] = False
    else:
        dummy_adaptive_fan_in[-1] = False

    print("dummy_adaptive_fan_in", dummy_adaptive_fan_in)
    model = build_adaptive_llama_from_llama_checkpoint(
        args.from_llama,
        dummy_adaptive_fan_in=dummy_adaptive_fan_in,
        hcg_fan_in_from=args.hcg_fan_in_from,
        adaptive_model_class=adaptive_model_class,
    )

    # llama_model = AutoModelForCausalLM.from_pretrained( args.from_llama )

    # assert (llama_model.model.layers[0].mlp.gate_proj.weight == model.model.layers_down[0].mlp.gate_proj.weight).all()

    if args.override_fan_in_layer_idx is not None:
        model.config.fan_in_idx = args.override_fan_in_layer_idx

    if args.override_fan_out_layer_idx is not None:
        model.config.fan_out_idx = args.override_fan_out_layer_idx

    print(model)
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.from_llama)
    tokenizer.save_pretrained(args.output_dir)

    breakpoint()

if __name__ == "__main__":
    main()
