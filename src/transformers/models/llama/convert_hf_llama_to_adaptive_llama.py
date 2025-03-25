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

from transformers import AutoModelForCausalLM, GenerationConfig, LlamaConfig, LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast, AutoConfig

from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInHCG

def build_adaptive_llama_from_llama_checkpoint(
        llama_checkpoint,
        dummy_adaptive_fan_in=None,
        generate_merges_transform_impl='python',
        fan_out_projection=True,
        merging_type='next_token_merge_mlp',
        freeze_lm_backbone=False,
        full_unmerge=None,
        fan_out_type=None,
        hcg_temperature=1.0,
        learnt_temperature=False,
        flash_attention=True,
        gumbel_tau=2.0,
        scale_not_pruned_gradients=0.0,
        concrete_random_mask_proba=None,
    ):

    torch_dtype = torch.bfloat16
    llama_model = AutoModelForCausalLM.from_pretrained(llama_checkpoint, torch_dtype=torch_dtype)
    llama_model_state_dict = llama_model.state_dict()

    config_kwargs = {}
    if flash_attention:
        config_kwargs["attn_implementation"] = 'flash_attention_2'

    config: LlamaConfig = AutoConfig.from_pretrained(llama_checkpoint, **config_kwargs)
    config.dummy_adaptive_fan_in = dummy_adaptive_fan_in
    config.generate_merges_transform_impl = generate_merges_transform_impl
    config.fan_out_projection = fan_out_projection
    config.merging_type = merging_type
    config.full_unmerge = full_unmerge
    config.fan_out_type = fan_out_type
    config.hcg_temperature = hcg_temperature
    config.learnt_temperature = learnt_temperature
    config.gumbel_tau = gumbel_tau
    config.scale_not_pruned_gradients = scale_not_pruned_gradients
    config.concrete_random_mask_proba = concrete_random_mask_proba
    if flash_attention:
        config._attn_implementation = 'flash_attention_2'

    num_hidden_layers = config.num_hidden_layers
    assert num_hidden_layers % 2 == 0
    half_num_hidden_layers = num_hidden_layers // 2

    dtype_orig = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    adaptive_llama_model = AdaptiveLlamaForCausalLM(config, attn_implementation='flash_attention_2')
    torch.set_default_dtype(dtype_orig)

    adaptive_llama_model_state_dict = adaptive_llama_model.state_dict()

    for param_name, param_value in llama_model_state_dict.items():
        param_name: str
        if param_name.startswith('model.layers.'):
            layer_num = int(param_name.split(".")[2])

            if layer_num < half_num_hidden_layers:
                adaptive_layer_num = layer_num
                param_name = param_name.replace(f"model.layers.{layer_num}", f"model.layers_down.{layer_num}")
            else:
                adaptive_layer_num = layer_num - half_num_hidden_layers
                param_name = param_name.replace(f"model.layers.{layer_num}", f"model.layers_up.{adaptive_layer_num}")
            # print("new param name:", param_name)

        adaptive_llama_model_state_dict[param_name] = param_value

    adaptive_llama_model.load_state_dict(adaptive_llama_model_state_dict)

    for adaptive_down in adaptive_llama_model.model.adaptive_down:
        if isinstance(adaptive_down, AdaptiveFanInHCG):
            adaptive_down.hcg.to(torch.float32)

    if freeze_lm_backbone:
        for p in adaptive_llama_model.parameters():
            p.requires_grad = False

        for p in adaptive_llama_model.model.adaptive_down.parameters():
            p.requires_grad = True

        for p in adaptive_llama_model.model.adaptive_up.parameters():
            p.requires_grad = True
            
    print("total parameters:", sum(p.numel() for p in adaptive_llama_model.parameters()))

    adaptive_llama_model._init_adaptive_layers()

    return adaptive_llama_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_llama",
        help="HF Llama checkpoint for weights conversion",
        default="HuggingFaceTB/SmolLM-135M"
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )

    args = parser.parse_args()

    llama_config = LlamaConfig.from_pretrained(args.from_llama)
    num_layers = llama_config.num_hidden_layers
    assert num_layers % 2 == 0

    dummy_adaptive_fan_in = [ True ] * (num_layers // 2)
    dummy_adaptive_fan_in[-1] = False
    print("dummy_adaptive_fan_in", dummy_adaptive_fan_in)
    model = build_adaptive_llama_from_llama_checkpoint(args.from_llama, dummy_adaptive_fan_in=dummy_adaptive_fan_in)

    llama_model = AutoModelForCausalLM.from_pretrained( args.from_llama )

    assert (llama_model.model.layers[0].mlp.gate_proj.weight == model.model.layers_down[0].mlp.gate_proj.weight).all()

    print(model)
    breakpoint()

if __name__ == "__main__":
    main()
