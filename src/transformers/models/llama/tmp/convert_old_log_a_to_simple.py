import argparse
import torch
import safetensors
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint
from transformers.models.llama.modeling_llama import LlamaForCausalLM

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--old_log_a_weights", type=str, required=True)
    args = parser.parse_args()


    old_log_a_weights = args.old_log_a_weights
    # old_log_a_weights = "adaptive_hcg_slm2_360M_w_0.0_l_12_VXZEU1DJ/checkpoint-249974/model.safetensors"

    old_log_a_weights_state_dict = safetensors.torch.load_file(old_log_a_weights)

    checkpoint = 'HuggingFaceTB/SmolLM2-360M'
    dummy_adaptive_fan_in = [ True ] * 12
    dummy_adaptive_fan_in[11] = False
    model = build_adaptive_llama_from_llama_checkpoint(
        checkpoint,
        dummy_adaptive_fan_in=dummy_adaptive_fan_in,
        generate_merges_transform_impl='cuda_kernel',
        fan_out_projection=True,
        merging_type='hcg',
        hcg_temperature=0.33,
        hcg_log_a=100.0,
        learnt_temperature=False,
        flash_attention=True,
        scale_not_pruned_gradients=0.0,
        concrete_random_mask_proba=None,
        concrete_uniform_pruning=None,
        pretrain_fan_out_projection=False,
    )

    model.model.fan_in.hcg.hcg_log_a.data = old_log_a_weights_state_dict['model.adaptive_down.11.hcg.hcg_log_a']

    save_path = old_log_a_weights.replace('.safetensors', '_log_a_converted')
    model.save_pretrained(save_path)

    print("converted model save to", save_path)
