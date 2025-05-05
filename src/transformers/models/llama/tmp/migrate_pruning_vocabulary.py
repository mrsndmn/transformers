
import torch
import argparse
from transformers import AutoTokenizer

from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInOutput
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--donor_checkpoint", type=str, required=True)
    parser.add_argument("--target_model_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    donor_checkpoint = args.donor_checkpoint
    target_model_checkpoint = args.target_model_checkpoint
    output_dir = args.output_dir
    donor_model = AdaptiveLlamaForCausalLM.from_pretrained(donor_checkpoint, torch_dtype=torch.bfloat16)

    target_model = build_adaptive_llama_from_llama_checkpoint(
        llama_checkpoint=target_model_checkpoint,
        dummy_adaptive_fan_in=donor_model.model.fan_in_idx,
        dummy_adaptive_fan_out=donor_model.model.fan_out_idx,
    )

    target_model.model.fan_in.hcg.hcg_log_a.data.copy_(donor_model.model.fan_in.hcg.hcg_log_a.data)

    target_model.save_pretrained(output_dir)
    print(f"Saved target model to {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(target_model_checkpoint)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved tokenizer to {output_dir}")
