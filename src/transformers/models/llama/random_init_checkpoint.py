import torch
from transformers import LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint

if __name__ == "__main__":


    # llama_checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    # llama_checkpoint = "HuggingFaceTB/SmolLM2-135M"
    llama_checkpoint = "HuggingFaceTB/SmolLM2-135M"

    llama_config = LlamaConfig.from_pretrained(llama_checkpoint)
    num_layers = llama_config.num_hidden_layers // 2
    dummy_adaptive_fan_in = [ True ] * num_layers
    dummy_adaptive_fan_in[10] = False

    torch_dtype = torch.bfloat16

    llama_model = AutoModelForCausalLM.from_pretrained(llama_checkpoint, torch_dtype=torch_dtype, device_map='cpu')
    llama_model_state_dict = llama_model.state_dict()

    model = build_adaptive_llama_from_llama_checkpoint(
        llama_checkpoint,
        dummy_adaptive_fan_in=dummy_adaptive_fan_in,
        fan_out_projection=False,
        hcg_log_a=1.0,
        torch_dtype=torch_dtype,
        llama_model_state_dict=llama_model_state_dict,
    )

    tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)

    print("tokenizer", len(tokenizer))

    output_dir = f"paper_checkpoints/pretrain/slm2_135M_random_init"

    llama_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    output_dir = f"paper_checkpoints/pretrain/adaptive_slm2_135M_random_init"

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
