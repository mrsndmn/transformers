import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint

if __name__ == "__main__":
    llama_checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    output_dir = f"paper_checkpoints/pretrain/slm2_1.7B_random_init_16L"

    llama_config = LlamaConfig.from_pretrained(llama_checkpoint)
    llama_config.num_hidden_layers = 16
    llama_config.torch_dtype = torch.float32
    num_layers = llama_config.num_hidden_layers // 2
    dummy_adaptive_fan_in = [ True ] * num_layers
    dummy_adaptive_fan_in[-1] = False

    llama_model = LlamaForCausalLM(llama_config)

    tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)
    tokenizer.save_pretrained(output_dir)
    llama_model.save_pretrained(output_dir)

