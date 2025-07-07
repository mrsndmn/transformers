import os
import shutil
import torch
from transformers import LlamaConfig, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint

if __name__ == "__main__":


    # llama_checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    # llama_checkpoint = "HuggingFaceTB/SmolLM2-135M"

    # llama_checkpoint = "HuggingFaceTB/SmolLM2-135M"
    # output_dir = f"paper_checkpoints/pretrain/slm2_135M_random_init"
    # adaptive_output_dir = f"paper_checkpoints/pretrain/adaptive_slm2_135M_random_init"

    llama_checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    output_dir          = f"./paper_checkpoints/pretrain2/slm2_1.7B_random_init"
    adaptive_output_dir = f"./paper_checkpoints/pretrain2/adaptive_slm2_1.7B_random_init"
    output_dir16L       = f"./paper_checkpoints/pretrain2/slm2_1.7B_random_init_16L"

    for output_dir_i in [output_dir, adaptive_output_dir, output_dir16L]:
        if os.path.exists(output_dir_i):
            shutil.rmtree(output_dir_i)
        os.makedirs(output_dir_i)

    llama_config = LlamaConfig.from_pretrained(llama_checkpoint)
    num_layers = llama_config.num_hidden_layers // 2
    dummy_adaptive_fan_in = [ True ] * num_layers
    dummy_adaptive_fan_in[10] = False

    llama_config.torch_dtype = torch.float32

    torch_dtype = torch.float32

    torch.set_default_dtype(torch_dtype)
    llama_model = LlamaForCausalLM(llama_config)
    torch.set_default_dtype(torch.float32)

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


    llama_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


    model.save_pretrained(adaptive_output_dir)
    tokenizer.save_pretrained(adaptive_output_dir)


    llama_config = LlamaConfig.from_pretrained(llama_checkpoint)
    llama_config.num_hidden_layers = 16
    llama_config.torch_dtype = torch.float32
    num_layers = llama_config.num_hidden_layers // 2
    dummy_adaptive_fan_in = [ True ] * num_layers
    dummy_adaptive_fan_in[-1] = False
    llama_config.dummy_adaptive_fan_in = dummy_adaptive_fan_in

    torch.set_default_dtype(torch_dtype)
    llama_model16L = LlamaForCausalLM(llama_config)
    torch.set_default_dtype(torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)
    tokenizer.save_pretrained(output_dir16L)
    llama_model16L.save_pretrained(output_dir16L)

