import torch
from transformers import LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint

if __name__ == "__main__":
    llama_checkpoint = './paper_checkpoints/pretrain/adaptive_slm2_135M_random_init'

    llama_model_from_checkpoint = AutoModelForCausalLM.from_pretrained(llama_checkpoint, torch_dtype=torch.bfloat16, device_map='cpu')

    llama_config = LlamaConfig.from_pretrained(llama_checkpoint)
    llama_config.num_hidden_layers = llama_model_from_checkpoint.config.num_hidden_layers - 10

    dummy_adaptive_fan_in = [ True ] * (llama_config.num_hidden_layers // 2)
    dummy_adaptive_fan_in[-1] = False
    llama_config.dummy_adaptive_fan_in = dummy_adaptive_fan_in

    llama_model = AutoModelForCausalLM.from_config(llama_config)

    new_state_dict = {}
    state_dict = llama_model_from_checkpoint.state_dict()
    for key in list(state_dict.keys()):
        if key.startswith('model.layers.'):
            layer_idx = int(key.removeprefix('model.layers.').split('.')[0])
            if layer_idx < 10:
                new_state_dict[key] = state_dict[key]
            elif 10 <= layer_idx < 20:
                # skip current layer
                pass
            else:
                new_key = key.replace(f'.{layer_idx}.', f'.{layer_idx - 10}.')
                new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]

    llama_model.load_state_dict(new_state_dict, strict=True)

    print("trainable params", sum(p.numel() for p in llama_model.parameters() if p.requires_grad))

    tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)

    output_dir = f"./paper_checkpoints/pretrain/slm2_127M_random_init_20L"

    llama_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    test_loaded_model =AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map='cpu')