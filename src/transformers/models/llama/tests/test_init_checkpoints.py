import torch
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers import AutoTokenizer, LlamaForCausalLM

from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS, GPT2TokenizerFast


def test_init_checkpoint_adaptive():
    checkpoint_path = "./paper_checkpoints/pretrain/adaptive_slm2_1.7B_random_init"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    device = 'cuda'

    adaptive_llama = AdaptiveLlamaForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    adaptive_llama.to(device)


    inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")
    inputs = inputs.to(device)

    outputs = adaptive_llama.forward(**inputs, labels=inputs['input_ids'])

    print("adaptive loss", outputs.loss)
    assert outputs.loss < 20

def test_init_checkpoint_adaptive_with_end_of_sentence_token():
    checkpoint_path = "./paper_checkpoints/pretrain/adaptive_slm2_1.7B_random_init"
    # checkpoint_path = "./adaptive_slm2_1.7B_pretrain_with_end_of_sentence_token_ftte_w_0.100_l_8-16_9U63C2LF/checkpoint-1000/"

    tokenizer = GPT2TokenizerFastEOS.from_pretrained(checkpoint_path)

    device = 'cuda'

    adaptive_llama = AdaptiveLlamaForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    # adaptive_llama = AdaptiveLlamaForCausalLM.from_pretrained(checkpoint_path)
    adaptive_llama.to(device)

    adaptive_llama.resize_token_embeddings(len(tokenizer))
    adaptive_llama.config.end_of_sentence_token_id = tokenizer.convert_tokens_to_ids('<end_of_sentence>')

    adaptive_llama.config.prune_all_except_end_of_sentence_token = True
    adaptive_llama.config.force_train_on_trimmed_embeddings = True

    adaptive_llama.config.fan_in_idx = 8
    adaptive_llama.config.fan_out_idx = 16

    adaptive_llama.model.recalc_fan_in_fan_out_idx()

    inputs = tokenizer("The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.", return_tensors="pt")
    inputs = inputs.to(device)

    outputs = adaptive_llama.forward(**inputs, labels=inputs['input_ids'])

    adaptive_llama.save_pretrained("./test_init_checkpoint_adaptive_with_end_of_sentence_token")

    print("adaptive with end of sentence token loss", outputs.loss)
    assert outputs.loss < 20



def test_init_checkpoint_vanilla():
    checkpoint_path = "./paper_checkpoints/pretrain/slm2_1.7B_random_init/"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    device = 'cuda'

    vanilla_llama = LlamaForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    vanilla_llama.to(device)

    inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")
    inputs = inputs.to(device)

    outputs = vanilla_llama.forward(**inputs, labels=inputs['input_ids'])

    print("vanilla loss", outputs.loss)
    assert outputs.loss < 20




def test_init_checkpoint_vanilla_16L():
    checkpoint_path = "./paper_checkpoints/pretrain/slm2_1.7B_random_init_16L/"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    device = 'cuda'

    vanilla_llama = LlamaForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    vanilla_llama.to(device)

    inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")
    inputs = inputs.to(device)

    outputs = vanilla_llama.forward(**inputs, labels=inputs['input_ids'])

    print("vanilla loss", outputs.loss)
    assert outputs.loss < 20


