from transformers import AutoTokenizer

from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS, GPT2TokenizerFast


# src/transformers/models/llama/modeling_sentence_llama.py
from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM

import torch


def test_sentence_llama_model_generate():


    checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    input_ids = tokenizer.encode("Russia - Moscow. France - Paris. Germany - Berlin. Italy - ", return_tensors="pt")
    output = model.generate(
        input_ids,
        max_new_tokens=5,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=False)
    print(response)

    assert 'rome' in response.lower()


def test_sentence_llama_model_generate_with_eos_token():

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
    model = SentenceLlamaForCausalLM.from_pretrained(checkpoint)
    tokenizer = GPT2TokenizerFastEOS.from_pretrained(checkpoint)

    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to vocabulary size: {len(tokenizer)}")
    model.config.end_of_sentence_token_id = tokenizer.convert_tokens_to_ids('<end_of_sentence>')

    model.config._attn_implementation = 'sentence_attention'

    input_ids = tokenizer.encode("Russia - Moscow. France - Paris. Germany - Berlin. Italy - ", return_tensors="pt")
    assert (input_ids == model.config.end_of_sentence_token_id).sum().item() == 3

    print("input_ids", input_ids)
    output = model.generate(
        input_ids,
        max_new_tokens=5,
    )

