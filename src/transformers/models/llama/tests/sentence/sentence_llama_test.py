from transformers import AutoTokenizer

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
