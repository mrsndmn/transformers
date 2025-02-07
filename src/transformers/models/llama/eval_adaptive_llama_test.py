import time
import argparse

import torch

from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInHCG


from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint

def test_eval_adaptive_hcg_llama():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bench_dtype = torch.bfloat16
    torch.set_default_dtype(bench_dtype)
    torch.set_default_device(device)

    model_orig = LlamaForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")

    # checkpoint = './adaptive_gumbel_8/checkpoint-996/'
    checkpoint = 'adaptive_gumbel_pretrain_8/checkpoint-3996'

    model = AdaptiveLlamaForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=bench_dtype,
    )

    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    text = "<|im_start|> The COVID-19 pandemic has who are you?"
    text_inputs = tokenizer([ text ], return_tensors='pt').to(device)

    special_embeddings_mask = torch.zeros_like(text_inputs['input_ids'])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1
    text_inputs['special_embeddings_mask'] = special_embeddings_mask
    text_inputs['labels'] = text_inputs['input_ids']

    with torch.no_grad():

        model.eval()
        eval_output = model.forward(**text_inputs)
        print(eval_output['loss'])

        model.train()
        train_output = model.forward(**text_inputs)
        print(train_output['loss'])

        del text_inputs['special_embeddings_mask']
        orig_output = model_orig.forward(**text_inputs)
        print("orig_output", orig_output['loss'])

    breakpoint()