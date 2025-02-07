import time
import argparse

import torch

from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInHCG


from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint

def test_eval_adaptive_hcg_llama():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bench_dtype = torch.bfloat16

    model = LlamaForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")

    # checkpoint = './adaptive_gumbel_8/checkpoint-996/'
    checkpoint = 'adaptive_hcg_10/checkpoint-24996/'

    model = AdaptiveLlamaForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=bench_dtype,
    )

    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    text = "<|im_start|> The COVID-19 pandemic has"
    text_inputs = tokenizer([ text ], return_tensors='pt').to(device)

    special_embeddings_mask = torch.zeros_like(text_inputs['input_ids'])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1
    text_inputs['special_embeddings_mask'] = special_embeddings_mask
    text_inputs['labels'] = text_inputs['input_ids']

    with torch.no_grad():

        model.eval()
        eval_input = None
        eval_output = None
        def eval_hook(module, input, output):
            print("eval hook")
            global eval_input
            eval_input = input
            global eval_output
            eval_output = output
            return

        # model.model.adaptive_up[9].register_forward_hook(eval_hook)
        forward_output = model.forward(**text_inputs)

        print(forward_output['loss'])
        model.train()
        train_input = None
        train_output = None
        def train_hook(module, input, output):
            print("train hook")
            global train_input
            global train_output
            train_input = input
            train_output = output
            return

        # model.model.adaptive_up[9].register_forward_hook(train_hook)

        forward_output = model.forward(**text_inputs)
        print(forward_output['loss'])

    breakpoint()