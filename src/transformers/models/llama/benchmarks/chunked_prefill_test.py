import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm.auto import tqdm

import torch
from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM, special_token_mask_to_clothest_token_idx_slow

from transformers import AutoTokenizer, DynamicCache

from transformers.models.llama.benchmarks.sentence_attention_bench import scrooge_prefill

LIPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."


if __name__ == "__main__":

    model_class = SentenceLlamaForCausalLM
    checkpoint_dir = "unsloth/Llama-3.2-1B-Instruct"

    model = model_class.from_pretrained(checkpoint_dir, torch_dtype=torch.float32)
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    model.config._attn_implementation = "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # Prefill
    # Peak Memory Usage, Time

    with torch.no_grad():

        model_inputs = tokenizer(LIPSUM, return_tensors="pt").to(device)

        full_outputs = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
        )


        partial_input_position = model_inputs['input_ids'].shape[1] // 2
        model_inputs_partial = {
            "input_ids": model_inputs["input_ids"][:, :partial_input_position],
            "attention_mask": model_inputs["attention_mask"][:, :partial_input_position],
        }

        past_key_values = DynamicCache()

        partial_outputs = model(
            input_ids=model_inputs_partial["input_ids"],
            attention_mask=model_inputs_partial["attention_mask"],
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
        )

        rest_input_ids = {
            "input_ids": model_inputs["input_ids"][:, partial_input_position:],
        }

        rest_outputs = model(
            input_ids=rest_input_ids["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )


        joined_hidden_states = torch.cat([partial_outputs.hidden_states[-1], rest_outputs.hidden_states[-1]], dim=1)

        full_hs = full_outputs.hidden_states[-1]

        torch.set_printoptions(linewidth = 30000, profile='full')
        diff = ((full_hs - joined_hidden_states) * 1000).round().mean(dim=-1)
        print(diff)

        assert torch.allclose(joined_hidden_states, full_hs, atol=1e-5), 'last hidden state should be the same'

        breakpoint()
