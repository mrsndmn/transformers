import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm.auto import tqdm

import torch
from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM, special_token_mask_to_clothest_token_idx_slow

from transformers import AutoTokenizer, DynamicCache

from transformers.models.llama.benchmarks.sentence_attention_bench import scrooge_prefill


if __name__ == "__main__":

    model_class = SentenceLlamaForCausalLM
    checkpoint_dir = "./sentence_Llama-3.2-1B_pretrain_with_end_of_sentence_full_BTLCR6IG/checkpoint-2000"

    model = model_class.from_pretrained(checkpoint_dir, torch_dtype=torch.float32)
    model.eval()
    model.to("cuda")

    # model.config._attn_implementation = "sdpa"
    model.config._attn_implementation = "sentence_attention"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    special_token_id = tokenizer.end_of_sentence_token_id

    # Prefill
    # Peak Memory Usage, Time

    dataset = Dataset.load_from_disk('./fineweb_edu_tokenized_Llama-3.2-1B_with_eos_token/shard_9')
    dataset = dataset.select(range(1))

    for item in dataset:

        input_ids = torch.tensor(item["input_ids"], device="cuda").unsqueeze(0)[:, :]
        attention_mask = torch.tensor(item["attention_mask"], device="cuda")[:, :]
        special_embeddings_mask = input_ids == special_token_id
        clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask)

        print("total tokens", attention_mask.sum().item())

        rich_prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
            use_cache=False,
            output_hidden_states=True,
        )

        rich_hs = rich_prefill_outputs.hidden_states
        rich_hs_last = rich_hs[-1]

        scroodge_outputs = scrooge_prefill(model, input_ids, attention_mask, special_embeddings_mask, clothest_end_of_sentence_token_idx, trim_kv_cache=False)

        scroodge_hs_last = torch.cat([ x[-1] for x in scroodge_outputs["hidden_states"]], dim=1)

        print("rich_hs_last, scroodge_hs", (rich_hs_last - scroodge_hs_last).mean(dim=-1)[-200:])

        assert torch.allclose(rich_hs_last, scroodge_hs_last, atol=1e-3), 'last hidden state should be the same'

        breakpoint()
