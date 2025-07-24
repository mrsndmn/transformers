import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import numpy as np
import torch.nn.functional as F
import pickle

from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM
from transformers.models.llama.benchmarks.sentence_attention_bench import scrooge_prefill, special_token_mask_to_clothest_token_idx_slow

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

if __name__ == "__main__":

    checkpoint = 'unsloth/Llama-3.2-1B'

    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    model.eval()
    model.to("cuda")

    model.config._attn_implementation = 'flash_attention_2'

    # model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    dataset = datasets.Dataset.load_from_disk("./pg19_test")

    tokens_log_probas = []
    samples_ppls = []

    per_sample_ppls = []

    with torch.no_grad():

        for item in tqdm(dataset):

            current_tokens_log_probas = []


            input_ids = tokenizer.encode(item["text"], return_tensors="pt", max_length=60000, truncation=True)
            input_ids = input_ids.to("cuda")

            # print("input_ids.shape", input_ids.shape)
            outputs = model(input_ids, use_cache=False)

            # chunked log softmax
            logits = F.log_softmax(outputs.logits.float(), dim=-1)

            logits_gathered = torch.gather(logits[:, :-1], dim=-1, index=input_ids[:, 1:].unsqueeze(-1))

            current_tokens_log_probas = logits_gathered[0, :, 0].cpu().numpy().tolist()

            ppl = np.exp(-np.mean(current_tokens_log_probas))
            samples_ppls.append(ppl)
            per_sample_ppls.append(current_tokens_log_probas)
            print("Sample PPL", ppl)

            tokens_log_probas.extend(current_tokens_log_probas)

            del outputs, logits, logits_gathered
            torch.cuda.empty_cache()

            # breakpoint()


        ppl = np.exp(-np.mean(tokens_log_probas))
        print("Full PPL", ppl)
        print(f"Samples PPLs {np.mean(samples_ppls):.2f} std {np.std(samples_ppls):.2f}")

        pd.DataFrame(samples_ppls).to_csv("./src/transformers/models/llama/benchmarks/data/pg19_samples_ppls_full.csv", index=False)

        with open("./src/transformers/models/llama/benchmarks/data/pg19_per_sample_ppls_full.pkl", "wb") as f:
            pickle.dump(per_sample_ppls, f)

        breakpoint()

        # plt.hist(tokens_counts, bins=100)
        # plt.savefig("./src/transformers/models/llama/benchmarks/plots/pg19_tokens_counts.png")