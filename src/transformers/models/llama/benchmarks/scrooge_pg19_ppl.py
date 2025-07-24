import torch
from transformers import AutoTokenizer
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

    checkpoint_dir = "./sentence_Llama-3.2-1B_pretrain_with_end_of_sentence_full_BTLCR6IG/checkpoint-2000"

    model_class = SentenceLlamaForCausalLM
    model = model_class.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16)
    model.eval()
    model.to("cuda")
    # model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    special_token_id = tokenizer.end_of_sentence_token_id

    dataset = datasets.Dataset.load_from_disk("./pg19_test")


    tokens_log_probas = []
    samples_ppls = []

    per_sample_ppls = []

    with torch.no_grad():

        for item in tqdm(dataset):

            current_tokens_log_probas = []


            input_ids = tokenizer.encode(item["text"], return_tensors="pt", max_length=128000, truncation=True)
            input_ids = input_ids.to("cuda")
            attention_mask = torch.ones_like(input_ids).to("cuda")
            special_embeddings_mask = input_ids == special_token_id
            clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask)

            def outputs_hook(outputs, prev_sentence_i, sentence_i):
                outputs_logits_normed = F.log_softmax(outputs.logits.float(), dim=-1)

                if sentence_i == input_ids.shape[1]:
                    labels = input_ids[:, prev_sentence_i+1:sentence_i]
                    labels = labels.unsqueeze(-1)
                    log_probas = torch.gather(outputs_logits_normed[:, :-1, :], dim=-1, index=labels)
                else:
                    labels = input_ids[:, prev_sentence_i+1:sentence_i+1]
                    labels = labels.unsqueeze(-1)
                    log_probas = torch.gather(outputs_logits_normed, dim=-1, index=labels)

                # print("log_probas", log_probas.shape)
                current_tokens_log_probas.extend(log_probas[0, :, 0].cpu().numpy().tolist())

            outputs = scrooge_prefill(
                model,
                input_ids,
                attention_mask=attention_mask,
                special_embeddings_mask=special_embeddings_mask,
                clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
                outputs_hook=outputs_hook,
            )

            ppl = np.exp(-np.mean(current_tokens_log_probas))
            samples_ppls.append(ppl)
            per_sample_ppls.append(current_tokens_log_probas)
            print("Sample PPL", ppl)

            tokens_log_probas.extend(current_tokens_log_probas)


        ppl = np.exp(-np.mean(tokens_log_probas))
        print("Full PPL", ppl)
        print("Samples PPLs", np.mean(samples_ppls), 'std', np.std(samples_ppls))

        pd.DataFrame(samples_ppls).to_csv("./src/transformers/models/llama/benchmarks/data/pg19_samples_ppls.csv", index=False)

        with open("./src/transformers/models/llama/benchmarks/data/pg19_per_sample_ppls.pkl", "wb") as f:
            pickle.dump(per_sample_ppls, f)

        breakpoint()

        # plt.hist(tokens_counts, bins=100)
        # plt.savefig("./src/transformers/models/llama/benchmarks/plots/pg19_tokens_counts.png")