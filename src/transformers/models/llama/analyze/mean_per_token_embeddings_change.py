import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import argparse
import logging

from transformers.models.llama.analyze.embeddings_change import compute_distances


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_checkpoint", type=str, required=True)
    parser.add_argument("--skip_layers", type=int, default=1)

    args = parser.parse_args()
    skip_layers = args.skip_layers

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {args.llama_checkpoint}")

    model_vanilla = AutoModelForCausalLM.from_pretrained(args.llama_checkpoint, torch_dtype=torch.bfloat16)
    model_vanilla.to(device)
    model_vanilla.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.llama_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    wikitext_103: datasets.Dataset = datasets.load_dataset("lighteval/wikitext_103", split="test")

    # token_id -> list[heatmaps]
    per_token_heatmaps = dict()

    batch_size = 4
    total_batches = len(wikitext_103) // batch_size

    for batch in tqdm(wikitext_103.iter(batch_size=batch_size), total=total_batches):
        texts = batch['text']

        model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
        model_inputs = model_inputs.to(device)

        seq_len = model_inputs['input_ids'].shape[1]

        # [ batch_size, seq_len ]
        model_inputs['input_ids'] = model_inputs['input_ids'][:, :seq_len]
        # [ batch_size, seq_len ]
        model_inputs['attention_mask'] = model_inputs['attention_mask'][:, :seq_len]

        outputs_vanilla = model_vanilla(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
        )

        heatmaps = {
            "cos": torch.ones(len(outputs_vanilla.hidden_states) - 1, seq_len, dtype=torch.float64),
            "l1": torch.ones(len(outputs_vanilla.hidden_states) - 1, seq_len, dtype=torch.float64),
        }

        for i, h_i in enumerate(outputs_vanilla.hidden_states[:-1]):
            h_i = h_i[:, :seq_len]

            position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)

            if i + skip_layers >= len(outputs_vanilla.hidden_states):
                break

            hs_j = outputs_vanilla.hidden_states[i+skip_layers]
            hs_j = hs_j[:, :seq_len]

            # Compute distances for each configuration
            cos_distance, l1_distance = compute_distances(
                h_i,
                hs_j,
                seq_len,
            )

            heatmaps["cos"][i, :] = cos_distance.to(torch.float64)
            heatmaps["l1"][i, :] = l1_distance.to(torch.float64)

        # Process each token in the batch considering attention mask
        batch_size = model_inputs['input_ids'].shape[0]
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                # Skip masked tokens
                if model_inputs['attention_mask'][batch_idx, seq_idx] == 0:
                    continue

                token_id = model_inputs['input_ids'][batch_idx, seq_idx].item()

                # Initialize entry for this token_id if it doesn't exist yet
                if token_id not in per_token_heatmaps:
                    per_token_heatmaps[token_id] = {
                        "cos": [],
                        "l1": [],
                        "count": 0
                    }

                # Add the heatmaps for this token occurrence
                per_token_heatmaps[token_id]["cos"].append(heatmaps["cos"][:, seq_idx])
                per_token_heatmaps[token_id]["l1"].append(heatmaps["l1"][:, seq_idx])
                per_token_heatmaps[token_id]["count"] += 1

    # Calculate mean heatmaps for each token
    for token_id, data in per_token_heatmaps.items():
        if data["count"] > 0:
            # Stack all collected heatmaps and compute mean
            cos_stack = torch.stack(data["cos"], dim=0)
            l1_stack = torch.stack(data["l1"], dim=0)

            data["mean_cos"] = torch.mean(cos_stack, dim=0)
            data["mean_l1"] = torch.mean(l1_stack, dim=0)

            # Optional: compute standard deviation for error bars
            data["std_cos"] = torch.std(cos_stack, dim=0)
            data["std_l1"] = torch.std(l1_stack, dim=0)

    # Save or process the results
    logger.info(f"Processed {len(per_token_heatmaps)} unique token ids")

    # Optional: save the results to disk
    output_dir = os.path.join("results", "token_embeddings_change")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(per_token_heatmaps, os.path.join(output_dir, f"token_heatmaps_{args.llama_checkpoint.split('/')[-1]}.pt"))
    logger.info(f"Saved results to {output_dir}")



