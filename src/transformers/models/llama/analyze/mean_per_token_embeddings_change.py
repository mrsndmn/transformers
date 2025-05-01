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
import gc

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
    parser.add_argument("--save_embeddings", action="store_true", help="Whether to save raw embeddings data")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Maximum number of tokens to save raw embeddings for")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--save_interval", type=int, default=100, help="Save intermediate results every N batches")
    parser.add_argument("--trim_quantile", type=float, default=0.0, help="Trim this quantile of outliers from each embedding before computing distances")

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

    print("len wikitext_103", len(wikitext_103))
    wikitext_103 = wikitext_103.select(range(30))

    # token_id -> list[heatmaps]
    per_token_heatmaps = dict()

    # Dictionary to store raw embeddings if requested
    if args.save_embeddings:
        per_token_embeddings = dict()

    batch_size = args.batch_size
    total_batches = len(wikitext_103) // batch_size

    # Create output directories
    output_dir = os.path.join("results", "token_embeddings_change")
    os.makedirs(output_dir, exist_ok=True)

    # For intermediate savings
    checkpoint_path = os.path.join(output_dir, f"token_heatmaps_{args.llama_checkpoint.split('/')[-1]}_checkpoint.pt")

    # Function to trim outliers from embeddings
    def trim_embeddings(h_i, hs_j, quantile):
        if quantile <= 0 or quantile >= 0.5:
            return h_i, hs_j

        # Create a copy to avoid modifying the original tensors
        h_i_trimmed = h_i.clone()
        hs_j_trimmed = hs_j.clone()

        h_i_float = h_i.float()
        hs_j_float = hs_j.float()

        lower_bound_h_i = torch.quantile(h_i_float, quantile, dim=-1)
        upper_bound_h_i = torch.quantile(h_i_float, 1.0 - quantile, dim=-1)

        lower_bound_hs_j = torch.quantile(hs_j_float, quantile, dim=-1)
        upper_bound_hs_j = torch.quantile(hs_j_float, 1.0 - quantile, dim=-1)

        h_i_trimmed[h_i_trimmed < lower_bound_h_i.unsqueeze(-1)] = 0
        h_i_trimmed[h_i_trimmed > upper_bound_h_i.unsqueeze(-1)] = 0

        hs_j_trimmed[hs_j_trimmed < lower_bound_hs_j.unsqueeze(-1)] = 0
        hs_j_trimmed[hs_j_trimmed > upper_bound_hs_j.unsqueeze(-1)] = 0

        return h_i_trimmed, hs_j_trimmed

    batch_count = 0
    for batch in tqdm(wikitext_103.iter(batch_size=batch_size), total=total_batches):
        batch_count += 1
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
            "cos": torch.ones(len(outputs_vanilla.hidden_states) - 1, seq_len, dtype=torch.float64).to("cpu"),
            "l1": torch.ones(len(outputs_vanilla.hidden_states) - 1, seq_len, dtype=torch.float64).to("cpu"),
        }

        for i, h_i in enumerate(outputs_vanilla.hidden_states[:-1]):
            h_i = h_i[:, :seq_len]

            position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)

            if i + skip_layers >= len(outputs_vanilla.hidden_states):
                break

            hs_j = outputs_vanilla.hidden_states[i+skip_layers]
            hs_j = hs_j[:, :seq_len]

            # Trim outliers from embeddings if requested
            if args.trim_quantile > 0:
                h_i_trimmed, hs_j_trimmed = trim_embeddings(h_i, hs_j, args.trim_quantile)
            else:
                h_i_trimmed, hs_j_trimmed = h_i, hs_j

            # Compute distances for each configuration
            cos_distance, l1_distance = compute_distances(
                h_i_trimmed,
                hs_j_trimmed,
                seq_len,
            )

            heatmaps["cos"][i, :] = cos_distance.to(torch.float64).to("cpu")
            heatmaps["l1"][i, :] = l1_distance.to(torch.float64).to("cpu")

        # Process each token in the batch considering attention mask
        batch_size = model_inputs['input_ids'].shape[0]

        if args.save_embeddings:
            # Pre-extract and move all hidden states to CPU to avoid memory issues
            cpu_hidden_states = []
            for layer_hidden in outputs_vanilla.hidden_states:
                cpu_hidden_states.append(layer_hidden.detach().cpu())

            # Free GPU memory
            del outputs_vanilla
            torch.cuda.empty_cache()

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

                # Store raw embeddings if requested and we haven't stored too many tokens yet
                if args.save_embeddings and len(per_token_embeddings) < args.max_tokens:
                    if token_id not in per_token_embeddings:
                        per_token_embeddings[token_id] = {
                            "layer_embeddings": [],
                            "count": 0
                        }

                    # Extract all layer embeddings for this token
                    token_embeddings = []
                    for layer_idx, layer_hidden in enumerate(cpu_hidden_states):
                        # Get the embedding vector for this token at this layer - already on CPU
                        emb = layer_hidden[batch_idx, seq_idx]
                        token_embeddings.append(emb)

                    # Store the list of embeddings for all layers
                    per_token_embeddings[token_id]["layer_embeddings"].append(token_embeddings)
                    per_token_embeddings[token_id]["count"] += 1

        # Free memory
        if args.save_embeddings:
            del cpu_hidden_states
        else:
            del outputs_vanilla

        model_inputs = model_inputs.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        # Save intermediate results every N batches
        if args.save_interval > 0 and batch_count % args.save_interval == 0:
            logger.info(f"Saving intermediate results after {batch_count} batches...")

            # Save heatmaps
            torch.save(per_token_heatmaps, checkpoint_path)

            # Also save embeddings if we're collecting them
            if args.save_embeddings:
                embeddings_checkpoint_path = os.path.join(output_dir, f"token_embeddings_{args.llama_checkpoint.split('/')[-1]}_checkpoint.pt")
                torch.save(per_token_embeddings, embeddings_checkpoint_path)

    # Calculate mean heatmaps for each token
    logger.info("Computing statistics for collected data...")
    for token_id, data in tqdm(per_token_heatmaps.items(), desc="Processing heatmaps"):
        if data["count"] > 0:
            # Stack all collected heatmaps and compute mean
            cos_stack = torch.stack(data["cos"], dim=0)
            l1_stack = torch.stack(data["l1"], dim=0)

            data["mean_cos"] = torch.mean(cos_stack, dim=0)
            data["mean_l1"] = torch.mean(l1_stack, dim=0)

            # Optional: compute standard deviation for error bars
            data["std_cos"] = torch.std(cos_stack, dim=0)
            data["std_l1"] = torch.std(l1_stack, dim=0)

            # Free memory by replacing the raw lists with their statistics
            data["cos"] = None
            data["l1"] = None

    # Calculate mean embeddings for each token if requested
    if args.save_embeddings:
        logger.info("Computing embedding statistics...")
        for token_id, data in tqdm(per_token_embeddings.items(), desc="Processing embeddings"):
            if data["count"] > 0:
                # For each layer, compute mean embedding across all occurrences
                mean_embeddings = []
                std_embeddings = []

                # Rearrange the data to get all embeddings for each layer
                num_layers = len(data["layer_embeddings"][0])
                layer_embeddings = [[] for _ in range(num_layers)]

                for token_occurrence in data["layer_embeddings"]:
                    for layer_idx, layer_emb in enumerate(token_occurrence):
                        layer_embeddings[layer_idx].append(layer_emb)

                # Compute mean and std for each layer
                for layer_idx in range(num_layers):
                    layer_embs = torch.stack(layer_embeddings[layer_idx], dim=0)
                    layer_embs = layer_embs.to('cuda')
                    mean_embeddings.append(torch.mean(layer_embs, dim=0).to('cpu'))
                    std_embeddings.append(torch.std(layer_embs, dim=0).to('cpu'))

                data["mean_embeddings"] = mean_embeddings
                data["std_embeddings"] = std_embeddings

                # Compute embedding differences between consecutive layers
                data["embedding_diffs"] = []
                for i in range(len(mean_embeddings) - 1):
                    data["embedding_diffs"].append(mean_embeddings[i+1] - mean_embeddings[i])

                # Free memory
                # data["layer_embeddings"] = None

    # Save or process the results
    logger.info(f"Processed {len(per_token_heatmaps)} unique token ids")

    # Save the results to disk
    output_path = os.path.join(output_dir, f"token_heatmaps_{args.llama_checkpoint.split('/')[-1]}.pt")
    torch.save(per_token_heatmaps, output_path)
    logger.info(f"Saved heatmaps to {output_path}")

    # Save raw embeddings if requested
    if args.save_embeddings:
        embeddings_path = os.path.join(output_dir, f"token_embeddings_{args.llama_checkpoint.split('/')[-1]}.pt")
        torch.save(per_token_embeddings, embeddings_path)
        logger.info(f"Saved raw embeddings for {len(per_token_embeddings)} tokens to {embeddings_path}")



