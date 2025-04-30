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
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

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

    batch_size = args.batch_size
    total_batches = len(wikitext_103) // batch_size

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

        # Analyze hidden states distribution
        # hidden_states is a tuple with (input_embeds, layer_0, layer_1, ..., layer_n)
        hidden_states = outputs_vanilla.hidden_states

        # Create directory for plots if it doesn't exist
        os.makedirs("hidden_states_analysis", exist_ok=True)

        # Analyze each layer's hidden states
        for layer_idx, layer_states in enumerate(hidden_states):
            layer_name = "input_embeds" if layer_idx == 0 else f"layer_{layer_idx-1}"
            # Convert to float32 numpy array for analysis
            layer_data = layer_states.detach().cpu().float().numpy()

            # Calculate statistics
            mean_val = np.mean(layer_data)
            median_val = np.median(layer_data)
            std_val = np.std(layer_data)
            min_val = np.min(layer_data)
            max_val = np.max(layer_data)

            # Define outliers as values outside 3 standard deviations from the mean
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val

            # Count outliers
            outliers = np.logical_or(layer_data < lower_bound, layer_data > upper_bound)
            outlier_count = np.sum(outliers)
            outlier_percentage = 100 * outlier_count / layer_data.size

            # Data without outliers
            data_no_outliers = layer_data[~outliers]

            # Print statistics
            print(f"\n{layer_name} statistics:")
            print(f"Mean: {mean_val:.4f}, Median: {median_val:.4f}, Std: {std_val:.4f}")
            print(f"Min: {min_val:.4f}, Max: {max_val:.4f}")
            print(f"Outliers: {outlier_count} ({outlier_percentage:.2f}%)")

            # Plot histograms - with outliers
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.hist(layer_data.flatten(), bins=100)
            plt.title(f"{layer_name} - With Outliers")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            # Plot histograms - without outliers
            plt.subplot(1, 2, 2)
            plt.hist(data_no_outliers.flatten(), bins=100)
            plt.title(f"{layer_name} - Without Outliers")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            plt.tight_layout()
            plt.savefig(f"hidden_states_analysis/{batch_count}_{layer_name}_distribution.png")
            plt.close()

        # Free memory
        del outputs_vanilla
        torch.cuda.empty_cache()
        gc.collect()
