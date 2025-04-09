import os
import torch
import pytest
import safetensors
import matplotlib.pyplot as plt
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers.models.llama.train_adaptive_llama import freeze_lm_backbone
import imageio  # Add imageio import
import io # Add io import


if __name__ == "__main__":


    torch.set_default_device('cuda')

    checkpoint_base_path = "./adaptive_hcg_slm2_360M_full_1.0_a100.4gpu_WC650AC2/"

    checkpoints = os.listdir(checkpoint_base_path)
    checkpoints = [x for x in checkpoints if x.startswith('checkpoint')]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

    images = [] # List to store plot images

    for i, checkpoint in enumerate(checkpoints):
        print(f"Processing checkpoint: {checkpoint} ({i+1}/{len(checkpoints)})")
        checkpoint_path = os.path.join(checkpoint_base_path, checkpoint)

        tokeniser = AutoTokenizer.from_pretrained(checkpoint_path)

        pretrained_checkpoint = os.path.join(checkpoint_path, "model.safetensors")
        state_dict = safetensors.torch.load_file(pretrained_checkpoint)

        # Assuming 'model.adaptive_down.3.hcg.hcg_log_a' exists, otherwise adjust key
        hcg_log_a_key = 'model.adaptive_down.3.hcg.hcg_log_a'
        if hcg_log_a_key not in state_dict:
            # Try finding a similar key if the exact one isn't present (example adjustment)
            potential_keys = [k for k in state_dict.keys() if 'hcg.hcg_log_a' in k]
            if not potential_keys:
                print(f"Warning: '{hcg_log_a_key}' not found in {checkpoint}. Skipping.")
                continue
            hcg_log_a_key = potential_keys[0] # Use the first found key
            print(f"Using alternative key: {hcg_log_a_key}")

        hcg_log_a = state_dict[hcg_log_a_key]

        # Calculate approximate pruning probability: sigmoid(log_a)
        # Higher log_a means higher chance of passing (lower chance of pruning).
        passing_probs = torch.sigmoid(hcg_log_a) # Changed name for clarity
        passing_probs_np = passing_probs.cpu().numpy()

        # Plot pruned probability dashboard (histogram)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(passing_probs_np, bins=50, color='skyblue', edgecolor='black', range=(0, 1)) # Set range to [0, 1]
        ax.set_title(f'Distribution of Passing Probabilities (Sigmoid(log_a)) Checkpoint: {checkpoint}')
        ax.set_xlabel('Approximate Passing Probability')
        ax.set_ylabel('Number of Tokens')
        ax.grid(axis='y', alpha=0.75)
        ax.set_ylim(bottom=0, top=5000) # Ensure y-axis starts at 0

        # Save plot to a BytesIO object
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))
        plt.close(fig) # Close the figure to free memory

    # Save images as an animated GIF
    if images:
        print("Creating animation...")
        output_path = 'src/transformers/models/llama/interpretation/pruning_probs_animation.gif'
        imageio.mimsave(output_path, images, fps=2, loop=1) # Adjust fps as needed
        print(f"Animation saved to {output_path}")
    else:
        print("No images were generated to create an animation.")


    # For last checkpoint see top pruned tokens.
    top_passed =hcg_log_a.argsort(dim=-1, descending=True)
    top_pruned = hcg_log_a.argsort(dim=-1, descending=False)


    # Analyze corresponding pruned tokens
    print("--- Top 100 Most Likely to be Passed Tokens ---")
    top_passed_tokens = tokeniser.batch_decode(top_passed[:100])
    print(top_passed_tokens)

    print("\n--- Top 100 Most Likely to be Pruned Tokens ---")
    top_pruned_tokens = tokeniser.batch_decode(top_pruned[:100])
    print(top_pruned_tokens)

