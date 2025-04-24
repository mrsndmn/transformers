from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import argparse

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="weight_distributions.gif", help="Output animation file")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)

    # Store weights for visualization
    input_layernorm_weights = []
    post_attention_layernorm_weights = []

    hidden_size = model.config.hidden_size

    for layer_idx in range(model.config.num_hidden_layers):
        input_layernorm_weight = model.model.layers[layer_idx].input_layernorm.weight
        post_attention_layernorm_weight = model.model.layers[layer_idx].post_attention_layernorm.weight

        # Convert to numpy arrays for plotting
        input_layernorm_weights.append(input_layernorm_weight.detach().cpu().numpy())
        post_attention_layernorm_weights.append(post_attention_layernorm_weight.detach().cpu().numpy())

        print("layer_idx", layer_idx, "in", (input_layernorm_weight.abs() < 0.1).sum().item(), f"\tmean {input_layernorm_weight.mean().item():.2f}", end="\t")
        print("layer_idx", layer_idx, "pa", (post_attention_layernorm_weight.abs() < 0.1).sum().item(), f"\tmean {post_attention_layernorm_weight.mean().item():.2f}")

    # Create animation
    print(f"Creating animation...")

    # Set up the figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"LayerNorm Weight Distributions Across Layers - {os.path.basename(args.checkpoint)}")

    # Setting up empty histograms
    nbins = 50
    max_val = max(
        np.max([np.max(np.abs(w)) for w in input_layernorm_weights]),
        np.max([np.max(np.abs(w)) for w in post_attention_layernorm_weights])
    )
    bins = np.linspace(-max_val, max_val, nbins)

    n1, _, patches1 = ax1.hist(input_layernorm_weights[0], bins=bins, alpha=0.7)
    n2, _, patches2 = ax2.hist(post_attention_layernorm_weights[0], bins=bins, alpha=0.7)
    n3, _, patches3 = ax3.hist(post_attention_layernorm_weights[0] * input_layernorm_weights[0], bins=bins, alpha=0.7)

    ax1.set_title("Input LayerNorm Weights")
    ax2.set_title("Post-Attention LayerNorm Weights")
    ax3.set_title("Diff LayerNorm Weights")

    ax1.set_xlabel("Weight Value")
    ax2.set_xlabel("Weight Value")
    ax3.set_xlabel("Diff Value")

    ax1.set_ylabel("Frequency")
    ax2.set_ylabel("Frequency")
    ax3.set_ylabel("Frequency")

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')

    layer_text = fig.text(0.5, 0.01, f"Layer: 0", ha='center', fontsize=12)

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        ax1.set_title("Input LayerNorm Weights")
        ax2.set_title("Post-Attention LayerNorm Weights")
        ax3.set_title("Diff LayerNorm Weights")

        ax1.set_xlabel("Weight Value")
        ax2.set_xlabel("Weight Value")
        ax3.set_xlabel("Weight Value")

        ax1.set_ylabel("Frequency")
        ax2.set_ylabel("Frequency")
        ax3.set_ylabel("Frequency")

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')

        n1, _, patches1 = ax1.hist(input_layernorm_weights[frame], bins=bins, alpha=0.7, color='blue')
        n2, _, patches2 = ax2.hist(post_attention_layernorm_weights[frame], bins=bins, alpha=0.7, color='green')
        n3, _, patches3 = ax3.hist(post_attention_layernorm_weights[frame] * input_layernorm_weights[frame], bins=bins, alpha=0.7, color='red')

        ax1.set_ylim(0, max(np.max(n1) * 1.1, hidden_size))
        ax2.set_ylim(0, max(np.max(n2) * 1.1, hidden_size))
        ax3.set_ylim(0, max(np.max(n3) * 1.1, hidden_size))

        layer_text.set_text(f"Layer: {frame}")

        return patches1, patches2, patches3, layer_text

    anim = FuncAnimation(
        fig, update, frames=model.config.num_hidden_layers,
        blit=False, interval=500
    )

    # Save animation
    anim.save(args.output, writer='gif', dpi=100)
    print(f"Animation saved to {args.output}")

