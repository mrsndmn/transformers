# fmt: off
import matplotlib.pyplot as plt
import torch
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
# fmt: on

from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM


# ------------------------------------------------------------------------------------
# Create GIF animation showing how mean attention maps evolve across checkpoints
# for each layer. Each GIF will have frames corresponding to successive checkpoints.
# ------------------------------------------------------------------------------------

if __name__ == "__main__":

    # List of checkpoints to visualize
    checkpoint_steps_list = [1000, 2000, 3000, 4000, 5000, 6000]

    # Will be initialized after the first forward pass when we know num_layers
    frames_per_layer = None  # list[list[np.ndarray]]

    # Output directory for GIFs (and optional individual PNGs)
    output_dir = Path("src/transformers/models/llama/tmp/attention_weights_analyze")
    output_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint_steps in checkpoint_steps_list:

        checkpoint_path = (
            f"./sentence_slm2_1.7B_pretrain_with_end_of_sentence_full_VYE9JVA0/checkpoint-{checkpoint_steps}"
        )

        with torch.no_grad():
            adaptive_llama = SentenceLlamaForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )

            # Force eager attention implementation to easily access attentions
            adaptive_llama.config._attn_implementation = "eager"

            adaptive_llama.to("cuda")
            adaptive_llama.eval()

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

            # A single example text – adjust as needed
            text = (
                "Family Life: How to choose a wedding dress. "
                "Choose an a-line fit for a pear-shape or apple-shape."
            )

            inputs = tokenizer([text], return_tensors="pt").to("cuda")

            # Forward pass with attentions
            outputs = adaptive_llama(**inputs, output_attentions=True)

            # Initialize frame containers on first iteration based on num_layers
            if frames_per_layer is None:
                num_layers = len(outputs.attentions)
                frames_per_layer = [[] for _ in range(num_layers)]

            # Collect attention maps per layer
            for layer_idx, attention in enumerate(outputs.attentions):
                # Mean over heads -> shape: (batch, seq_len, seq_len)
                attention_mean = attention.mean(dim=1)[0].to(torch.float32).cpu().numpy()

                # Plot the attention map for the current checkpoint and layer
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(attention_mean, cmap="viridis", origin="lower")
                ax.set_title(f"Layer {layer_idx}\nCheckpoint {checkpoint_steps}")
                ax.axis("off")
                fig.tight_layout(pad=0.1)

                # Draw the canvas and convert to RGB array
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                image = image.reshape((h, w, 3))

                frames_per_layer[layer_idx].append(image)

                plt.close(fig)

    # Save GIFs per layer
    for layer_idx, frames in enumerate(frames_per_layer):
        gif_path = output_dir / f"attention_layer_{layer_idx}.gif"
        # `duration` controls time per frame; here 1 sec per frame
        imageio.mimsave(gif_path, frames, duration=3, loop=1)
        print(f"Saved GIF for layer {layer_idx}: {gif_path}")
