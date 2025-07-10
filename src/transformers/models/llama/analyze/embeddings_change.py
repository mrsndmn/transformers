import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def norm_compute_cosine_distance(h_i, hs_j):
    h_i = h_i / h_i.norm(dim=-1, keepdim=True)
    hs_j = hs_j / hs_j.norm(dim=-1, keepdim=True)
    return (1 - ((h_i * hs_j).sum(dim=-1))) / 2

def compute_l1_distance(h_i, hs_j):
    return (h_i - hs_j).abs().sum(dim=-1)

def apply_forward_residuals(model, hidden_state, layer_idx, residual_type="all"):
    """Apply forward residuals to the hidden state based on residual type."""
    if residual_type == "all":
        return model.model.layers[layer_idx].forward_residuals(hidden_state)[0]
    elif residual_type == "attn_only":
        return model.model.layers[layer_idx].forward_residuals_attention_only(hidden_state)[0]
    elif residual_type == "mlp_only":
        return model.model.layers[layer_idx].forward_residuals_mlp_only(hidden_state)[0]
    else:
        raise ValueError(f"Unknown residual type: {residual_type}")

def compute_distances(h_i, hs_j, seq_len):
    """Compute cosine and L1 distances between hidden states."""
    cosine_distance = norm_compute_cosine_distance(h_i, hs_j)
    cosine_distance = cosine_distance[0, :seq_len]

    assert cosine_distance.shape[0] == 1

    l1_diff = compute_l1_distance(h_i, hs_j)
    l1_diff = l1_diff[0, :seq_len] * 10

    return cosine_distance, l1_diff

def create_animation(heatmap_data, metric_name):
    """Create animation for the given heatmap data and metric name."""
    fig, ax = plt.subplots(figsize=(10, 6))

    vmin = min([x.min() for x in heatmap_data])
    vmax = max([x.max() for x in heatmap_data])
    print("vmin", vmin, "vmax", vmax)
    if vmin == vmax:
        vmax = vmin + 1

    def update(frame):
        ax.clear()
        # Transpose the heatmap data
        im = ax.imshow(heatmap_data[frame].T, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f"Layer {frame} - {metric_name}")

        # Add token labels on the y-axis
        ax.set_yticks(np.arange(len(token_texts)))
        ax.set_yticklabels(token_texts)

        if is_adaptive and frame > 0 and frame - 1 > model.model.fan_in_idx and frame - 1 < model.model.fan_out_idx:
            # Ser color of ylabels
            for i, label in enumerate(ax.get_yticklabels()):
                if (merging_logits[0, i, 0] == 0.0).item():
                    label.set_color('red')
                else:
                    pass

        # Add layer labels on the x-axis
        ax.set_xlabel("Layer")
        ax.set_ylabel("Token")

        return [im]

    ani = FuncAnimation(fig, update, frames=len(heatmap_data), blit=True)
    return fig, ani

def save_animation(fig, ani, output_prefix, checkpoint_name, name_suffix, metric_type):
    """Save the animation to a file and close the figure."""
    video_path = f"{output_prefix}/{checkpoint_name}_embeddings_{metric_type}_{name_suffix}.mp4"
    ani.save(video_path, writer='ffmpeg', fps=1)
    plt.close(fig)
    print(f"{metric_type.capitalize()} distance video{' (' + name_suffix + ')' if name_suffix != 'distance' else ''} saved to {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--llama_checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, required=False)

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {args.checkpoint}")

    logger.info("Loading Adaptive Llama model")
    is_adaptive = 'adaptive' in args.checkpoint
    if is_adaptive:
        model = AdaptiveLlamaForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)

    logger.info("Loading Vanilla Llama model")

    model_vanilla = model
    if args.llama_checkpoint != args.checkpoint:
        model_vanilla = AutoModelForCausalLM.from_pretrained(args.llama_checkpoint, torch_dtype=torch.bfloat16)

    model_vanilla.to(device)
    model_vanilla.requires_grad_(False)

    model.to(device)
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.llama_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    wikitext_103 = datasets.load_dataset("lighteval/wikitext_103", split="test")

    texts = ["The quick brown fox jumps over the lazy dog. Быстрая коричневая лиса прыгает через ленивую собаку."]

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
    model_inputs = model_inputs.to(device)

    seq_len = min(30, model_inputs['input_ids'].shape[1])  # Limit sequence length

    model_inputs['input_ids'] = model_inputs['input_ids'][:, :seq_len]
    model_inputs['attention_mask'] = model_inputs['attention_mask'][:, :seq_len]

    # Get the token texts for the x-axis labels
    input_ids = model_inputs['input_ids'][0]
    token_texts = [tokenizer.decode(token_id) for token_id in input_ids]

    outputs = model(**model_inputs, output_hidden_states=True)
    outputs_vanilla = model_vanilla(**model_inputs, output_hidden_states=True)

    assert model_inputs['input_ids'].shape[0] == 1

    num_layers = model.config.num_hidden_layers

    # Define configurations for different residual types
    residual_configs = [
        {"type": "baseline", "name": "distance", "description": "Original"},
        {"type": "attn_only", "name": "forward_residual_attn_only", "description": "Forward Residuals (Attention Only)"},
        {"type": "mlp_only", "name": "forward_residual_mlp_only", "description": "Forward Residuals (MLP Only)"}
    ]

    # Initialize storage for results
    results = {
        config["name"]: {
            "cos": [],
            "l1": []
        } for config in residual_configs
    }

    logger.info(f"outputs.hidden_states[0].shape: {outputs.hidden_states[0].shape}")

    got_fan_in = False
    fan_out_residuals = None
    merged_embeddings_counts = None
    attention_mask = None
    residuals_attention_mask = model_inputs['attention_mask']

    merging_logits = None
    if is_adaptive:
        merging_logits = outputs.fan_in_merging_logits[ model.model.fan_in_idx ].cpu()

    for i, h_i in enumerate(outputs.hidden_states[:-1]):
        if is_adaptive:
            if h_i.shape[1] != seq_len:
                if not got_fan_in:
                    got_fan_in = True
                    fan_out_residuals = outputs.hidden_states[i - 1]
                    merged_embeddings_counts = outputs.merged_embeddings_counts
                    attention_mask = torch.ones([ h_i.shape[0], h_i.shape[1] ], device=h_i.device)

                h_i = model.model.fan_out(
                    hidden_states=h_i,
                    attention_mask=attention_mask,
                    merged_embeddings_counts=merged_embeddings_counts,
                    residual_hidden_states=fan_out_residuals,
                    residual_attention_mask=residuals_attention_mask,
                ).hidden_state

        h_i = h_i[:, :seq_len]

        # Initialize heatmaps for each configuration
        heatmaps = {
            config["name"]: {
                "cos": torch.ones(len(outputs.hidden_states) - 1, seq_len),
                "l1": torch.ones(len(outputs.hidden_states) - 1, seq_len)
            } for config in residual_configs
        }

        # Initialize hidden states for each residual type
        hidden_states = {
            config["type"]: h_i.clone() for config in residual_configs[1:]  # Skip baseline
        }
        hidden_states["baseline"] = h_i  # Baseline doesn't change

        position_ids = torch.arange(0, seq_len, device='cuda').unsqueeze(0)
        position_embeddings = model.model.rotary_emb(h_i, position_ids)

        compare_hidden_states = outputs_vanilla.hidden_states[i+1:]
        for j, hs_j in enumerate(compare_hidden_states):
            hs_j = hs_j[:, :seq_len]
            llama_layer_for_forward_residuals = i + j

            # Apply forward residuals for each type
            for config in residual_configs[1:]:  # Skip baseline
                residual_type = config["type"]
                hidden_states[residual_type] = apply_forward_residuals(
                    model,
                    hidden_states[residual_type],
                    llama_layer_for_forward_residuals,
                    residual_type
                )

            # Apply final normalization if at last layer
            if j == len(compare_hidden_states) - 1:
                for config in residual_configs[1:]:  # Skip baseline
                    residual_type = config["type"]
                    hidden_states[residual_type] = model.model.norm(hidden_states[residual_type])

            # Compute distances for each configuration
            for config in residual_configs:
                residual_type = config["type"]
                name = config["name"]

                cos_distance, l1_distance = compute_distances(
                    hidden_states[residual_type],
                    hs_j,
                    seq_len
                )

                heatmaps[name]["cos"][i + j, :] = cos_distance
                heatmaps[name]["l1"][i + j, :] = l1_distance

        # Store results
        for config in residual_configs:
            name = config["name"]
            results[name]["cos"].append(heatmaps[name]["cos"].cpu().numpy())
            results[name]["l1"].append(heatmaps[name]["l1"].cpu().numpy())

    # Make sure directory exists
    output_prefix = "src/transformers/models/llama/analyze/videos"
    os.makedirs(output_prefix, exist_ok=True)
    checkpoint_name = args.checkpoint.split("/")[-1]
    if args.checkpoint_name is not None:
        checkpoint_name = args.checkpoint_name

    # Create and save animations
    plt.rcParams.update({'font.size': 20})

    for config in residual_configs:
        name = config["name"]
        description = config["description"]

        # Cosine distance animations
        logger.info(f"Creating Animation for Cosine Distance - {description}")
        fig_cos, ani_cos = create_animation(results[name]["cos"], f"Cosine Distance{' (' + description + ')' if description != 'Original' else ''}")
        save_animation(fig_cos, ani_cos, output_prefix, checkpoint_name, name, "cosine")

        # Uncomment to create L1 distance animations
        # logger.info(f"Creating Animation for L1 Distance - {description}")
        # fig_l1, ani_l1 = create_animation(results[name]["l1"], f"L1 Distance{' (' + description + ')' if description != 'Original' else ''}")
        # save_animation(fig_l1, ani_l1, output_prefix, checkpoint_name, name, "l1")

    breakpoint()

