import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.to(device)
    model.requires_grad_(False)


    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    wikitext_103 = datasets.load_dataset("lighteval/wikitext_103", split="test")

    texts = [ wikitext_103[0]['text'] ]

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
    model_inputs = model_inputs.to(device)

    # Get the token texts for the x-axis labels
    input_ids = model_inputs['input_ids'][0]
    token_texts = [tokenizer.decode(token_id) for token_id in input_ids[:100]]  # Limit to 100 tokens

    outputs = model(**model_inputs, output_hidden_states=True)

    # [ num_layers - 1, seq_len ]
    num_layers = model.config.num_hidden_layers
    seq_len = outputs.hidden_states[0].shape[1]
    seq_len = min(100, seq_len)  # Limit sequence length

    # Store all similarity heatmaps for animation
    all_cos_distances = []
    all_l1_distances = []

    for i, h_i in enumerate(outputs.hidden_states[:-1]):
        cos_distances_headmap = torch.ones(len(outputs.hidden_states) - 1, seq_len)
        l1_distances_heatmap = torch.ones(len(outputs.hidden_states) - 1, seq_len)

        compare_hidden_states = outputs.hidden_states[i+1:]
        for j, hs_j in enumerate(compare_hidden_states):
            # [ 1, seq_len ]
            cosine_distance = 1 - (h_i * hs_j).sum(dim=-1) / (h_i.norm(dim=-1) * hs_j.norm(dim=-1))
            assert cosine_distance.shape[0] == 1
            cos_distances_headmap[i + j, :] = cosine_distance[0, :seq_len]

            # Calculate L2 norm differences
            l1_diff = (h_i - hs_j).abs().sum(dim=-1)
            l1_distances_heatmap[i + j, :] = l1_diff[0, :seq_len] * 10

        all_cos_distances.append(cos_distances_headmap.cpu().numpy())
        all_l1_distances.append(l1_distances_heatmap.cpu().numpy())

    # Create animation for cosine similarity
    plt.rcParams.update({'font.size': 25})

    # Function to create animations with shared setup
    def create_animation(heatmap_data, metric_name):
        fig, ax = plt.subplots(figsize=(50, 30))

        def update(frame):
            ax.clear()
            # Transpose the heatmap data
            im = ax.imshow(heatmap_data[frame].T, aspect='auto')
            ax.set_title(f"Layer {frame} - {metric_name}")

            # Add token labels on the y-axis
            ax.set_yticks(np.arange(len(token_texts)))
            ax.set_yticklabels(token_texts)
            
            # Add layer labels on the x-axis
            ax.set_xlabel("Layer")
            ax.set_ylabel("Token")

            return [im]

        ani = FuncAnimation(fig, update, frames=len(heatmap_data), blit=True)
        return fig, ani

    # Make sure directory exists
    output_prefix = "src/transformers/models/llama/analyze"
    os.makedirs(output_prefix, exist_ok=True)
    checkpoint_name = args.checkpoint.split("/")[-1]

    # Create and save cosine similarity animation
    fig_cos, ani_cos = create_animation(all_cos_distances, "Cosine Distance")
    video_path_cos = f"{output_prefix}/{checkpoint_name}_embeddings_cosine_distance.mp4"
    ani_cos.save(video_path_cos, writer='ffmpeg', fps=1)
    plt.close(fig_cos)
    print(f"Cosine distance video saved to {video_path_cos}")

    # Create and save L2 distance animation
    fig_l1, ani_l1 = create_animation(all_l1_distances, "L1 Distance")
    video_path_l1 = f"{output_prefix}/{checkpoint_name}_embeddings_l1_distance.mp4"
    ani_l1.save(video_path_l1, writer='ffmpeg', fps=1)
    plt.close(fig_l1)
    print(f"L1 distance video saved to {video_path_l1}")



