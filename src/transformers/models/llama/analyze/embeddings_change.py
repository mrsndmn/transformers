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

    outputs = model(**model_inputs, output_hidden_states=True)

    # [ num_layers - 1, seq_len ]
    num_layers = model.config.num_hidden_layers
    seq_len = outputs.hidden_states[0].shape[1]
    seq_len = min(100, seq_len)  # Limit sequence length

    # Store all similarity heatmaps for animation
    all_similarities = []

    for i, h_i in enumerate(outputs.hidden_states[:-1]):
        similarities_headmap = torch.zeros(len(outputs.hidden_states) - 1, seq_len)

        compare_hidden_states = outputs.hidden_states[i+1:]
        for j, hs_j in enumerate(compare_hidden_states):
            # [ 1, seq_len ]
            cosine_similarity = (h_i * hs_j).sum(dim=-1) / (h_i.norm(dim=-1) * hs_j.norm(dim=-1))
            assert cosine_similarity.shape[0] == 1
            similarities_headmap[i + j, :] = cosine_similarity[0, :seq_len]

        all_similarities.append(similarities_headmap.cpu().numpy())

    # Create animation
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(30, 10))

    def update(frame):
        ax.clear()
        im = ax.imshow(all_similarities[frame], aspect='auto')
        ax.set_title(f"Layer {frame}")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(all_similarities), blit=True)

    # Save as MP4
    output_prefix = "src/transformers/models/llama/analyze"
    checkpoint_name = args.checkpoint.split("/")[-1]
    video_path = f"{output_prefix}/{checkpoint_name}_embeddings_change.mp4"

    # Make sure directory exists
    os.makedirs(output_prefix, exist_ok=True)

    # Save animation
    ani.save(video_path, writer='ffmpeg', fps=1)
    print(f"Video saved to {video_path}")



