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
    parser.add_argument("--output", type=str, default="weight_distributions.gif", help="Output animation file")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    wikitext_103 = datasets.load_dataset("lighteval/wikitext_103", split="test")

    texts = [ wikitext_103[0]['text'] ]

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
    model_inputs = model_inputs.to(device)

    outputs = model(**model_inputs, output_hidden_states=True)

    for i, hs in enumerate(outputs.hidden_states):
        print(f"layer {i} shape: {hs.shape}")


