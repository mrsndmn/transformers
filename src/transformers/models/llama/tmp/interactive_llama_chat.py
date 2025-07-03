import argparse
import torch

from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers import pipeline, AutoTokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint

    print(f"Loading checkpoint: {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AdaptiveLlamaForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    response = pipeline("test test", max_new_tokens=100)


    while True:
        prompt = input("Enter a prompt: ")
        if prompt in ("exit", "quit", "q"):
            print("Exiting...")
            break
        response = pipeline(prompt, max_new_tokens=100)
        print(response)
