from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
import torch.nn as nn
import sys
import torch

if __name__ == "__main__":

    model_path = sys.argv[1]

    model = AdaptiveLlamaForCausalLM.from_pretrained(model_path)

    log_a = nn.Sigmoid()(model.model.fan_in.hcg.hcg_log_a)
    print(f"log_a     : {log_a.shape}")
    print(f"log_a < 0.001: {(log_a < 0.001).sum()}")
    print(f"log_a == 0: {(log_a == 0).sum()}")
    print(f"log_a == 1: {(log_a == 1).sum()}")

    # Quantile of log_a from 0.01 to 0.99
    print(f"Quantile 0.01 of log_a: {torch.quantile(log_a, 0.01).item()}")
    print(f"Quantile 0.05 of log_a: {torch.quantile(log_a, 0.05).item()}")
    print(f"Quantile 0.25 of log_a: {torch.quantile(log_a, 0.25).item()}")
    print(f"Quantile 0.50 of log_a: {torch.quantile(log_a, 0.50).item()}")
    print(f"Quantile 0.75 of log_a: {torch.quantile(log_a, 0.75).item()}")
    print(f"Quantile 0.95 of log_a: {torch.quantile(log_a, 0.95).item()}")
    print(f"Quantile 0.99 of log_a: {torch.quantile(log_a, 0.99).item()}")
