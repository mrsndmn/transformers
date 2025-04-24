
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    print(args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)

    for layer_idx in range(model.config.num_hidden_layers):

        input_layernorm_weight = model.model.layers[layer_idx].input_layernorm.weight
        post_attention_layernorm_weight = model.model.layers[layer_idx].post_attention_layernorm.weight

        print("layer_idx", layer_idx, "in", (input_layernorm_weight.abs() < 0.1).sum().item(), f"\tmean {input_layernorm_weight.mean().item():.2f}", end="\t")
        print("layer_idx", layer_idx, "pa", (post_attention_layernorm_weight.abs() < 0.1).sum().item(), f"\tmean {post_attention_layernorm_weight.mean().item():.2f}")

