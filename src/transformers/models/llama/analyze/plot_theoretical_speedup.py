import numpy as np
import matplotlib.pyplot as plt
import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_layers", type=int, default=32)
    parser.add_argument("--pruned_tokens_percent", type=str, default="0.15")
    args = parser.parse_args()

    print(f"Plotting theoretical speedup for model with {args.total_layers} layers and {args.pruned_tokens_percent} pruned tokens")

    total_layers = args.total_layers
    pruned_layers = np.arange(1, total_layers)

    pruned_tokens_percent = map(float, args.pruned_tokens_percent.split(","))
    for pruned_tokens_percent in pruned_tokens_percent:
        assert pruned_tokens_percent < 1.0
        speedup = 1 + 1/((total_layers - pruned_layers) / total_layers) * pruned_tokens_percent

        plt.plot(pruned_layers, speedup, label=f"{pruned_tokens_percent} pruned")

    plt.xlabel("Number of pruned layers")
    plt.ylabel("Speedup Times")
    plt.ylim(0, 5)
    plt.title("Approximate theoretical Decoding speedup")
    plt.legend()
    plt.show()

    file_name = f"src/transformers/models/llama/analyze/figures/theoretical_speedup_{args.total_layers}_{args.pruned_tokens_percent}.png"
    plt.savefig(file_name)
    print(f"Saved figure to {file_name}")
