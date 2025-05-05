import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers.models.llama.analyze.mean_per_token_embeddings_change import remove_outliers

if __name__ == "__main__":

    # token_idx = 279 # the
    token_idx = 3668 # character
    layer_embeddings = torch.load(f"results/token_embeddings_change/token_embeddings_{token_idx}_Qwen2.5-7B-Instruct.pt")
    layer_embeddings = layer_embeddings

    # layer_embeddings = token_embsddings['layer_embeddings']

    # OUTLIER_QUANTILE = 0.1
    OUTLIER_QUANTILE = 0.3

    occurence_diffs = []
    for occurence_idx, occurence in enumerate(layer_embeddings):

        diffs = []
        current_embedding = occurence[0]

        if OUTLIER_QUANTILE > 0.0:
            current_embedding = remove_outliers(current_embedding, quantile=OUTLIER_QUANTILE)

        for layer_idx, layer_embedding in enumerate(occurence[1:]):
            # [ 3584 ]
            # layer_embedding.shape
            if OUTLIER_QUANTILE > 0.0:
                layer_embedding = remove_outliers(layer_embedding, quantile=OUTLIER_QUANTILE)

            diff = layer_embedding - current_embedding
            diffs.append(diff.float())
            current_embedding = layer_embedding

        # num_layers - 1
        diffs = torch.stack(diffs, dim=0)

        occurence_diffs.append(diffs)

    # [ n_occurences, n_layers - 1, 3584 ]
    occurence_diffs = torch.stack(occurence_diffs, dim=0)
    print(occurence_diffs.shape)

    occurence_diffs_residual_from_mean = occurence_diffs - occurence_diffs.mean(dim=0, keepdim=True)
    # occurence_diffs_residual_from_mean = occurence_diffs - occurence_diffs[0:1]

    occurence_diffs_norm = occurence_diffs.norm(2, dim=-1)
    occurence_diffs_residual_from_mean_norm = occurence_diffs_residual_from_mean.norm(2, dim=-1)

    diff_percentage = occurence_diffs_residual_from_mean_norm / occurence_diffs_norm


    plt.clf()
    sns.heatmap(occurence_diffs_norm, cmap='viridis')
    plt.show()
    plt.savefig(f"results/token_embeddings_change/occurence_diffs_norm_{token_idx}_Qwen2.5-7B-Instruct.png")
    print(f"saved to results/token_embeddings_change/occurence_diffs_norm_{token_idx}_Qwen2.5-7B-Instruct.png")

    plt.clf()
    sns.heatmap(occurence_diffs_residual_from_mean_norm, cmap='viridis')
    plt.show()
    plt.savefig(f"results/token_embeddings_change/occurence_diffs_residual_from_mean_norm_{token_idx}_Qwen2.5-7B-Instruct.png")
    print(f"saved to results/token_embeddings_change/occurence_diffs_residual_from_mean_norm_{token_idx}_Qwen2.5-7B-Instruct.png")

    plt.clf()
    sns.heatmap(diff_percentage, cmap='viridis', vmin=0, vmax=max(1, diff_percentage.max().item()))
    plt.show()
    plt.savefig(f"results/token_embeddings_change/diff_percentage_{token_idx}_Qwen2.5-7B-Instruct.png")
    print(f"saved to results/token_embeddings_change/diff_percentage_{token_idx}_Qwen2.5-7B-Instruct.png")

    breakpoint()