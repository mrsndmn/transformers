import matplotlib
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
from brokenaxes import brokenaxes

if __name__ == "__main__":

    # Threshold 0.8
    input_files = [
        [
            "results/calibrate_ppl/ppl_results_checkpoint-2500_hcg_llama31_8B_w_1.000.csv",
            "Llama3.1-8B w1.0",
        ],
        [
            "results/calibrate_ppl/ppl_results_checkpoint-5000_hcg_llama31_8B_w_0.100.csv",
            "Llama3.1-8B w0.1",
        ],
        [
            "results/calibrate_ppl/ppl_results_checkpoint-7500_hcg_qwen25_7B_w_1.000.csv",
            "Qwen2.5-7B w1.0",
        ],
        [
            "results/calibrate_ppl/ppl_results_checkpoint-7500_hcg_qwen25_7B_w_0.100.csv",
            "Qwen2.5-7B w0.1",
        ],
    ]

    # Threshold 0.6
    input_files = [
        [
            "results/calibrate_ppl/ppl_results_checkpoint-5306_hcg_llama31_8B_w_1.000_thshold_0.6.csv",
            "Llama3.1-8B w1.0",
        ],
        [
            "results/calibrate_ppl/ppl_results_checkpoint-5306_hcg_llama31_8B_w_0.100_thshold_0.6.csv",
            "Llama3.1-8B w0.1",
        ],
        [
            "results/calibrate_ppl/ppl_results_checkpoint-5306_hcg_qwen25_7B_w_1.000_thshold_0.6.csv",
            "Qwen2.5-7B w1.0",
        ],
        [
            "results/calibrate_ppl/ppl_results_checkpoint-5306_hcg_qwen25_7B_w_0.100_thshold_0.6.csv",
            "Qwen2.5-7B w0.1",
        ],
    ]

    fontsize = 29
    scale = 2
    plt.gcf().set_size_inches(8 * scale, 6 * scale)

    plt.style.use('seaborn-v0_8')

    # Threshold 0.8
    # plt.scatter([ 8.46 ], [ 3.51 ], label="Llama3.1-8B Stat. Vocab", color="blue", marker="x", s=300)
    # plt.scatter([ 8.43 ], [ 9.03 ], label="Qwen2.5-7B Stat. Vocab", color="red", marker="x", s=300)

    # Threshold 0.6
    plt.scatter([ 5.58 ], [ 4.71 ], label="Llama3.1-8B Stat. Vocab", color="blue", marker="x", s=300)
    plt.scatter([ 5.85 ], [ 8.97 ], label="Qwen2.5-7B Stat. Vocab", color="red", marker="x", s=300)

    plt.scatter([ 0 ], [ 5.34 ], label="Llama3.1-8B Orig.", color="blue", marker="o", s=200)
    plt.scatter([ 0 ], [ 9.704 ], label="Qwen2.5-7B Orig.", color="red", marker="o", s=200)


    for file_path, model_name in input_files:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            continue

        df = pd.read_csv(file_path)
        plt.plot(df['sparsity'], df['ppl'], label=model_name, linewidth=5.0)

    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Sparsity", fontsize=fontsize)
    plt.ylabel("PPL", fontsize=fontsize)
    plt.title("WikiText-103 Sparsity vs PPL", fontsize=fontsize)
    plot_path = os.path.join("results/calibrate_ppl", f"sparsity_ppl_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved sparsity PPL curves to {plot_path}")



