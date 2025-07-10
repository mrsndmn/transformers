from typing import Callable, Optional
import matplotlib
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
from brokenaxes import brokenaxes

def draw_sparsity_ppl_curves(
        input_files,
        plot_path,
        plot_fuffix="",
        wikitext=True,
        hellaswag=True,
        tiny_stories=False,
        hellaswag_ylim=(0, 1),
        extra_points: Optional[Callable]=None,
    ):
    if len(plot_fuffix) > 0:
        plot_fuffix = "_" + plot_fuffix

    os.makedirs(plot_path, exist_ok=True)

    fontsize = 29
    scale = 2
    plt.gcf().set_size_inches(8 * scale, 6 * scale)

    plt.style.use('seaborn-v0_8')

    if extra_points is not None:
        extra_points()

    if wikitext:
        for file_path, model_name in input_files:
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist")
                continue

            df = pd.read_csv(file_path)
            plt.plot(df['wikitext_pruned_percent'], df['wikitext_ppl'], label=model_name, linewidth=5.0)
            # plt.errorbar(df['sparsity'], df['wikitext_ppl'], yerr=df['wikitext_ppl_stderr'], label=model_name, linewidth=5.0, capsize=5, capthick=2)

        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel("Sparsity", fontsize=fontsize)
        plt.ylabel("PPL", fontsize=fontsize)

        plt.ylim(0, 40)

        plt.title("WikiText: Sparsity vs PPL", fontsize=fontsize)
        wikitext_plot_path = os.path.join(plot_path, f"sparsity_wikitext_ppl_curves{plot_fuffix}.png")
        plt.tight_layout()
        plt.savefig(wikitext_plot_path)
        print(f"Saved sparsity PPL curves to {wikitext_plot_path}")

        plt.clf()

    if hellaswag:

        for file_path, model_name in input_files:
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist")
                continue

            df = pd.read_csv(file_path)
            plt.plot(df['hellaswag_pruned_percent'], df['hellaswag_acc_norm'], label=model_name, linewidth=5.0)

        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel("Sparsity", fontsize=fontsize)
        plt.ylabel("Acc", fontsize=fontsize)

        plt.ylim(*hellaswag_ylim)

        plt.title("HellaSwag: Sparsity vs Accuracy", fontsize=fontsize)
        hellaswag_plot_path = os.path.join(plot_path, f"sparsity_hellaswag_acc_curves{plot_fuffix}.png")
        plt.tight_layout()
        plt.savefig(hellaswag_plot_path)
        print(f"Saved sparsity HellaSwag Acc curves to {hellaswag_plot_path}")

        plt.clf()

    if tiny_stories:
        for file_path, model_name in input_files:
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist")
                continue

            df = pd.read_csv(file_path)
            plt.plot(df['tiny_stories_pruned_percent'], df['tiny_stories_ppl'], label=model_name, linewidth=5.0)

        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel("Sparsity", fontsize=fontsize)
        plt.ylabel("PPL", fontsize=fontsize)

        plt.ylim(0, 5)

        plt.title("TinyStories: Sparsity vs PPL", fontsize=fontsize)
        tiny_stories_plot_path = os.path.join(plot_path, f"sparsity_tiny_stories_ppl_curves{plot_fuffix}.png")
        plt.tight_layout()
        plt.savefig(tiny_stories_plot_path)
        print(f"Saved sparsity TinyStories PPL curves to {tiny_stories_plot_path}")

        plt.clf()

if __name__ == "__main__":

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

    # Custom points
    def extra_points():
        plt.scatter([ 5.58 ], [ 4.71 ], label="Llama3.1-8B Stat. Vocab", color="blue", marker="x", s=300, alpha=0.5)
        plt.scatter([ 5.85 ], [ 8.97 ], label="Qwen2.5-7B Stat. Vocab", color="red", marker="x", s=300, alpha=0.5)

        plt.scatter([ 0 ], [ 5.34 ], label="Llama3.1-8B Orig.", color="blue", marker="o", s=200, alpha=0.5)
        plt.scatter([ 0 ], [ 9.704 ], label="Qwen2.5-7B Orig.", color="red", marker="o", s=200, alpha=0.5)


    draw_sparsity_ppl_curves(input_files, "results/calibrate_ppl", plot_fuffix="threshold_0.6", extra_points=extra_points)


    # SLMs
    input_files = [
        [
            "results/calibrate_slm_ppl/ppl_results___adaptive_slm2_135M_pretrain_w_0_100_l_10-20_AZJQ5WL0_checkpoint-12420__adaptive_slm2_135M_pretrain_w_0.100_l_10-20_AZJQ5WL0.csv",
            "Adaptive SLM2-135M w1.0",
        ],
        [
            "results/calibrate_slm_ppl/ppl_results___adaptive_slm2_135M_pretrain_with_end_of_sentence_token_w_0_100_l_10-20_4REEAIIL_checkpoint-12420__adaptive_slm2_135M_pretrain_with_end_of_sentence_token_w_0.100_l_10-20_4REEAIIL.csv",
            "Adaptive SLM2-135M w1.0 EoS",
        ],
    ]

    draw_sparsity_ppl_curves(
        input_files,
        "results/calibrate_slm_ppl",
        plot_fuffix="",
        wikitext=False,
        tiny_stories=True,
        hellaswag_ylim=()
        hellaswag_ylim=(0, 0.5),
    )