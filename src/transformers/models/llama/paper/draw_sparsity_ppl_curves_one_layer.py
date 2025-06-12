import matplotlib

import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

if __name__ == "__main__":
    # Clear any existing plots
    plt.clf()
    
    input_files_names_full = sorted(glob.glob("results/calibrate_ppl_single_layer/ppl_results_*.csv"))

    input_files = []
    for file_name in input_files_names_full:
        match = re.search(r"ppl_results_hcg_llama31_8B_L(\d+)-(\d+)", file_name)
        if match:
            l_from = match.group(1)
            l_to = match.group(2)
            input_files.append([
                file_name,
                f"Llama3.1-8B L{l_from}-{l_to}",
            ])

    fontsize = 29
    scale = 5
    
    # Set style first
    plt.style.use('seaborn-v0_8')
    
    # Create new figure with specified size
    plt.figure(figsize=(8 * scale, 6 * scale))

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
    plt.title("WikiText-103 Sparsity vs PPL", fontsize=fontsize, pad=20)
    plot_path = os.path.join("results/calibrate_ppl_single_layer", f"sparsity_ppl_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved sparsity PPL curves to {plot_path}")



