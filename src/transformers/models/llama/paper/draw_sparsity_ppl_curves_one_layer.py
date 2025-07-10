import matplotlib.pyplot as plt
import re
import argparse
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import os
import glob

# Set Plotly renderer to VSCode
pio.renderers.default = "vscode"

if __name__ == "__main__":
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

    # Create figure for PPL
    fig_ppl = go.Figure()

    for file_path, model_name in input_files:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            continue

        df = pd.read_csv(file_path)
        fig_ppl.add_trace(go.Scatter(
            x=df['wikitext_pruned_percent'],
            y=df['wikitext_ppl'],
            name=model_name,
            line=dict(width=5)
        ))

    # Update layout for PPL
    fig_ppl.update_layout(
        title="WikiText-103 Sparsity vs PPL",
        xaxis_title="Sparsity",
        yaxis_title="PPL",
        font=dict(size=29),
        showlegend=True,
        legend=dict(font=dict(size=29)),
        width=1200,
        height=900
    )

    # Save the PPL plot
    plot_path = os.path.join("results/calibrate_ppl_single_layer", f"sparsity_wikitext_ppl_curves.html")
    fig_ppl.write_html(plot_path)
    print(f"Saved interactive sparsity PPL curves to {plot_path}")

    # Create figure for HellaSwag
    fig_hellaswag = go.Figure()

    for file_path, model_name in input_files:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            continue

        df = pd.read_csv(file_path)
        fig_hellaswag.add_trace(go.Scatter(
            x=df['hellaswag_pruned_percent'],
            y=df['hellaswag_acc_norm'],
            name=model_name,
            line=dict(width=5)
        ))

    # Update layout for HellaSwag
    fig_hellaswag.update_layout(
        title="WikiText Sparsity vs HellaSwag Acc",
        xaxis_title="Sparsity",
        yaxis_title="Acc",
        font=dict(size=29),
        showlegend=True,
        legend=dict(font=dict(size=29)),
        width=1200,
        height=900,
        # yaxis=dict(range=[0, 1])
    )

    # Save the HellaSwag plot
    plot_path = os.path.join("results/calibrate_ppl_single_layer", f"sparsity_hellaswag_acc_curves.html")
    fig_hellaswag.write_html(plot_path)
    print(f"Saved interactive sparsity HellaSwag Acc curves to {plot_path}")
