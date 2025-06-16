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

    # Create figure
    fig = go.Figure()

    for file_path, model_name in input_files:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            continue

        df = pd.read_csv(file_path)
        fig.add_trace(go.Scatter(
            x=df['sparsity'],
            y=df['ppl'],
            name=model_name,
            line=dict(width=5)
        ))

    # Update layout
    fig.update_layout(
        title="WikiText-103 Sparsity vs PPL",
        xaxis_title="Sparsity",
        yaxis_title="PPL",
        font=dict(size=29),
        showlegend=True,
        legend=dict(font=dict(size=29)),
        width=1200,
        height=900
    )

    # Save the plot
    plot_path = os.path.join("results/calibrate_ppl_single_layer", f"sparsity_ppl_curves.html")
    fig.write_html(plot_path)
    print(f"Saved interactive sparsity PPL curves to {plot_path}")
