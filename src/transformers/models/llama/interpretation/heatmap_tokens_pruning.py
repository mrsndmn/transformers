import re
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Draw a heatmap from CSV files')
    parser.add_argument('--input', nargs='+', required=True, help='CSV files to process')
    parser.add_argument('--output', required=True, help='Output directory to save the heatmap')
    parser.add_argument('--output_prefix', required=True, help='Output file prefix')
    parser.add_argument('--fig_width', type=int, default=20, help='Figure width')
    parser.add_argument('--fig_height', type=int, default=5, help='Figure height')
    parser.add_argument('--max_value', type=float, default=10, help='Maximum PPL value to consider')
    parser.add_argument('--metric_name', type=str, default='ppl', help='Metric name to consider')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load all CSV files
    data_frames = []
    for csv_file in args.input:
        df = pd.read_csv(csv_file)
        # Extract basename without extension as a label
        basename = os.path.splitext(os.path.basename(csv_file))[0]
        num_hop_layers = int(re.sub(r'.*hop_layers_(\d+).*', r'\1', basename))

        df['file'] = f"hop_layers_{num_hop_layers:02d}"
        data_frames.append(df)

    # Combine all data
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df[args.metric_name][combined_df[args.metric_name] > args.max_value] = args.max_value

    # Create a pivot table for the heatmap
    pivot_df = combined_df.pivot_table(
        values=args.metric_name,
        index='file',
        columns='fan_in_idx',
        aggfunc='first'  # Use first value if there are duplicates
    )

    vmin = 0
    cmap = 'rocket'
    if args.metric_name == 'ppl':
        vmin = 3
        cmap = 'rocket_r'

    # Create the heatmap
    plt.figure(figsize=(args.fig_width, args.fig_height))
    heatmap = sns.heatmap(
        pivot_df,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        vmin=vmin,
        linewidths=0.5,
        robust=True  # Use robust estimation for color scale
    )
    plt.title(f"{args.output_prefix}: {args.metric_name}")
    plt.xlabel("Fan-in Index")
    plt.ylabel("File")
    # plt.yscale('log')

    # Save the figure
    output_path = os.path.join(args.output, f"{args.output_prefix}_{args.metric_name}_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Heatmap saved to {output_path}")

