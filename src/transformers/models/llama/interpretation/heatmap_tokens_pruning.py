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
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load all CSV files
    data_frames = []
    for csv_file in args.input:
        df = pd.read_csv(csv_file)
        # Extract basename without extension as a label
        basename = os.path.splitext(os.path.basename(csv_file))[0]
        df['file'] = re.sub(r'.*(hop_layers_\d+).*', r'\1', basename)
        data_frames.append(df)

    # Combine all data
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df['ppl'][combined_df['ppl'] > 10] = 10

    # Create a pivot table for the heatmap
    pivot_df = combined_df.pivot_table(
        values='ppl',
        index='file',
        columns='fan_in_idx',
        aggfunc='first'  # Use first value if there are duplicates
    )

    # Create the heatmap
    plt.figure(figsize=(20, 5))
    heatmap = sns.heatmap(
        pivot_df,
        annot=True,
        cmap='rocket_r',
        fmt=".2f",
        vmin=3,
        linewidths=0.5,
        robust=True  # Use robust estimation for color scale
    )
    plt.title("PPL Values by Fan-in Index and File")
    plt.xlabel("Fan-in Index")
    plt.ylabel("File")
    # plt.yscale('log')

    # Save the figure
    output_path = os.path.join(args.output, "ppl_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Heatmap saved to {output_path}")

