# /workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/adaptive_hcg_llama31_8B_no_forward_residuals_w_0.010_l_14_OJNVL5F5/checkpoint-124987/num_hop_layers_4_prune_percent_0.2_ppl_results.csv
# /workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/adaptive_hcg_llama31_8B_no_forward_residuals_w_0.010_l_14_OJNVL5F5/checkpoint-124987/num_hop_layers_2_prune_percent_0.2_ppl_results.csv
# /workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/adaptive_hcg_llama31_8B_no_forward_residuals_w_0.010_l_14_OJNVL5F5/checkpoint-124987/num_hop_layers_1_prune_percent_0.2_ppl_results.csv

# num_hop_layers_4_prune_percent_0.2_ppl_results.csv
# df     fan_in_idx  fan_out_idx        ppl
# 0            0            4  91.470419
# 1            1            5  10.243339
# 2            2            6  10.721656
# 3            3            7   9.600824
# 4            4            8   8.692064
# 5            5            9   6.992516
# 6            6           10   5.740384
# 7            7           11   5.832965
# 8            8           12   5.233830
# 9            9           13   5.779100
# 10          10           14   5.860275
# 11          11           15   5.664833
# 12          12           16   6.647267
# 13          13           17   5.561055
# 14          14           18   5.778818
# 15          15           19   5.493226
# 16          16           20   5.944042
# 17          17           21   6.526022
# 18          18           22   6.202088
# 19          19           23   5.610238
# 20          20           24   6.500368
# 21          21           25   8.709837
# 22          22           26   4.334691
# 23          23           27   5.939303
# 24          24           28   8.397463
# 25          25           29   6.324311
# 26          26           30  32.140509
# 27          27           31  48.232915



if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

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
        df['file'] = basename[:len("num_hop_layers_X")]
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

