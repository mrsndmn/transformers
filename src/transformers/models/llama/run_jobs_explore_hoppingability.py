import time
import string
import random
import client_lib # импортируем библиотеку для работы с ML Space

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rich.console import Console

import os

assert os.environ.get("WANDB_API_KEY", "") != "", "WANDB_API_KEY is required"

from copy import deepcopy

REGION = "SR004"

SEED = 1008

INSTANCE_TYPE = "a100.1gpu"
N_WORKERS = 1
BASE_IMAGE = "cr.ai.cloud.ru/f51af5b1-d43b-4db4-938d-569d7cfffb7a/cuda12.1-torch2-py310-adaptive_attention:0.0.3"

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out"

def run_experiments(experiments, job_description_prefix="", dry=False):

    for exp in experiments:
        exp = deepcopy(exp)

        instance_type = exp.pop('instance_type', INSTANCE_TYPE)
        checkpoint_base_path = exp.pop('checkpoint_base_path', '')
        exp_prefix = exp.pop('exp_prefix')
        concrete_random_mask_proba = exp.pop('concrete_random_mask_proba')
        fan_in_idxs = exp.pop('fan_in_idxs')
        fan_out_idxs = exp.pop('fan_out_idxs')

        if len(exp.keys()) > 0:
            raise ValueError(f"unknown parsms:{exp}")

        assert instance_type == 'a100.1gpu'


        script_str = f"/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/python {workdir_prefix}/src/transformers/models/llama/interpretation/explore_eval_hard_concrete_percent.py --checkpoint_base_path {workdir_prefix}/{checkpoint_base_path} --exp_prefix {exp_prefix} --concrete_random_mask_proba {concrete_random_mask_proba} --fan_in_idxs {fan_in_idxs} --fan_out_idxs {fan_out_idxs}"

        print(f"\n\n{script_str}\n\n")

        job_w_args = client_lib.Job(
            base_image=BASE_IMAGE,
            script=script_str,
            # flags={
            #     # "TODO"
            # },
            type='binary', # =='binary' allows to run bash scripts
            region=REGION,
            instance_type=instance_type,
            n_workers=N_WORKERS,
            # conda_env="test_client_lib",
            processes_per_worker=1,
            job_desc=f"{job_description_prefix} #rnd #multimodality #tarasov",
            # stop_timer=600, # в минутах, = 10 часов
            env_variables={
                "PATH": "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/user/conda/bin",
                "WANDB_PROJECT": "adaptive_attention",
                "WANDB_MODE": "online",
                "PYTHONPATH": f"{workdir_prefix}/src:/workspace-SR004.nfs2/d.tarasov/lighteval/src",
                "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface"
            },
        )

        if dry:
            print("JOB WAS NOT LAUNCHED")
        else:
            print(exp_prefix, job_w_args.submit())

    return

def run_plot_results(experiments, title=None):

    checkpoint_base_path = experiments[0]['checkpoint_base_path']
    checkpoints = os.listdir(checkpoint_base_path)
    checkpoints = [x for x in checkpoints if x.startswith('checkpoint')]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

    if experiments[0]['checkpoint_base_path'] == "adaptive_hcg_llama31_8B_no_forward_residuals_w_0.010_l_14_OJNVL5F5":
        checkpoints[-1] = "checkpoint-70000"
    last_checkpoint_path = os.path.join(checkpoint_base_path, checkpoints[-1])

    # Get all unique pruned percentages
    pruned_percents = sorted(set(exp['concrete_random_mask_proba'] for exp in experiments))

    # Create figure and axis
    fig, ax = plt.subplots()

    # Dictionary to store data for each pruned percentage
    data_by_percent = {}

    # Collect all data
    for pruned_percent in pruned_percents:
        data_by_percent[pruned_percent] = []

        for exp in experiments:
            exp_prefix = exp['exp_prefix']
            prune_percent = exp['concrete_random_mask_proba']

            if prune_percent != pruned_percent:
                continue

            csv_file_name = f"{exp_prefix}_ppl_results.csv"
            csv_path = os.path.join(last_checkpoint_path, csv_file_name)

            if not os.path.exists(csv_path):
                print(f"CSV file {csv_path} does not exist")
                continue

            df = pd.read_csv(csv_path)
            data_by_percent[pruned_percent].append({
                'fan_in_idx': df['fan_in_idx'],
                'ppl': df['ppl'],
                'label': exp_prefix
            })

    # Function to update the plot for animation
    def update(frame):
        ax.clear()
        pruned_percent = pruned_percents[frame]

        for data in data_by_percent[pruned_percent]:
            ax.plot(data['fan_in_idx'], data['ppl'], label=data['label'])

        ax.legend()
        ax.set_title(f"{title} pruned percent {pruned_percent}")
        ax.set_ylim(0, 20)
        ax.set_xlabel("Layer")
        ax.set_ylabel("PPL")

        return ax

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(pruned_percents),
        interval=1000, blit=False
    )

    # Save animation
    animation_path = os.path.join(checkpoint_base_path, f"ppl_results_animation.gif")
    ani.save(animation_path, writer='pillow')
    print(f"Animation saved to {animation_path}")

    plt.show()


def run_hcg_smollm2_1dot7B_hopppingability(plot_results=False, **kwargs):

    hcg_experiments = []

    common_params = {
        "checkpoint_base_path": "adaptive_hcg_slm2_1.7B_no_forward_residuals_w_0.010_l_10_59B9H4TB",
    }

    max_layer_i = 23

    for num_hop_layers in [ 1, 4, 8 ]:
        for prune_percent in [ 0.2, 0.4, 0.8 ]:
            fan_in_idxs = []
            fan_out_idxs = []
            for start_layer in [ 0, 8, 12, 16, 19 ]:
                if start_layer + num_hop_layers > max_layer_i:
                    continue
                fan_in_idxs.append(start_layer)
                fan_out_idxs.append(start_layer + num_hop_layers)

            exp_config = {
                **common_params,
                "fan_in_idxs": ",".join(map(str, fan_in_idxs)),
                "fan_out_idxs": ",".join(map(str, fan_out_idxs)),
                "concrete_random_mask_proba": prune_percent,
                "exp_prefix": f"num_hop_layers_{num_hop_layers}_prune_percent_{prune_percent}",
            }
            hcg_experiments.append(exp_config)

    if not plot_results:
        run_experiments(hcg_experiments, job_description_prefix="Hoppingability SLM2 1.7B: ", **kwargs)
    else:
        run_plot_results(hcg_experiments, title="SLM2 1.7B")

    return


def run_hcg_llama31_8B_hopppingability(plot_results=False, **kwargs):

    hcg_experiments = []

    common_params = {
        "checkpoint_base_path": "adaptive_hcg_llama31_8B_no_forward_residuals_w_0.010_l_14_OJNVL5F5",
    }

    max_layer_i = 31

    for num_hop_layers in [ 1, 4, 8 ]:
        for prune_percent in [ 0.2, 0.4, 0.8 ]:
            fan_in_idxs = []
            fan_out_idxs = []
            for start_layer in [ 0, 8, 16, 24, 27 ]:
                if start_layer + num_hop_layers > max_layer_i:
                    continue
                fan_in_idxs.append(start_layer)
                fan_out_idxs.append(start_layer + num_hop_layers)

            exp_config = {
                **common_params,
                "fan_in_idxs": ",".join(map(str, fan_in_idxs)),
                "fan_out_idxs": ",".join(map(str, fan_out_idxs)),
                "concrete_random_mask_proba": prune_percent,
                "exp_prefix": f"num_hop_layers_{num_hop_layers}_prune_percent_{prune_percent}",
            }
            hcg_experiments.append(exp_config)

    if not plot_results:
        run_experiments(hcg_experiments, job_description_prefix="Hoppingability Llama3.1 8B: ", **kwargs)
    else:
        run_plot_results(hcg_experiments, title="Llama3.1 8B")

    return



if __name__ == "__main__":

    import sys
    import subprocess

    dry = len(sys.argv) > 1 and sys.argv[1] == 'dry'
    plot_results = len(sys.argv) > 1 and sys.argv[1] == 'plot_results'
    print("dry", dry)

    run_hcg_smollm2_1dot7B_hopppingability(dry=dry, plot_results=plot_results)
    run_hcg_llama31_8B_hopppingability(dry=dry, plot_results=plot_results)