import glob
import time
import client_lib # импортируем библиотеку для работы с ML Space
import json
from rich.console import Console

import os

assert os.environ.get("WANDB_API_KEY", "") != "", "WANDB_API_KEY is required" 

REGION = "SR004"

SEED = 1008

console = Console()
console.print(client_lib.get_instance_types(regions="SR004"))

INSTANCE_TYPE = "a100.1gpu"
N_WORKERS = 1
BASE_IMAGE = "cr.ai.cloud.ru/f51af5b1-d43b-4db4-938d-569d7cfffb7a/cuda12.1-torch2-py310-adaptive_attention:0.0.3"

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out"

def run_explore_pruningability_experiments(experiments, job_description_prefix="Explore Pruningability", dry=False):

    env_bin_path = "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin"

    print("len experiments:", len(experiments))

    for exp in experiments:

        checkpoint_base_path = exp.pop('checkpoint_base_path')
        exp_prefix = exp.pop('exp_prefix')
        fan_in_idxs = exp.pop('fan_in_idxs')
        fan_out_idxs = exp.pop('fan_out_idxs')
        concrete_random_mask_proba = exp.pop('concrete_random_mask_proba', '')

        if concrete_random_mask_proba != '':
            concrete_random_mask_proba = f"--concrete_random_mask_proba {concrete_random_mask_proba}"

        if len(exp.keys()) > 0:
            raise ValueError("Invalid exp values:" + ",".join(exp.keys()))

        script_str = f'bash -c \'date && cd {workdir_prefix} && {env_bin_path}/python src/transformers/models/llama/interpretation/explore_eval_hard_concrete_percent.py --checkpoint_base_path {checkpoint_base_path} --exp_prefix {exp_prefix} --fan_in_idxs {fan_in_idxs} --fan_out_idxs {fan_out_idxs} {concrete_random_mask_proba} \''

        print(f"\n\n{script_str}\n\n")

        job_w_args = client_lib.Job(
            base_image=BASE_IMAGE,
            script=script_str,
            type='binary', # =='binary' allows to run bash scripts
            region=REGION,
            instance_type=INSTANCE_TYPE,
            n_workers=N_WORKERS,
            # conda_env="test_client_lib",
            processes_per_worker=1,
            job_desc=f"{job_description_prefix} {exp_prefix} #rnd #multimodality",
            # stop_timer=600, # в минутах, = 10 часов
            env_variables={
                "PATH": f"{env_bin_path}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/user/conda/bin",
                "WANDB_PROJECT": "adaptive_attention",
                "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
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

def llama31_8b_pruningability(**kwargs):


    experiments = []

    # --- Vocab 20 ---
    for hop_layers in [ 1, 2, 4, 6, 8, 10, 12, 16 ]:
        experiments.append({
            "checkpoint_base_path": "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/adaptive_hcg_llama31_8B_w_0.100_l_14_4FMRKTX3/",
            "exp_prefix": f"llama31_8B_vocab20_hop_layers_{hop_layers}",
            "fan_in_idxs": ",".join(map(str, range(32-hop_layers))),
            "fan_out_idxs": ",".join(map(lambda x: str(x+hop_layers), range(32-hop_layers))),
        })

    # --- Vocab 50 ---
    for hop_layers in [ 1, 2, 4, 6, 8, 10, 12, 16 ]:
        experiments.append({
            "checkpoint_base_path": "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/adaptive_hcg_llama31_8B_w_1.000_l_14_IHHIQR0I/",
            "exp_prefix": f"llama31_8B_vocab50_hop_layers_{hop_layers}",
            "fan_in_idxs": ",".join(map(str, range(32-hop_layers))),
            "fan_out_idxs": ",".join(map(lambda x: str(x+hop_layers), range(32-hop_layers))),
        })

    run_explore_pruningability_experiments(experiments, **kwargs)

    return


def llama31_8b_instruct_pruningability(**kwargs):

    experiments = []

    # --- Vocab 20 ---
    for hop_layers in [ 1, 2, 4, 6, 8, 10, 12, 16 ]:
        experiments.append({
            "checkpoint_base_path":  "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/adaptive_hcg_llama31_8B_instruct_w_0.100_l_14_4FMRKTX3/",
            "exp_prefix": f"llama31_8B_vocab20_hop_layers_{hop_layers}",
            "fan_in_idxs": ",".join(map(str, range(32-hop_layers))),
            "fan_out_idxs": ",".join(map(lambda x: str(x+hop_layers), range(32-hop_layers))),
        })

    # --- Vocab 50 ---
    for hop_layers in [ 1, 2, 4, 6, 8, 10, 12, 16 ]:
        experiments.append({
            "checkpoint_base_path":  "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/adaptive_hcg_llama31_8B_instruct_w_1.000_l_14_IHHIQR0I/",
            "exp_prefix": f"llama31_8B_vocab50_hop_layers_{hop_layers}",
            "fan_in_idxs": ",".join(map(str, range(32-hop_layers))),
            "fan_out_idxs": ",".join(map(lambda x: str(x+hop_layers), range(32-hop_layers))),
        })

    run_explore_pruningability_experiments(experiments, **kwargs)

    return


def qwen25_7b_pruningability(**kwargs):

    experiments = []

    # --- Rand 20 ---
    for hop_layers in [ 1, 2, 4, 6, 8, 10, 12, 14 ]:
        experiments.append({
            "checkpoint_base_path": "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/adaptive_hcg_qwen25-7B-init/",
            "exp_prefix": f"qwen25_7B_rand20_hop_layers_{hop_layers}",
            "fan_in_idxs": ",".join(map(str, range(28-hop_layers))),
            "fan_out_idxs": ",".join(map(lambda x: str(x+hop_layers), range(28-hop_layers))),
            "concrete_random_mask_proba": '0.2',
        })

    run_explore_pruningability_experiments(experiments, **kwargs)

    return


def qwen25_7b_instruct_pruningability(**kwargs):

    experiments = []

    # --- Rand 20 ---
    for hop_layers in [ 1, 2, 4, 6, 8, 10, 12, 14 ]:
        experiments.append({
            "checkpoint_base_path":  "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/adaptive_hcg_qwen25-7B-instruct-init/",
            "exp_prefix": f"qwen25_7B_instruct_rand20_hop_layers_{hop_layers}",
            "fan_in_idxs": ",".join(map(str, range(28-hop_layers))),
            "fan_out_idxs": ",".join(map(lambda x: str(x+hop_layers), range(28-hop_layers))),
            "concrete_random_mask_proba": '0.2',
        })

    run_explore_pruningability_experiments(experiments, **kwargs)

    return

#
# Generate heatmaps
#

# Llama3.1 8B vocab 20
# python src/transformers/models/llama/interpretation/heatmap_tokens_pruning.py  --input adaptive_hcg_llama31_8B_w_0.100_l_14_4FMRKTX3/checkpoint-124987/llama31_8B_vocab20_hop_layers_{1,2,4,6,8,10,12,16}_ppl_results.csv  --output adaptive_hcg_llama31_8B_w_0.100_l_14_4FMRKTX3/checkpoint-124987/

# Llama3.1 8B vocab 50
# python src/transformers/models/llama/interpretation/heatmap_tokens_pruning.py  --input adaptive_hcg_llama31_8B_w_1.000_l_14_IHHIQR0I/checkpoint-90000/llama31_8B_vocab50_hop_layers_{1,2,4,6,8,10,12,16}_ppl_results.csv  --output adaptive_hcg_llama31_8B_w_1.000_l_14_IHHIQR0I/checkpoint-90000

# Llama3.1 8B instruct vocab 20
# python src/transformers/models/llama/interpretation/heatmap_tokens_pruning.py  --input adaptive_hcg_llama31_8B_instruct_w_0.100_l_14_4FMRKTX3/checkpoint-124987/llama31_8B_vocab20_hop_layers_{1,2,4,6,8,10,12,16}_ppl_results.csv  --output adaptive_hcg_llama31_8B_instruct_w_0.100_l_14_4FMRKTX3/checkpoint-124987/

# Llama3.1 8B instruct vocab 50
# python src/transformers/models/llama/interpretation/heatmap_tokens_pruning.py  --input adaptive_hcg_llama31_8B_instruct_w_1.000_l_14_IHHIQR0I/checkpoint-90000/llama31_8B_vocab50_hop_layers_{1,2,4,6,8,10,12,16}_ppl_results.csv  --output adaptive_hcg_llama31_8B_instruct_w_1.000_l_14_IHHIQR0I/checkpoint-90000/

# Qwen2.5 7B vocab 20
# python src/transformers/models/llama/interpretation/heatmap_tokens_pruning.py  --input adaptive_hcg_qwen25-7B-init/checkpoint-1/qwen25_7B_rand20_hop_layers_{1,2,4,6,8,10,12,16}_ppl_results.csv  --output adaptive_hcg_qwen25-7B-init/checkpoint-1/

# Qwen2.5 7B vocab 20
# python src/transformers/models/llama/interpretation/heatmap_tokens_pruning.py  --input adaptive_hcg_qwen25-7B-init/checkpoint-1/qwen25_7B_rand20_hop_layers_{1,2,4,6,8,10,12,16}_ppl_results.csv  --output adaptive_hcg_qwen25-7B-init/checkpoint-1/

if __name__ == "__main__":

    import sys

    dry = len(sys.argv) > 1 and sys.argv[1] == 'dry'

    print("dry", dry)

    llama31_8b_pruningability(dry=dry)
    llama31_8b_instruct_pruningability(dry=dry)

    qwen25_7b_pruningability(dry=dry)
    qwen25_7b_instruct_pruningability(dry=dry)