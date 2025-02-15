import time
import client_lib # импортируем библиотеку для работы с ML Space

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

def run_eval_experiments(experiments, job_description_prefix="eval", dry=False):

    env_bin_path = "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin"

    print("len experiments:", len(experiments))

    for exp in experiments:

        pretrained_model = exp.pop('pretrained_model')
        output_dir = exp.pop('output_dir', './exps_evaluation')

        if len(exp.keys()) > 0:
            raise ValueError("Invalid exp values!")

        # script_str = f'bash -c \'date && cd {workdir_prefix} && {env_bin_path}/python {env_bin_path}/lighteval accelerate --override-batch-size 32 --output-dir {output_dir} --custom-tasks /workspace-SR004.nfs2/d.tarasov/cosmopedia/evaluation/lighteval_tasks.py "pretrained={pretrained_model},dtype=bfloat16,device=cuda" "custom|trivia_qa|0|1"\''
        script_str = f'bash -c \'date && cd {workdir_prefix} && {env_bin_path}/python {env_bin_path}/lighteval accelerate --override-batch-size 32 --output-dir {output_dir} --custom-tasks /workspace-SR004.nfs2/d.tarasov/cosmopedia/evaluation/lighteval_tasks.py "pretrained={pretrained_model},dtype=bfloat16,device=cuda" "custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|trivia_qa|0|1,custom|arc|0|1,custom|piqa|0|1"\''

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
            job_desc=f"{job_description_prefix} {pretrained_model}",
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
            print(output_dir, job_w_args.submit())

    return


def eval_hcg_adaptive_pretrain(**kwargs):

    hcg_experiments = [
        # {
        #     "pretrained_model": "HuggingFaceTB/SmolLM-360M",
        # },
        # {
        #     "pretrained_model": "HuggingFaceTB/SmolLM-1.7B",
        # },
        # # {
        # #     "pretrained_model": "HuggingFaceTB/SmolLM2-360M",
        # # },
        # {
        #     "pretrained_model": "HuggingFaceTB/SmolLM2-1.7B",
        # },

        {
            "pretrained_model": "./adaptive_hcg_360M_layersi_1/checkpoint-24996/",
        },
        {
            "pretrained_model": "./adaptive_hcg_360M_layersi_2/checkpoint-24996/",
        },
        {
            "pretrained_model": "./adaptive_hcg_360M_layersi_4/checkpoint-24996/",
        },
        {
            "pretrained_model": "./adaptive_hcg_360M_layersi_8/checkpoint-24996/",
        },
        {
            "pretrained_model": "./adaptive_hcg_360M_layersi_12/checkpoint-24996/",
        },


        {
            "pretrained_model": "./adaptive_hcg_1.7B_layersi_max_loss_1.1_1/checkpoint-18743",
        },
        {
            "pretrained_model": "./adaptive_hcg_1.7B_layersi_max_loss_1.1_2/checkpoint-18743",
        },
        {
            "pretrained_model": "./adaptive_hcg_1.7B_layersi_max_loss_1.1_4/checkpoint-18743",
        },
        {
            "pretrained_model": "./adaptive_hcg_1.7B_layersi_max_loss_1.1_8/checkpoint-18743",
        },


        {
            "pretrained_model": "./adaptive_hcg_slm2_1.7B_layersi_max_loss_1.25_1/checkpoint-18743",
        },
        {
            "pretrained_model": "./adaptive_hcg_slm2_1.7B_layersi_max_loss_1.25_2/checkpoint-18743",
        },
        {
            "pretrained_model": "./adaptive_hcg_slm2_1.7B_layersi_max_loss_1.25_4/checkpoint-18743",
        },
        {
            "pretrained_model": "./adaptive_hcg_slm2_1.7B_layersi_max_loss_1.25_8/checkpoint-18743",
        },
    ]

    run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG: ", **kwargs)

    return

def eval_hcg_fixed_percent(**kwargs):

    hcg_experiments = [
        {
            "pretrained_model": "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_8_pr_pct20/checkpoint-4993",
        },
        {
            "pretrained_model": "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_8_pr_pct30/checkpoint-4993",
        },
        {
            "pretrained_model": "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_8_pr_pct40/checkpoint-4993",
        },
        {
            "pretrained_model": "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_8_pr_pct60/checkpoint-4993",
        },
        {
            "pretrained_model": "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_8_pr_pct80/checkpoint-4993",
        },
    ]

    run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG: ", **kwargs)

    return

def eval_hcg_strange_8layer(**kwargs):

    hcg_experiments = [
        {
            "pretrained_model": "./run_hcg_smollm1dot7B_layer_8_8/checkpoint-18743",
        },
        {
            "pretrained_model": "./adaptive_hcg_slm1_1.7B_layer_8_fixed_pruning_percent_8_pr_pct10/checkpoint-18743",
        },
        {
            "pretrained_model": "./adaptive_hcg_slm1_1.7B_layer_8_fixed_pruning_percent_8_pr_pct20/checkpoint-18743",
        },
        {
            "pretrained_model": "./adaptive_hcg_slm1_1.7B_layer_8_fixed_pruning_percent_8_pr_pct40/checkpoint-18743",
        },
    ]

    run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG: ", **kwargs)

    return

def saturday_morninig_eval(**kwargs):

    checkpoints = [
        # Strange 8 layer
        # "./run_hcg_smollm1dot7B_layer_8_8_lmv_1.25/_backup_checkpoint-35000",
        "./run_hcg_smollm1dot7B_layer_8_8_lmv_1.1/_backup_checkpoint-35000",

        # Strange 8 layer
        # Fixed percent pruning slm2 1.7b 2 layer
        # TODO
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_2_pr_pct20/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_2_pr_pct30/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_2_pr_pct40/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_2_pr_pct60/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_2_pr_pct80/checkpoint-4993",

        # Fixed percent pruning slm2 1.7b 4 layer
        "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_4_pr_pct20/checkpoint-4993",
        # Посчиталось
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_4_pr_pct30/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_4_pr_pct40/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_4_pr_pct60/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_4_pr_pct80/checkpoint-4993",

        # Fan out ablations
        # TODO

        # Random 20% tokens
        "./random_hcg_4_random0.2_1.7B/checkpoint-24993",
        "./random_hcg_8_random0.2_1.7B/checkpoint-24993",
    ]

    hcg_experiments = [ { "pretrained_model": x } for x in checkpoints ]

    run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG: ", **kwargs)

    return

if __name__ == "__main__":

    import sys

    dry = len(sys.argv) > 1 and sys.argv[1] == 'dry'
    print("dry", dry)

    # Gumbel
    # eval_gumbel_adaptive_pretrain(dry=dry)
    # run_gumbel_adaptive(dry=dry)

    # eval_hcg_adaptive_pretrain(dry=dry)
    # eval_hcg_fixed_percent(dry=dry)

    # eval_hcg_strange_8layer(dry=dry)

    saturday_morninig_eval(dry=dry)