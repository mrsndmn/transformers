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

    for exp in experiments:

        pretrained_model = exp['pretrained_model']
        output_dir = exp['output_dir']
        script_str = f'bash -c \'cd {workdir_prefix} && {env_bin_path}/python {env_bin_path}/lighteval accelerate --override-batch-size 64 --output-dir {output_dir} --custom-tasks /workspace-SR004.nfs2/d.tarasov/cosmopedia/evaluation/lighteval_tasks.py "pretrained={pretrained_model},dtype=bfloat16,device=cuda" "custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1"\''

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
                "HF_HOME": "/workspace-SR004.nfs2/d.tarasov/.cache/huggingface"
            },
        )

        if dry:
            print("JOB WAS NOT LAUNCHED")
        else:
            print(output_dir, job_w_args.submit())

    return


def eval_gumbel_adaptive_pretrain(**kwargs):

    gumbel_experiments = [
        # {
        #     "pretrained_model": "HuggingFaceTB/SmolLM-360M",
        #     "output_dir": f"exps_evaluation/hf_smollm_360m",
        # },
        # {
        #     "pretrained_model": f"./adaptive_gumbel_pretrain_12/_backup_checkpoint-15000",
        #     "output_dir": f"exps_evaluation/adaptive_gumbel_pretrain_12-checkpoint-15k",
        # },
        {
            "pretrained_model": f"./adaptive_gumbel_pretrain_12/_backup_checkpoint-20000",
            "output_dir": f"exps_evaluation/adaptive_gumbel_pretrain_12-checkpoint-20k",
        },
        {
            "pretrained_model": f"./adaptive_gumbel_pretrain_8/_backup_checkpoint-5000",
            "output_dir": f"exps_evaluation/adaptive_gumbel_pretrain_8-checkpoint-5k",
        },
    ]

    run_eval_experiments(gumbel_experiments, job_description_prefix="Eval Gumbel Adaptive: ", **kwargs)

    return



if __name__ == "__main__":

    import sys

    dry = len(sys.argv) > 1 and sys.argv[1] == 'dry'
    print("dry", dry)

    # Gumbel
    eval_gumbel_adaptive_pretrain(dry=dry)
    # run_gumbel_adaptive(dry=dry)