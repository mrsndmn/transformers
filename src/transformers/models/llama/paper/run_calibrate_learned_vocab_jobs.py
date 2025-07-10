import glob
import time
import client_lib # импортируем библиотеку для работы с ML Space
import json
import copy
from rich.console import Console

import os

assert os.environ.get("WANDB_API_KEY", "") != "", "WANDB_API_KEY is required" 

REGION = "SR004"

SEED = 1008

# console = Console()
# console.print(client_lib.get_instance_types(regions="SR004"))

INSTANCE_TYPE = "a100.1gpu"
N_WORKERS = 1
BASE_IMAGE = "cr.ai.cloud.ru/f51af5b1-d43b-4db4-938d-569d7cfffb7a/cuda12.1-torch2-py310-adaptive_attention:0.0.3"

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out"

def run_eval_experiments(experiments, job_description_prefix="eval calibrate", dry=False):

    experiments = copy.deepcopy(experiments)

    env_bin_path = "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin"

    print("len experiments:", len(experiments))

    for exp in experiments:

        llama_checkpoint = exp.pop('llama_checkpoint')
        output_suffix = exp.pop('output_suffix')
        output_dir = exp.pop('output_dir', './results/calibrate_ppl')
        calibration_dataset = exp.pop('calibration_dataset', 'wikitext_103')

        if len(exp.keys()) > 0:
            raise ValueError("Invalid exp values!")

        script_str = f'bash -c \'date && cd {workdir_prefix} && {env_bin_path}/python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --output_suffix {output_suffix} --llama_checkpoint {llama_checkpoint} --output_dir {output_dir} --calibration_dataset {calibration_dataset}\''
        # script_str = f'python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl.py --output_suffix {output_suffix} --llama_checkpoint {llama_checkpoint} --output_dir {output_dir} --sparsity_only'
        # script_str = f'bash -c \'date && cd {workdir_prefix} && {env_bin_path}/python {env_bin_path}/lighteval accelerate --override-batch-size 4 --output-dir {output_dir} --custom-tasks /workspace-SR004.nfs2/d.tarasov/cosmopedia/evaluation/lighteval_tasks.py "pretrained={pretrained_model},dtype=bfloat16,device=cuda" "custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|arc|0|1,custom|piqa|0|1,custom|wikitext_103|0|1"\''

        # TODO gsm8k, math - 8--shot

        print(f"\n\n{script_str}\n\n")

        job_w_args = client_lib.Job(
            base_image=BASE_IMAGE,
            script=script_str,
            type='binary', # =='binary' allows to run bash scripts
            region=REGION,
            instance_type=INSTANCE_TYPE,
            n_workers=N_WORKERS,
            processes_per_worker=1,
            job_desc=f"{job_description_prefix} {output_suffix} #rnd #multimodality @mrsndmn",
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


def run_eval_experiments_single_layer(experiments, job_description_prefix="eval calibrate single layer", dry=False):

    experiments = copy.deepcopy(experiments)

    env_bin_path = "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin"

    print("len experiments:", len(experiments))

    for exp in experiments:

        layer_idx = exp.pop('layer_idx')

        if len(exp.keys()) > 0:
            raise ValueError("Invalid exp values!")

        script_str = f'bash -c \'date && cd {workdir_prefix} && {env_bin_path}/python src/transformers/models/llama/paper/calibrate_learned_vocabulary_calc_ppl_single_layer.py --layer_idx {layer_idx}\''
        # script_str = f'bash -c \'date && cd {workdir_prefix} && {env_bin_path}/python {env_bin_path}/lighteval accelerate --override-batch-size 4 --output-dir {output_dir} --custom-tasks /workspace-SR004.nfs2/d.tarasov/cosmopedia/evaluation/lighteval_tasks.py "pretrained={pretrained_model},dtype=bfloat16,device=cuda" "custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|arc|0|1,custom|piqa|0|1,custom|wikitext_103|0|1"\''

        # TODO gsm8k, math - 8--shot

        print(f"\n\n{script_str}\n\n")

        job_w_args = client_lib.Job(
            base_image=BASE_IMAGE,
            script=script_str,
            type='binary', # =='binary' allows to run bash scripts
            region=REGION,
            instance_type=INSTANCE_TYPE,
            n_workers=N_WORKERS,
            processes_per_worker=1,
            job_desc=f"{job_description_prefix} i={layer_idx} #rnd #multimodality @mrsndmn",
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
            print(layer_idx, job_w_args.submit())

    return



if __name__ == "__main__":

    import sys

    dry = len(sys.argv) > 1 and sys.argv[1] == 'dry'
    extract_metrics = len(sys.argv) > 1 and sys.argv[1] == 'extract_metrics'

    print("dry", dry, 'extract_metrics', extract_metrics)


    # if True:
    if False:
        run_eval_experiments([
            {
                'llama_checkpoint': './paper_checkpoints/base_thshld_0.6/adaptive_hcg_llama31_8B_learned_vocab_w_1.000_l_18-26_M0K275CH/checkpoint-5306',
                'output_suffix': 'hcg_llama31_8B_w_1.000_thshold_0.6',
                'output_dir': './results/calibrate_ppl/'
            },
            {
                'llama_checkpoint': './paper_checkpoints/base_thshld_0.6/adaptive_hcg_llama31_8B_learned_vocab_w_0.100_l_18-26_DAZ0UXGX/checkpoint-5306',
                'output_suffix': 'hcg_llama31_8B_w_0.100_thshold_0.6',
                'output_dir': './results/calibrate_ppl/'
            },
            {
                'llama_checkpoint': './paper_checkpoints/base_thshld_0.6/adaptive_hcg_qwen25_7B_learned_vocab_w_1.000_l_9-20_J3BTODV3/checkpoint-5306',
                'output_suffix': 'hcg_qwen25_7B_w_1.000_thshold_0.6',
                'output_dir': './results/calibrate_ppl/'
            },
            {
                'llama_checkpoint': './paper_checkpoints/base_thshld_0.6/adaptive_hcg_qwen25_7B_learned_vocab_w_0.100_l_9-20_3P72MPZU/checkpoint-5306',
                'output_suffix': 'hcg_qwen25_7B_w_0.100_thshold_0.6',
                'output_dir': './results/calibrate_ppl/'
            },
            {
                'llama_checkpoint': './adaptive_hcg_llama31_8B_fan_out_projection_w_1.000_l_18-26_R2DI580B/checkpoint-10999',
                'output_suffix': 'adaptive_hcg_llama31_8B_fan_out_projection_w_1.000_l_18-26_R2DI580B',
                'output_dir': './results/calibrate_ppl/',
            },
        ], dry=dry)

    if True:
    # if False:
        run_eval_experiments([
            {
                'llama_checkpoint': './adaptive_slm2_135M_pretrain_w_0.100_l_10-20_AZJQ5WL0/checkpoint-12420/',
                'output_suffix': 'adaptive_slm2_135M_pretrain_w_0.100_l_10-20_AZJQ5WL0',
                'output_dir': './results/calibrate_slm_ppl/',
                'calibration_dataset': 'tiny_stories'
            },
            {
                'llama_checkpoint': './adaptive_slm2_135M_pretrain_with_end_of_sentence_token_w_0.100_l_10-20_4REEAIIL/checkpoint-12420/',
                'output_suffix': 'adaptive_slm2_135M_pretrain_with_end_of_sentence_token_w_0.100_l_10-20_4REEAIIL',
                'output_dir': './results/calibrate_slm_ppl/',
                'calibration_dataset': 'tiny_stories'
            },
        ], dry=dry)



    # run_eval_experiments_single_layer([
    #     {
    #         'layer_idx': i,
    #     } for i in range(2, 30)
    #     # } for i in range(2, 3)
    # ], dry=dry)
