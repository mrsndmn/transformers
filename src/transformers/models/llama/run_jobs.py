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

def run_experiments(experiments):
    
    for exp in experiments:
        dummy_adaptive_fan_in_layers_str = exp['dummy_adaptive_fan_in_layers_str']
        output_dir = exp['output_dir']
        output_dir_full_path = os.path.join(workdir_prefix, output_dir)
        
        freeze_lm_backbone = exp['freeze_lm_backbone']
        gumbel_tau = exp.get('gumbel_tau', 1.0)
        
        full_unmerge_str_default: str = dummy_adaptive_fan_in_layers_str
        full_unmerge_str_default = full_unmerge_str_default.replace('1', '0') # all zeros
        full_unmerge_str = exp.get('full_unmerge_str', full_unmerge_str_default)
        full_unmerge_loss_weight = exp.get('full_unmerge_loss_weight', 0.0)

        warmup_steps = exp.get('warmup_steps', 2000)
        num_train_epochs = exp.get('num_train_epochs', 1)
        generate_merges_transform_impl = exp.get('generate_merges_transform_impl', 'cuda_kernel')
        select_train_dataset_items = exp.get('select_train_dataset_items', 150000)
        scale_not_pruned_gradients = exp.get('scale_not_pruned_gradients', 0.0)
        merging_type = exp.get('merging_type', 'hcg') # attention_output_mlp
        fan_out_type = exp.get('fan_out_type', 'hcg') # residual_linear_projection
        ce_merging_loss_weight = exp.get('ce_merging_loss_weight', 0.0)
        hcg_loss_weight = exp.get('hcg_loss_weight', 0.0)
        model_type = exp.get('model_type', 'pretrained') # pretrained_checkpoint
        llama_checkpoint = exp.get('llama_checkpoint', '""')

        learning_rate = exp.get('learning_rate', 3e-5)

        seed = SEED

        script_str = f"{workdir_prefix}/src/transformers/models/llama/train_adaptive_llama.py --save_strategy epoch --generate_merges_transform_impl {generate_merges_transform_impl} --per_device_train_batch_size 5 --learning_rate {learning_rate} --num_train_epochs {num_train_epochs} --seed {seed} --training_dataset smollm-corpus --model_type {model_type} --llama_checkpoint {llama_checkpoint} --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers_str {dummy_adaptive_fan_in_layers_str} --full_unmerge_str {full_unmerge_str} --full_unmerge_loss_weight {full_unmerge_loss_weight} --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type linear --merging_type {merging_type} --fan_out_type {fan_out_type} --temperature_schedule 0 --freeze_lm_backbone {freeze_lm_backbone} --fan_out_projection 1 --logging_steps 50 --warmup_steps {warmup_steps} --output_dir {output_dir_full_path} --max_steps_pretrain_fan_modules 0  --learnt_temperature 0 --select_train_dataset_items {select_train_dataset_items} --eval_steps 250 --gumbel_tau {gumbel_tau} --weight_decay 0.1 --scale_not_pruned_gradients {scale_not_pruned_gradients} --hcg_loss_weight {hcg_loss_weight} --ce_merging_loss_weight {ce_merging_loss_weight}"
        
        print(f"\n\n{script_str}\n\n")
        
        job_w_args = client_lib.Job(
            base_image=BASE_IMAGE,
            script=script_str,
            # flags={
            #     # "TODO"
            # },
            # type='', # =='binary' allows to run bash scripts
            region=REGION,
            instance_type=INSTANCE_TYPE,
            n_workers=N_WORKERS,
            # conda_env="test_client_lib",
            processes_per_worker=1,
            job_desc=f"AA Gumbel: {output_dir}",
            # stop_timer=600, # в минутах, = 10 часов
            env_variables={
                "WANDB_PROJECT": "adaptive_attention",
                "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""), 
                "WANDB_MODE": "online",
                "PYTHONPATH": f"{workdir_prefix}/src",
                "HF_HOME": "/workspace-SR004.nfs2/d.tarasov/.cache/huggingface"
            },
        )

        print(output_dir, job_w_args.submit())
        # print("JOB WAS NOT LAUNCHED")

    return


def run_gumbel_adaptive():

    experiment_prefix_base_name = "adaptive_gumbel"

    hcg_experiments = [
        {
            "fan_out_type": "adaptive_fan_out_gumbel",
            "merging_type": "next_token_merge_mlp",
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_10",
            "freeze_lm_backbone": 0,
            "ce_merging_loss_weight": 1.0,
            "min_ce_merging_loss_value": 1.0,
            "select_train_dataset_items": 0,
        },
    ]

    run_experiments(hcg_experiments)

    return



def run_hcg_adaptive_pretrain():

    experiment_prefix_base_name = "adaptive_hcg"

    hcg_experiments = [
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_10_pretrain",
            "freeze_lm_backbone": 1,
            "hcg_loss_weight": 1,
            "select_train_dataset_items": 15000,
        },
    ]

    run_experiments(hcg_experiments)

    return


def run_hcg_adaptive():

    experiment_prefix_base_name = "adaptive_hcg"

    hcg_experiments = [
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_10_weight_1",
            "freeze_lm_backbone": 0,
            "hcg_loss_weight": 1,
            "model_type": "pretrained_checkpoint",
            "llama_checkpoint": "./adaptive_hcg_10_pretrain",
            "select_train_dataset_items": 0,
        },
    ]

    run_experiments(hcg_experiments)

    return




if __name__ == "__main__":

    # run_hcg_adaptive_pretrain()
    # run_hcg_adaptive()
    run_gumbel_adaptive()