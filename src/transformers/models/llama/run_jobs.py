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


def run_gumbel_with_unmerge():

    experiment_prefix_base_name = "adaptive_gumbel"

    gumbel_experiments = [
        {
            "dummy_adaptive_fan_in_layers_str": "0,1,1,1,1,1,1,1,1,1,1,1",
            "full_unmerge_str":                 "1,0,0,0,0,0,0,0,0,0,0,0",
            "output_dir": f"{experiment_prefix_base_name}_1",
            "freeze_lm_backbone": "1",
            "gumbel_tau": "1.0",
            "full_unmerge_loss_weight": "2.0",
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,0,1,1,1,1,1,1,1",
            "full_unmerge_str":                 "0,0,0,0,1,0,0,0,0,0,0,0",
            "output_dir": f"{experiment_prefix_base_name}_5",
            "freeze_lm_backbone": "1",
            "gumbel_tau": "1.0",
            "full_unmerge_loss_weight": "2.0",
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,1,1,0,1,1",
            "full_unmerge_str":                 "0,0,0,0,0,0,0,0,0,1,0,0",
            "output_dir": f"{experiment_prefix_base_name}_10",
            "freeze_lm_backbone": "1",
            "gumbel_tau": "1.0",
            "full_unmerge_loss_weight": "2.0",
        },
    ]

    for exp in gumbel_experiments:
        
        dummy_adaptive_fan_in_layers_str = exp['dummy_adaptive_fan_in_layers_str']
        output_dir = exp['output_dir']
        freeze_lm_backbone = exp['freeze_lm_backbone']
        gumbel_tau = exp['gumbel_tau']
        full_unmerge_str = exp['full_unmerge_str']
        full_unmerge_loss_weight = exp['full_unmerge_loss_weight']
        seed = SEED

        job_w_args = client_lib.Job(
            base_image=BASE_IMAGE,
            script=f"{workdir_prefix}/src/transformers/models/llama/train_adaptive_llama.py --save_strategy epoch --per_device_train_batch_size 20 --learning_rate 0.0003 --num_train_epochs 1 --seed {seed} --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers_str {dummy_adaptive_fan_in_layers_str} --full_unmerge_str {full_unmerge_str} --generate_merges_transform_impl cuda_kernel --adam_beta1 0.9 --adam_beta2 0.95 --merging_type attention_output_mlp --temperature_schedule 0 --freeze_lm_backbone {freeze_lm_backbone} --fan_out_projection 1 --logging_steps 100 --warmup_steps 1000 --output_dir {output_dir} --max_steps_pretrain_fan_modules 0  --learnt_temperature 0 --select_train_dataset_items 50000 --eval_steps 250 --gumbel_tau {gumbel_tau} --weight_decay 0.1 --full_unmerge_loss_weight {full_unmerge_loss_weight}",
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

    return

def run_gumbel_adaptive():

    experiment_prefix_base_name = "adaptive_gumbel_python"

    gumbel_experiments = [
        {
            "dummy_adaptive_fan_in_layers_str": "0,1,1,1,1,1,1,1,1,1,1,1",
            "full_unmerge_str":                 "0,0,0,0,0,0,0,0,0,0,0,0",
            "output_dir": f"{experiment_prefix_base_name}_1",
            "freeze_lm_backbone": "1",
            "gumbel_tau": "1.0",
            "generate_merges_transform_impl": "python",
            "warmup_steps": 2000,
        },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,1,0,1,1,1,1,1,1,1",
        #     "full_unmerge_str":                 "0,0,0,0,0,0,0,0,0,0,0,0",
        #     "output_dir": f"{experiment_prefix_base_name}_5",
        #     "freeze_lm_backbone": "1",
        #     "generate_merges_transform_impl": "python",
        #     "warmup_steps": "2000",
        #     "gumbel_tau": "1.0",
        # },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,1,1,0,1,1",
        #     "full_unmerge_str":                 "0,0,0,0,0,0,0,0,0,0,0,0",
        #     "output_dir": f"{experiment_prefix_base_name}_10",
        #     "freeze_lm_backbone": "1",
        #     "gumbel_tau": "1.0",
        # },
    ]

    for exp in gumbel_experiments:
        
        dummy_adaptive_fan_in_layers_str = exp['dummy_adaptive_fan_in_layers_str']
        output_dir = exp['output_dir']
        output_dir_full_path = os.path.join(workdir_prefix, output_dir)
        
        freeze_lm_backbone = exp['freeze_lm_backbone']
        gumbel_tau = exp['gumbel_tau']
        full_unmerge_str = exp['full_unmerge_str']
        full_unmerge_loss_weight = exp.get('full_unmerge_loss_weight', 0.0)
        warmup_steps = exp.get('warmup_steps', 1000)
        generate_merges_transform_impl = exp.get('generate_merges_transform_impl', 'cuda_kernel')
        seed = SEED

        job_w_args = client_lib.Job(
            base_image=BASE_IMAGE,
            script=f"{workdir_prefix}/src/transformers/models/llama/train_adaptive_llama.py --save_strategy epoch --generate_merges_transform_impl {generate_merges_transform_impl} --per_device_train_batch_size 20 --learning_rate 0.0003 --num_train_epochs 1 --seed {seed} --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers_str {dummy_adaptive_fan_in_layers_str} --full_unmerge_str {full_unmerge_str} --adam_beta1 0.9 --adam_beta2 0.95 --merging_type attention_output_mlp --temperature_schedule 0 --freeze_lm_backbone {freeze_lm_backbone} --fan_out_projection 1 --logging_steps 100 --warmup_steps {warmup_steps} --output_dir {output_dir_full_path} --max_steps_pretrain_fan_modules 0  --learnt_temperature 0 --select_train_dataset_items 100000 --eval_steps 250 --gumbel_tau {gumbel_tau} --weight_decay 0.1 --full_unmerge_loss_weight {full_unmerge_loss_weight}",
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

    return



if __name__ == "__main__":

    # run_gumbel_with_unmerge()
    run_gumbel_adaptive()