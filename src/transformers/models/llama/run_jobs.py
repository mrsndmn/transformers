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

def run_experiments(experiments, job_description_prefix="", dry=False):
    
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
        sparsity_level = exp.get('sparsity_level', 0.0)
        fan_out_projection = exp.get('fan_out_projection', '1') # residual_linear_projection
        ce_merging_loss_weight = exp.get('ce_merging_loss_weight', 0.0)
        hcg_loss_weight = exp.get('hcg_loss_weight', 0.0)
        model_type = exp.get('model_type', 'pretrained') # pretrained_checkpoint
        llama_checkpoint = exp.get('llama_checkpoint', '""')
        hcg_loss_weight_dynamic = exp.get('hcg_loss_weight_dynamic', '0')
        gumbel_loss_weight_dynamic = exp.get('gumbel_loss_weight_dynamic', '0')

        learning_rate = exp.get('learning_rate', 1e-4)
        per_device_train_batch_size = exp.get('per_device_train_batch_size', 32)
        save_steps = exp.get('save_steps', 1000)
        torch_compile = exp.get('torch_compile', 1)

        concrete_random_mask_proba = exp.get('concrete_random_mask_proba', '0')

        seed = SEED

        script_str = f"/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/python {workdir_prefix}/src/transformers/models/llama/train_adaptive_llama.py --save_strategy steps --save_steps {save_steps} --generate_merges_transform_impl {generate_merges_transform_impl} --per_device_train_batch_size {per_device_train_batch_size} --learning_rate {learning_rate} --num_train_epochs {num_train_epochs} --seed {seed} --training_dataset smollm-corpus --model_type {model_type} --llama_checkpoint {llama_checkpoint} --dummy_adaptive_fan_in_layers_str {dummy_adaptive_fan_in_layers_str} --full_unmerge_str {full_unmerge_str} --full_unmerge_loss_weight {full_unmerge_loss_weight} --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type cosine --merging_type {merging_type} --fan_out_type {fan_out_type} --temperature_schedule 0 --freeze_lm_backbone {freeze_lm_backbone} --fan_out_projection {fan_out_projection} --logging_steps 50 --warmup_steps {warmup_steps} --output_dir {output_dir_full_path} --max_steps_pretrain_fan_modules 0  --learnt_temperature 0 --select_train_dataset_items {select_train_dataset_items} --eval_steps 250 --gumbel_tau {gumbel_tau} --weight_decay 0.1 --scale_not_pruned_gradients {scale_not_pruned_gradients} --hcg_loss_weight {hcg_loss_weight} --ce_merging_loss_weight {ce_merging_loss_weight} --gumbel_loss_weight_dynamic {gumbel_loss_weight_dynamic} --hcg_loss_weight_dynamic {hcg_loss_weight_dynamic} --dataloader_num_workers 0 --bf16 1 --torch_compile {torch_compile} --sparsity_level {sparsity_level} --concrete_random_mask_proba {concrete_random_mask_proba}"

        print(f"\n\n{script_str}\n\n")

        job_w_args = client_lib.Job(
            base_image=BASE_IMAGE,
            script=script_str,
            # flags={
            #     # "TODO"
            # },
            type='binary', # =='binary' allows to run bash scripts
            region=REGION,
            instance_type=INSTANCE_TYPE,
            n_workers=N_WORKERS,
            # conda_env="test_client_lib",
            processes_per_worker=1,
            job_desc=f"{job_description_prefix}{output_dir}",
            # stop_timer=600, # в минутах, = 10 часов
            env_variables={
                "PATH": "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/user/conda/bin",
                "WANDB_PROJECT": "adaptive_attention",
                "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                "WANDB_MODE": "online",
                "PYTHONPATH": f"{workdir_prefix}/src",
                "HF_HOME": "/workspace-SR004.nfs2/d.tarasov/.cache/huggingface"
            },
        )

        if dry:
            print("JOB WAS NOT LAUNCHED")
        else:
            print(output_dir, job_w_args.submit())

    return


def run_hcg_smollm360M_pretrain(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_360M_pretrain"

    common_params = {
        "freeze_lm_backbone": 1,
        "select_train_dataset_items": 32000,
        "per_device_train_batch_size": 16,
        "llama_checkpoint": "HuggingFaceTB/SmolLM-360M",
        "warmup_steps": 100,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "1",
        "hcg_loss_weight": 10,
        "lm_loss_max_value": 1.0,
        "fan_out_projection": "1",
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_1",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_2",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_12",
            **common_params,
        },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return


def run_hcg_smollm1dot7B_pretrain(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_1.7B_pretrain"

    common_params = {
        "freeze_lm_backbone": 1,
        "select_train_dataset_items": 32000,
        "per_device_train_batch_size": 16,
        "llama_checkpoint": "HuggingFaceTB/SmolLM-1.7B",
        "warmup_steps": 100,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "1",
        "hcg_loss_weight": 10,
        "lm_loss_max_value": 1.0,
        "fan_out_projection": "1",
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "0,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_1",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,0,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_2",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8",
            **common_params,
        },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return


def run_hcg_adaptive_smollm1dot7B(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_1.7B"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 800000,
        "per_device_train_batch_size": 16,
        "llama_checkpoint": "HuggingFaceTB/SmolLM-1.7B",
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "1",
        "lm_loss_max_value": 1.0,
    }

    hcg_experiments = [
        # Fan out projection
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_4_w10_nofanoutproj_1.7B",
        #     "fan_out_projection": "0",
        #     "hcg_loss_weight": 10,
        #     **common_params,
        # },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_4_w10_1.7B",
        #     "fan_out_projection": "1",
        #     "hcg_loss_weight": 10,
        #     **common_params,
        # },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return


def run_hcg_smollm1dot7B_layers_iterate(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_1.7B_layersi"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 800000,
        "per_device_train_batch_size": 16,
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "1",
        "hcg_loss_weight": 10,
        "lm_loss_max_value": 1.1,
        "fan_out_projection": "1",
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "0,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_1",
            "llama_checkpoint": "adaptive_hcg_1.7B_pretrain_1/checkpoint-1993",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,0,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_2",
            "llama_checkpoint": "adaptive_hcg_1.7B_pretrain_2/checkpoint-1993",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4",
            "llama_checkpoint": "adaptive_hcg_1.7B_pretrain_4/checkpoint-1993",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8",
            "llama_checkpoint": "adaptive_hcg_1.7B_pretrain_8/checkpoint-1993",
            **common_params,
        },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return



def run_hcg_random_sampling(**kwargs):

    experiment_prefix_base_name = "random_hcg"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 800000,
        "warmup_steps": 2000,
        "model_type": "pretrained",
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": 1,
        "hcg_loss_weight": 0,
    }

    hcg_experiments = [
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_8_random0.2",
        #     "fan_out_projection": "0",
        #     "concrete_random_mask_proba": 0.2,
        #     **common_params,
        # },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4_random0.2_1.7B",
            "fan_out_projection": "0",
            "concrete_random_mask_proba": 0.2,
            **common_params,
        },
    ]

    run_experiments(hcg_experiments, job_description_prefix="Rand: ", **kwargs)

    return



if __name__ == "__main__":

    import sys

    dry = len(sys.argv) > 1 and sys.argv[1] == 'dry'
    print("dry", dry)

    # HCG
    # run_hcg_adaptive(dry=dry)
    # run_hcg_adaptive_pretrain(dry=dry)

    run_hcg_smollm360M_pretrain(dry=dry)

    # run_hcg_smollm1dot7B_pretrain(dry=dry)
    # run_hcg_adaptive_smollm1dot7B(dry=dry)
    # run_hcg_adaptive_smollm1dot7B_pretrain(dry=dry)

    # Random
    # run_hcg_random_sampling(dry=dry)
    # run_hcg_random_sampling(dry=dry)

    # Gumbel
    # run_gumbel_adaptive_pretrain(dry=dry)
    # run_gumbel_adaptive(dry=dry)