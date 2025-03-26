import time
import client_lib # импортируем библиотеку для работы с ML Space

from rich.console import Console

import os

assert os.environ.get("WANDB_API_KEY", "") != "", "WANDB_API_KEY is required" 

from copy import deepcopy

REGION = "SR004"

SEED = 1008

console = Console()
console.print(client_lib.get_instance_types(regions="SR004"))

INSTANCE_TYPE = "a100.4gpu"
N_WORKERS = 1
BASE_IMAGE = "cr.ai.cloud.ru/f51af5b1-d43b-4db4-938d-569d7cfffb7a/cuda12.1-torch2-py310-adaptive_attention:0.0.3"

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out"

def run_experiments(experiments, job_description_prefix="", dry=False):
    
    for exp in experiments:
        exp = deepcopy(exp)

        dummy_adaptive_fan_in_layers_str = exp.pop('dummy_adaptive_fan_in_layers_str')
        output_dir = exp.pop('output_dir')
        output_dir_full_path = os.path.join(workdir_prefix, output_dir)

        freeze_lm_backbone = exp.pop('freeze_lm_backbone')
        gumbel_tau = exp.pop('gumbel_tau', 1.0)

        full_unmerge_str_default: str = dummy_adaptive_fan_in_layers_str
        full_unmerge_str_default = full_unmerge_str_default.replace('1', '0') # all zeros
        full_unmerge_str = exp.pop('full_unmerge_str', full_unmerge_str_default)
        full_unmerge_loss_weight = exp.pop('full_unmerge_loss_weight', 0.0)

        warmup_steps = exp.pop('warmup_steps', 2000)
        num_train_epochs = exp.pop('num_train_epochs', 1)
        generate_merges_transform_impl = exp.pop('generate_merges_transform_impl', 'cuda_kernel')
        select_train_dataset_items = exp.pop('select_train_dataset_items', 150000)
        scale_not_pruned_gradients = exp.pop('scale_not_pruned_gradients', 0.0)
        merging_type = exp.pop('merging_type', 'hcg') # attention_output_mlp
        fan_out_type = exp.pop('fan_out_type', 'hcg') # residual_linear_projection
        sparsity_level = exp.pop('sparsity_level', 0.0)
        fan_out_projection = exp.pop('fan_out_projection', '1') # residual_linear_projection
        ce_merging_loss_weight = exp.pop('ce_merging_loss_weight', 0.0)
        hcg_loss_weight = exp.pop('hcg_loss_weight', 0.0)
        model_type = exp.pop('model_type', 'pretrained') # pretrained_checkpoint
        llama_checkpoint = exp.pop('llama_checkpoint', '""')
        hcg_loss_weight_dynamic = exp.pop('hcg_loss_weight_dynamic', '0')
        gumbel_loss_weight_dynamic = exp.pop('gumbel_loss_weight_dynamic', '0')

        learning_rate = exp.pop('learning_rate', 1e-4)
        per_device_train_batch_size = exp.pop('per_device_train_batch_size', 32)
        save_steps = exp.pop('save_steps', 10000)
        torch_compile = exp.pop('torch_compile', 1)

        concrete_random_mask_proba = exp.pop('concrete_random_mask_proba', '0')
        lm_loss_max_value = exp.pop('lm_loss_max_value', 1.5)
        hcg_loss_max_value = exp.pop('hcg_loss_max_value', 0.0)

        prohibit_end_of_sentence_pruning = exp.pop('prohibit_end_of_sentence_pruning', 0)
        scale_token_frequency = exp.pop('scale_token_frequency', 0)
        early_stopping_for_pretraining = exp.pop('early_stopping_for_pretraining', 0)

        instance_type = exp.pop('instance_type', INSTANCE_TYPE)

        if len(exp.keys()) > 0:
            raise ValueError(f"unknown parsms:{exp}")

        if instance_type == 'a100.4gpu':
            accelerate_config = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/accelerate_config_4gpu.yaml'
        elif instance_type == 'a100.1gpu':
            accelerate_config = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/accelerate_config_1gpu.yaml'
        else:
            raise ValueError(f"unknown instance_type:{instance_type}")

        seed = SEED

        script_str = f"/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/python /workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/accelerate launch --config_file {accelerate_config} {workdir_prefix}/src/transformers/models/llama/train_adaptive_llama.py --save_strategy steps --save_steps {save_steps} --generate_merges_transform_impl {generate_merges_transform_impl} --per_device_train_batch_size {per_device_train_batch_size} --learning_rate {learning_rate} --num_train_epochs {num_train_epochs} --seed {seed} --training_dataset smollm-corpus --model_type {model_type} --llama_checkpoint {llama_checkpoint} --dummy_adaptive_fan_in_layers_str {dummy_adaptive_fan_in_layers_str} --full_unmerge_str {full_unmerge_str} --full_unmerge_loss_weight {full_unmerge_loss_weight} --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type cosine --merging_type {merging_type} --fan_out_type {fan_out_type} --temperature_schedule 0 --freeze_lm_backbone {freeze_lm_backbone} --fan_out_projection {fan_out_projection} --warmup_steps {warmup_steps} --output_dir {output_dir_full_path} --max_steps_pretrain_fan_modules 0  --learnt_temperature 0 --select_train_dataset_items {select_train_dataset_items} --gumbel_tau {gumbel_tau} --weight_decay 0.1 --scale_not_pruned_gradients {scale_not_pruned_gradients} --hcg_loss_weight {hcg_loss_weight} --ce_merging_loss_weight {ce_merging_loss_weight} --gumbel_loss_weight_dynamic {gumbel_loss_weight_dynamic} --hcg_loss_weight_dynamic {hcg_loss_weight_dynamic} --bf16 1 --torch_compile {torch_compile} --sparsity_level {sparsity_level} --concrete_random_mask_proba {concrete_random_mask_proba} --lm_loss_max_value {lm_loss_max_value} --hcg_loss_max_value {hcg_loss_max_value} --prohibit_end_of_sentence_pruning {prohibit_end_of_sentence_pruning} --scale_token_frequency {scale_token_frequency} --early_stopping_for_pretraining {early_stopping_for_pretraining}"

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
            job_desc=f"{job_description_prefix}{output_dir} #rnd #multimodality",
            # stop_timer=600, # в минутах, = 10 часов
            env_variables={
                "PATH": "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/user/conda/bin",
                "WANDB_PROJECT": "adaptive_attention",
                "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                "WANDB_MODE": "online",
                "PYTHONPATH": f"{workdir_prefix}/src",
                "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface"
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
        "select_train_dataset_items": 80000,
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
        # {
        #     "dummy_adaptive_fan_in_layers_str": "0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_1",
        #     **common_params,
        # },
        {
            "dummy_adaptive_fan_in_layers_str": "1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_2",
            **common_params,
        },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_4",
        #     **common_params,
        # },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8",
            **common_params,
        },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_12",
        #     **common_params,
        # },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return


def run_hcg_smollm1dot7B_pretrain(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_1.7B_pretrain"

    common_params = {
        "freeze_lm_backbone": 1,
        "select_train_dataset_items": 80000,
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

def run_hcg_smollm2_360M_pretrain(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_360M_pretrain_with_early_stopping"

    common_params = {
        "freeze_lm_backbone": 1,
        "select_train_dataset_items": 160000,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 16,
        "llama_checkpoint": "HuggingFaceTB/SmolLM2-360M",
        "warmup_steps": 100,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "hcg_loss_weight": 2,
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",
        "scale_token_frequency": "0",
        "early_stopping_for_pretraining": '1',
        'instance_type': 'a100.1gpu',
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4",
            **common_params,
        },

    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return

def run_hcg_smollm2_1dot7B_pretrain(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_1.7B_pretrain"

    common_params = {
        "freeze_lm_backbone": 1,
        "select_train_dataset_items": 80000,
        "per_device_train_batch_size": 16,
        "llama_checkpoint": "HuggingFaceTB/SmolLM2-1.7B",
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

def run_hcg_smollm2_1dot7B_pretrain_nofoutproj(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_1.7B_pretrain_nofoutproj"

    common_params = {
        "freeze_lm_backbone": 1,
        "select_train_dataset_items": 80000,
        "per_device_train_batch_size": 16,
        "llama_checkpoint": "HuggingFaceTB/SmolLM2-1.7B",
        "warmup_steps": 100,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "1",
        "hcg_loss_weight": 10,
        "lm_loss_max_value": 1.25,
        "fan_out_projection": "0",
    }

    hcg_experiments = [
        # Fan out projection
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


def run_hcg_smollm2_1dot7B_nofoutproj(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_1.7B_nofoutproj"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 300000,
        "per_device_train_batch_size": 16,
        "model_type": "pretrained_checkpoint",
        "learning_rate": 0.00005,
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "1",
        "hcg_loss_weight": 10,
        "lm_loss_max_value": 1.25,
        "fan_out_projection": "0",
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_nofoutproj_4/checkpoint-4993",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,0,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_nofoutproj_8/checkpoint-4993",
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


def run_hcg_smollm360M_layers_iterate(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_360M_layersi"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 800000,
        "per_device_train_batch_size": 32,
        "model_type": "pretrained_checkpoint",
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "1",
        "hcg_loss_weight": 10,
        "lm_loss_max_value": 1.5,
        "fan_out_projection": "1",
    }

    hcg_experiments = [
        # Fan out projection
        # {
        #     "dummy_adaptive_fan_in_layers_str": "0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_1",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_360M_pretrain_1/checkpoint-1993",
        #     **common_params,
        # },
        {
            "dummy_adaptive_fan_in_layers_str": "1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_2",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_360M_pretrain_2/checkpoint-4993",
            **common_params,
        },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_4",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_360M_pretrain_4/checkpoint-1993",
        #     **common_params,
        # },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_360M_pretrain_8/checkpoint-4993/",
            **common_params,
        },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_12",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_360M_pretrain_12/checkpoint-1993",
        #     **common_params,
        # },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return


def run_hcg_smollm1dot7B_layers_iterate(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_1.7B_layersi_max_loss_1.1"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 300000,
        "per_device_train_batch_size": 16,
        "model_type": "pretrained_checkpoint",
        "learning_rate": 0.00005,
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
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_1.7B_pretrain_1/checkpoint-1993",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,0,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_2",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_1.7B_pretrain_2/checkpoint-1993",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_1.7B_pretrain_4/checkpoint-1993",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_1.7B_pretrain_8/checkpoint-1993",
            **common_params,
        },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return


def run_hcg_smollm1dot7B_layer_8(**kwargs):

    experiment_prefix_base_name = "run_hcg_smollm1dot7B_layer_8"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 800000,
        "per_device_train_batch_size": 16,
        "model_type": "pretrained_checkpoint",
        "learning_rate": 0.00005,
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "1",
        "hcg_loss_weight": 10,
        "fan_out_projection": "1",
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8_lmv_1.1",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_1.7B_pretrain_8/checkpoint-1993",
            "lm_loss_max_value": 1.1,
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8_lmv_1.25",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_1.7B_pretrain_8/checkpoint-1993",
            "lm_loss_max_value": 1.25,
            **common_params,
        },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return


def run_hcg_smollm2_1dot7B_layers_iterate(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_1.7B_layersi_max_loss_1.25_fix_fanoutproj"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 300000,
        "per_device_train_batch_size": 16,
        "model_type": "pretrained_checkpoint",
        "learning_rate": 0.00005,
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "1",
        "hcg_loss_weight": 10,
        "lm_loss_max_value": 1.25,
        "fan_out_projection": "1",
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "0,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_1",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_1/checkpoint-4993/",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,0,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_2",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_2/checkpoint-4993/",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_8/checkpoint-4993/",
            **common_params,
        },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return

def run_hcg_smollm2_1dot7B_hcg_scale_token_frequency(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_1.7B_hcg_scale_token_frequency"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 0,
        "per_device_train_batch_size": 16,
        "model_type": "pretrained_checkpoint",
        "learning_rate": 0.0002,
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",

        "scale_token_frequency": 1,
    }

    hcg_experiments = [
        # Fan out projection
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_2.0",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
        #     "hcg_loss_weight": 2.0,
        #     **common_params,
        # },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_2.0_no_eossp",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
            "hcg_loss_weight": 2.0,

            "prohibit_end_of_sentence_pruning": "1",
            **common_params,
        },

    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return

def run_hcg_smollm2_360M_hcg_scale_token_frequency(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_360M_hcg_scale_token_frequency_es_pretrain"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 0,
        "per_device_train_batch_size": 16,
        "model_type": "pretrained_checkpoint",
        "learning_rate": 0.0001,
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",

        "scale_token_frequency": 1,
    }

    hcg_experiments = []


    for hcg_loss_weight in [ 1.0, 2.0, 2.5, 3.0, 3.5 ]:
        exp_config = {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_{hcg_loss_weight}_no_eossp",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_360M_pretrain_with_early_stopping_4/TODO_CHECKPOINT/",
            "hcg_loss_weight": hcg_loss_weight,

            "prohibit_end_of_sentence_pruning": "1",
            **common_params,
        }
        hcg_experiments.append(exp_config)


    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return



def run_hcg_smollm2_1dot7B_hcg_prohibit_end_of_sentence_pruning(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_1.7B_hcg_prohibit_end_of_sentence_pruning"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 0,
        "per_device_train_batch_size": 8,
        "model_type": "pretrained_checkpoint",
        "learning_rate": 0.00005,
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",

        "prohibit_end_of_sentence_pruning": '1',
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_2.0_peosp_punkt",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
            "hcg_loss_weight": 2.0,
            **common_params,
        },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return

def run_hcg_smollm2_1dot7B_hcg_lambda_iterate(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_1.7B_hcg_lambda_iterate"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 0,
        "per_device_train_batch_size": 8,
        "model_type": "pretrained_checkpoint",
        "learning_rate": 0.00005,
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_1.5",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
            "hcg_loss_weight": 1.5,
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_1.75",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
            "hcg_loss_weight": 1.75,
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_2.0",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
            "hcg_loss_weight": 2.0,
            **common_params,
        },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_2.25",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
        #     "hcg_loss_weight": 2.25,
        #     **common_params,
        # },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_2.5",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
        #     "hcg_loss_weight": 2.5,
        #     **common_params,
        # },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_2.75",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
        #     "hcg_loss_weight": 2.75,
        #     **common_params,
        # },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_3",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
        #     "hcg_loss_weight": 3,

        #     **common_params,
        # },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return



def run_hcg_smollm2_1dot7B_fixed_pruning_percent(**kwargs):
    experiment_prefix_base_name = "adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 80000,
        "per_device_train_batch_size": 16,
        "model_type": "pretrained_checkpoint",
        "learning_rate": 0.00005,
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "lm_loss_max_value": 0.0,
        "hcg_loss_weight": 10,
        "fan_out_projection": "1",
    }

    hcg_experiments = [
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_8_pr_pct20",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_8/checkpoint-4993/",
        #     "hcg_loss_max_value": 0.8,
        #     **common_params,
        # },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_8_pr_pct30",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_8/checkpoint-4993/",
        #     "hcg_loss_max_value": 0.7,
        #     **common_params,
        # },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_8_pr_pct40",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_8/checkpoint-4993/",
        #     "hcg_loss_max_value": 0.6,
        #     **common_params,
        # },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_8_pr_pct60",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_8/checkpoint-4993/",
        #     "hcg_loss_max_value": 0.4,
        #     **common_params,
        # },
        # {
        #     "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
        #     "output_dir": f"{experiment_prefix_base_name}_8_pr_pct80",
        #     "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_8/checkpoint-4993/",
        #     "hcg_loss_max_value": 0.2,
        #     **common_params,
        # },
    ]

    # 4th layer
    for percent in [ 20, 30, 40, 60, 80 ]:
        exp_config = {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4_pr_pct{percent}",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_4/checkpoint-4993/",
            "hcg_loss_max_value": percent / 100,
            **common_params,
        }
        hcg_experiments.append(exp_config)

    # 2nd layer
    for percent in [ 20, 30, 40, 60, 80 ]:
        exp_config = {
            "dummy_adaptive_fan_in_layers_str": "1,0,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_2_pr_pct{percent}",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_2/checkpoint-4993/",
            "hcg_loss_max_value": percent / 100,
            **common_params,
        }
        hcg_experiments.append(exp_config)


    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return

def run_hcg_smollm1_1dot7B_fixed_pruning_percent(**kwargs):
    experiment_prefix_base_name = "adaptive_hcg_slm1_1.7B_layer_8_fixed_pruning_percent"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 300000,
        "per_device_train_batch_size": 16,
        "model_type": "pretrained_checkpoint",
        "learning_rate": 0.00005,
        "warmup_steps": 2000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "lm_loss_max_value": 0.0,
        "hcg_loss_weight": 10,
        "fan_out_projection": "1",
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8_pr_pct10",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_8/checkpoint-4993/",
            "hcg_loss_max_value": 0.90,
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8_pr_pct20",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_8/checkpoint-4993/",
            "hcg_loss_max_value": 0.80,
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8_pr_pct40",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_1.7B_pretrain_8/checkpoint-4993/",
            "hcg_loss_max_value": 0.6,
            **common_params,
        },
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return



def run_hcg_random_sampling(**kwargs):

    experiment_prefix_base_name = "random_hcg"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 400000,
        "warmup_steps": 2000,
        "model_type": "pretrained",
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": 1,
        "hcg_loss_weight": 0,
        "per_device_train_batch_size": 16,
        "llama_checkpoint": "HuggingFaceTB/SmolLM2-1.7B",
    }

    hcg_experiments = [
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4_random0.2_1.7B",
            "fan_out_projection": "1",
            "concrete_random_mask_proba": 0.2,
            **common_params,
        },
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,0,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_8_random0.2_1.7B",
            "fan_out_projection": "1",
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

    # Pretrain
    # run_hcg_smollm360M_pretrain(dry=dry)
    # run_hcg_smollm1dot7B_pretrain(dry=dry)
    # SmolLM2 pretrain
    # run_hcg_smollm2_1dot7B_pretrain(dry=dry)
    run_hcg_smollm2_360M_pretrain(dry=dry)

    # Iterate over layers
    # run_hcg_smollm360M_layers_iterate(dry=dry)
    # run_hcg_smollm1dot7B_layers_iterate(dry=dry)
    # run_hcg_smollm2_1dot7B_layers_iterate(dry=dry)

    # run_hcg_smollm2_1dot7B_hcg_lambda_iterate(dry=dry)
    # run_hcg_smollm2_1dot7B_hcg_prohibit_end_of_sentence_pruning(dry=dry)

    # run_hcg_smollm2_1dot7B_hcg_scale_token_frequency(dry=dry)
    # run_hcg_smollm2_360M_hcg_scale_token_frequency(dry=dry)

    # schedule pruned percent loss
    # run_hcg_smollm2_1dot7B_fixed_pruning_percent(dry=dry)

    # Strange 1.7B SmolLM2 8 Layer
    # run_hcg_smollm1dot7B_layer_8(dry=dry)
    # run_hcg_smollm1_1dot7B_fixed_pruning_percent(dry=dry)

    # Ablations
    # Pretrain no fan out projection
    # run_hcg_smollm2_1dot7B_pretrain_nofoutproj(dry=dry)
    # run_hcg_smollm2_1dot7B_nofoutproj(dry=dry)


    # Random
    # run_hcg_random_sampling(dry=dry)

    # Gumbel
    # run_gumbel_adaptive_pretrain(dry=dry)
    # run_gumbel_adaptive(dry=dry)