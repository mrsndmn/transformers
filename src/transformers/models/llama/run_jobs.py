import time
import string
import random
import client_lib # импортируем библиотеку для работы с ML Space

from rich.console import Console

import os

assert os.environ.get("WANDB_API_KEY", "") != "", "WANDB_API_KEY is required" 

from copy import deepcopy

REGION = "SR004"

SEED = 1008

INSTANCE_TYPE = "a100.4gpu"
N_WORKERS = 1
BASE_IMAGE = "cr.ai.cloud.ru/f51af5b1-d43b-4db4-938d-569d7cfffb7a/cuda12.1-torch2-py310-adaptive_attention:0.0.3"

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out"

def run_experiments(experiments, job_description_prefix="", dry=False):

    for exp in experiments:
        exp = deepcopy(exp)

        dummy_adaptive_fan_in_layers_str = exp.pop('dummy_adaptive_fan_in_layers_str')
        output_dir = exp.pop('output_dir')

        output_dir += f"_{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}"

        output_dir_full_path = os.path.join(workdir_prefix, output_dir)

        freeze_lm_backbone = exp.pop('freeze_lm_backbone')

        warmup_steps = exp.pop('warmup_steps', 2000)
        num_train_epochs = exp.pop('num_train_epochs', 1)
        generate_merges_transform_impl = exp.pop('generate_merges_transform_impl', 'cuda_kernel')
        select_train_dataset_items = exp.pop('select_train_dataset_items', 150000)
        scale_not_pruned_gradients = exp.pop('scale_not_pruned_gradients', 0.0)
        merging_type = exp.pop('merging_type', 'hcg') # attention_output_mlp
        fan_out_type = exp.pop('fan_out_type', 'hcg') # residual_linear_projection
        sparsity_level = exp.pop('sparsity_level', 0.0)
        fan_out_projection = exp.pop('fan_out_projection', '1') # residual_linear_projection
        hcg_loss_weight = exp.pop('hcg_loss_weight', 0.0)
        model_type = exp.pop('model_type', 'pretrained') # pretrained_checkpoint
        llama_checkpoint = exp.pop('llama_checkpoint', '""')
        hcg_loss_weight_dynamic = exp.pop('hcg_loss_weight_dynamic', '0')

        learning_rate = exp.pop('learning_rate', 1e-4)
        hcg_learning_rate = exp.pop('hcg_learning_rate', 1e-3)
        per_device_train_batch_size = exp.pop('per_device_train_batch_size', 32)
        gradient_accumulation_steps = exp.pop('gradient_accumulation_steps', 1)

        save_steps = exp.pop('save_steps', 10000)
        torch_compile = exp.pop('torch_compile', 1)

        concrete_random_mask_proba = exp.pop('concrete_random_mask_proba', '0')
        concrete_uniform_pruning = exp.pop('concrete_uniform_pruning', '0')
        concrete_stop_word_pruning = exp.pop('concrete_stop_word_pruning', '0')

        lm_loss_max_value = exp.pop('lm_loss_max_value', 1.5)
        hcg_loss_max_value = exp.pop('hcg_loss_max_value', 0.0)

        prohibit_end_of_sentence_pruning = exp.pop('prohibit_end_of_sentence_pruning', 0)
        early_stopping_for_pretraining = exp.pop('early_stopping_for_pretraining', 0)
        pretrain_fan_out_projection = exp.pop('pretrain_fan_out_projection', 0)

        instance_type = exp.pop('instance_type', INSTANCE_TYPE)

        eval_strategy = exp.pop('eval_strategy', 'steps')

        if len(exp.keys()) > 0:
            raise ValueError(f"unknown parsms:{exp}")

        if instance_type == 'a100.4gpu':
            accelerate_config = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/accelerate_config_4gpu.yaml'
        elif instance_type == 'a100.2gpu':
            accelerate_config = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/accelerate_config_2gpu.yaml'
        elif instance_type == 'a100.1gpu':
            accelerate_config = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/accelerate_config_1gpu.yaml'
        else:
            raise ValueError(f"unknown instance_type:{instance_type}")

        seed = SEED

        script_str = f"/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/python /workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/accelerate launch --config_file {accelerate_config} {workdir_prefix}/src/transformers/models/llama/train_adaptive_llama.py --save_strategy steps --save_steps {save_steps} --generate_merges_transform_impl {generate_merges_transform_impl} --per_device_train_batch_size {per_device_train_batch_size} --learning_rate {learning_rate} --num_train_epochs {num_train_epochs} --seed {seed} --training_dataset smollm-corpus --model_type {model_type} --llama_checkpoint {llama_checkpoint} --dummy_adaptive_fan_in_layers_str {dummy_adaptive_fan_in_layers_str} --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type cosine --merging_type {merging_type} --fan_out_type {fan_out_type} --temperature_schedule 0 --freeze_lm_backbone {freeze_lm_backbone} --fan_out_projection {fan_out_projection} --warmup_steps {warmup_steps} --output_dir {output_dir_full_path} --learnt_temperature 0 --select_train_dataset_items {select_train_dataset_items} --weight_decay 0.1 --scale_not_pruned_gradients {scale_not_pruned_gradients} --hcg_loss_weight {hcg_loss_weight} --hcg_loss_weight_dynamic {hcg_loss_weight_dynamic} --bf16 1 --torch_compile {torch_compile} --sparsity_level {sparsity_level} --concrete_random_mask_proba {concrete_random_mask_proba} --lm_loss_max_value {lm_loss_max_value} --hcg_loss_max_value {hcg_loss_max_value} --prohibit_end_of_sentence_pruning {prohibit_end_of_sentence_pruning} --early_stopping_for_pretraining {early_stopping_for_pretraining} --hcg_learning_rate {hcg_learning_rate} --gradient_accumulation_steps {gradient_accumulation_steps} --pretrain_fan_out_projection {pretrain_fan_out_projection} --eval_strategy {eval_strategy} --concrete_uniform_pruning {concrete_uniform_pruning} --concrete_stop_word_pruning {concrete_stop_word_pruning}"

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
            job_desc=f"{job_description_prefix}{output_dir} #rnd #multimodality #tarasov",
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
        "fan_out_projection": "0",
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
        "learning_rate": 0.001,
        "hcg_learning_rate": 0.001,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "hcg_loss_weight": -1.0,
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",
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


def run_hcg_llama31_8B_pretrain(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_llama_8B_pretrain"

    common_params = {
        "freeze_lm_backbone": 1,
        "select_train_dataset_items": 160000,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "llama_checkpoint": "unsloth/Meta-Llama-3.1-8B",
        "warmup_steps": 100,
        "learning_rate": 0.001,
        "hcg_learning_rate": 0.001,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "hcg_loss_weight": -1.0,
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",

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

def run_hcg_llama31_8B_pretrain_fan_out_projection(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_llama_8B_pretrain_fan_out_projection"

    common_params = {
        "freeze_lm_backbone": 1,
        "select_train_dataset_items": 50000,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 16,
        "model_type": "pretrained_checkpoint",
        "warmup_steps": 1000,
        "learning_rate": 0.0001,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "hcg_loss_weight": 0.0,
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",
        "pretrain_fan_out_projection": "1",
        'instance_type': 'a100.1gpu',
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_llama_8B_pretrain_4/checkpoint-601/",
            **common_params,
        },

    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return


def run_hcg_llama31_8B_train_iterative(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_llama_8B_train_iterative"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 15001,
        "per_device_train_batch_size": 1,
        "eval_strategy": "no",
        "model_type": "pretrained_checkpoint",
        "learning_rate": 0.00005,
        "warmup_steps": 3000,
        "torch_compile": 0,
        "hcg_loss_weight_dynamic": "1",
        "lm_loss_max_value": 2.0,
        "hcg_loss_weight": 10,
        "fan_out_projection": "1",
        "instance_type": "a100.1gpu",

        "prohibit_end_of_sentence_pruning": '1',
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_lm_loss_max_value_1.1_peosp_punkt",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_llama_8B_pretrain_fan_out_projection_4/checkpoint-1556/",
            **common_params,
        }
    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return



def run_hcg_qwen_7B_pretrain(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_qwen_7B_pretrain"

    common_params = {
        "freeze_lm_backbone": 1,
        "select_train_dataset_items": 160000,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "llama_checkpoint": "Qwen/Qwen2.5-7B",
        "warmup_steps": 100,
        "learning_rate": 0.001,
        "hcg_learning_rate": 0.001,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "hcg_loss_weight": -1.0,
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",

        "early_stopping_for_pretraining": '1',
        'instance_type': 'a100.1gpu',
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4",
            **common_params,
        },

    ]

    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return



def run_hcg_smollm2_360M_pretrain_fan_out_projection(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_360M_pretrain_fan_out_projection"

    common_params = {
        "freeze_lm_backbone": 1,
        "select_train_dataset_items": 50000,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 16,
        "model_type": "pretrained_checkpoint",
        "warmup_steps": 1000,
        "learning_rate": 0.0001,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "hcg_loss_weight": 0.0,
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",
        "pretrain_fan_out_projection": "1",
        'instance_type': 'a100.1gpu',
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_360M_pretrain_with_early_stopping_4/checkpoint-601/",
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

def run_hcg_smollm2_360M_hcg(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_360M"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 0,
        "model_type": "pretrained_checkpoint",

        "learning_rate": 0.0005,
        "gradient_accumulation_steps": 1,
        "per_device_train_batch_size": 8,
        "instance_type": "a100.1gpu",

        "warmup_steps": 5000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",
        "pretrain_fan_out_projection": '0',
    }

    hcg_experiments = []

    # for hcg_loss_weight in [ 1.0, 2.0, 2.5, 3.0, 3.5 ]:
    # for hcg_loss_weight in [ 1.1, 1.5 ]:
    for hcg_loss_weight in [ 1.0, 1.5 ]:
        exp_config = {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_{hcg_loss_weight}_{common_params['instance_type']}",
            "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_360M_pretrain_fan_out_projection_4/_no_fout_proj_checkpoint-3118/",
            "hcg_loss_weight": hcg_loss_weight,

            "prohibit_end_of_sentence_pruning": "1",
            **common_params,
        }
        hcg_experiments.append(exp_config)


    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return

def run_hcg_smollm2_360M_hcg_rule_based(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_360M_rule_based"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 300000,
        "model_type": "pretrained_checkpoint",

        "learning_rate": 0.0005,
        "gradient_accumulation_steps": 1,
        "per_device_train_batch_size": 8,
        "instance_type": "a100.1gpu",

        "warmup_steps": 5000,
        "torch_compile": 1,
        "hcg_loss_weight_dynamic": "0",
        "lm_loss_max_value": 0.0,
        "fan_out_projection": "1",
        "pretrain_fan_out_projection": '0',
        "hcg_loss_weight": 0,
    }

    hcg_experiments = []

    for freeze_lm_backbone in [ 0, 1 ]:
        current_hcg_experiments = [
            {
                "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
                "output_dir": f"{experiment_prefix_base_name}_random_0.1_freeze_{freeze_lm_backbone}",
                "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_360M_pretrain_fan_out_projection_4/_no_fout_proj_checkpoint-3118/",
                "concrete_random_mask_proba": 0.1,
                **common_params,
            },
            {
                "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
                "output_dir": f"{experiment_prefix_base_name}_random_0.2_freeze_{freeze_lm_backbone}",
                "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_360M_pretrain_fan_out_projection_4/_no_fout_proj_checkpoint-3118/",
                "concrete_random_mask_proba": 0.2,
                **common_params,
            },
            {
                "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
                "output_dir": f"{experiment_prefix_base_name}_uniform_0.1_freeze_{freeze_lm_backbone}",
                "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_360M_pretrain_fan_out_projection_4/_no_fout_proj_checkpoint-3118/",
                "concrete_uniform_pruning": 10, # 10% of the tokens
                **common_params,
            },
            {
                "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
                "output_dir": f"{experiment_prefix_base_name}_uniform_0.2_freeze_{freeze_lm_backbone}",
                "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_360M_pretrain_fan_out_projection_4/_no_fout_proj_checkpoint-3118/",
                "concrete_uniform_pruning": 5, # 20% of the tokens
                **common_params,
            },
            {
                "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
                "output_dir": f"{experiment_prefix_base_name}_stop_words_pruning_freeze_{freeze_lm_backbone}",
                "llama_checkpoint": f"{workdir_prefix}/adaptive_hcg_slm2_360M_pretrain_fan_out_projection_4/_no_fout_proj_checkpoint-3118/",
                "concrete_stop_word_pruning": 1,
                **common_params,
            },
        ]

        for exp in current_hcg_experiments:
            exp["freeze_lm_backbone"] = freeze_lm_backbone

        hcg_experiments += current_hcg_experiments

    run_experiments(hcg_experiments, job_description_prefix="HCG Rule Based: ", **kwargs)

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
    import subprocess

    console = Console()
    console.print(client_lib.get_instance_types(regions="SR004"))

    dry = len(sys.argv) > 1 and sys.argv[1] == 'dry'
    print("dry", dry)

    if not dry:
        tests_run = subprocess.run(["pytest", "src/transformers/models/llama/tests/"])
        if tests_run.returncode != 0:
            print("Tests failed")
            exit(1)

    # Pretrain
    # run_hcg_smollm360M_pretrain(dry=dry)
    # run_hcg_smollm1dot7B_pretrain(dry=dry)
    # SmolLM2 pretrain
    # run_hcg_smollm2_1dot7B_pretrain(dry=dry)

    # Llama 8B
    # run_hcg_llama31_8B_pretrain(dry=dry)
    # run_hcg_llama31_8B_pretrain_fan_out_projection(dry=dry)
    # run_hcg_llama31_8B_train_iterative(dry=dry)

    # Qwen 7B
    # run_hcg_qwen_7B_pretrain(dry=dry)
    # run_hcg_qwen_7B_pretrain_fan_out_projection(dry=dry)

    # run_hcg_smollm2_360M_pretrain(dry=dry)
    # run_hcg_smollm2_360M_pretrain_fan_out_projection(dry=dry)

    # Iterate over layers
    # run_hcg_smollm360M_layers_iterate(dry=dry)
    # run_hcg_smollm1dot7B_layers_iterate(dry=dry)
    # run_hcg_smollm2_1dot7B_layers_iterate(dry=dry)

    # run_hcg_smollm2_1dot7B_hcg_lambda_iterate(dry=dry)
    # run_hcg_smollm2_1dot7B_hcg_prohibit_end_of_sentence_pruning(dry=dry)

    # run_hcg_smollm2_1dot7B_hcg_scale_token_frequency(dry=dry)
    # run_hcg_smollm2_360M_hcg(dry=dry)
    run_hcg_smollm2_360M_hcg_rule_based(dry=dry)

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