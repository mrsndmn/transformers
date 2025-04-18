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

        warmup_steps = exp.pop('warmup_steps', 5000)
        num_train_epochs = exp.pop('num_train_epochs', 1)
        generate_merges_transform_impl = exp.pop('generate_merges_transform_impl', 'cuda_kernel')
        select_train_dataset_items = exp.pop('select_train_dataset_items', 150000)
        scale_not_pruned_gradients = exp.pop('scale_not_pruned_gradients', 0.0)
        merging_type = exp.pop('merging_type', 'hcg') # attention_output_mlp
        sparsity_level = exp.pop('sparsity_level', 0.0)
        fan_out_projection = exp.pop('fan_out_projection', '1') # residual_linear_projection
        hcg_loss_weight = exp.pop('hcg_loss_weight', 0.0)
        model_type = exp.pop('model_type', 'pretrained') # pretrained_checkpoint
        llama_checkpoint = exp.pop('llama_checkpoint', '""')
        hcg_loss_weight_dynamic = exp.pop('hcg_loss_weight_dynamic', '0')

        learning_rate = exp.pop('learning_rate', 1e-4)
        hcg_learning_rate = exp.pop('hcg_learning_rate', learning_rate)
        lr_scheduler_type = exp.pop('lr_scheduler_type', 'cosine')
        max_grad_norm = exp.pop('max_grad_norm', 1)

        hard_hcg_log_a = exp.pop('hard_hcg_log_a', 0)

        init_hcg_a = exp.pop('init_hcg_a', '')
        if init_hcg_a != '':
            init_hcg_a = f"--init_hcg_a {init_hcg_a}"

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

        single_layer_hopping = exp.pop('single_layer_hopping', 0)

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

        script_str = f"/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/python /workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/accelerate launch --config_file {accelerate_config} {workdir_prefix}/src/transformers/models/llama/train_adaptive_llama.py --save_strategy steps --save_steps {save_steps} --generate_merges_transform_impl {generate_merges_transform_impl} --per_device_train_batch_size {per_device_train_batch_size} --learning_rate {learning_rate} --hcg_learning_rate {hcg_learning_rate} {init_hcg_a} --hard_hcg_log_a {hard_hcg_log_a} --max_grad_norm {max_grad_norm} --num_train_epochs {num_train_epochs} --seed {seed} --training_dataset smollm-corpus --model_type {model_type} --llama_checkpoint {llama_checkpoint} --dummy_adaptive_fan_in_layers_str {dummy_adaptive_fan_in_layers_str} --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type {lr_scheduler_type} --merging_type {merging_type} --temperature_schedule 0 --freeze_lm_backbone {freeze_lm_backbone} --fan_out_projection {fan_out_projection} --warmup_steps {warmup_steps} --output_dir {output_dir_full_path} --learnt_temperature 0 --select_train_dataset_items {select_train_dataset_items} --weight_decay 0.1 --scale_not_pruned_gradients {scale_not_pruned_gradients} --hcg_loss_weight {hcg_loss_weight} --hcg_loss_weight_dynamic {hcg_loss_weight_dynamic} --bf16 1 --torch_compile {torch_compile} --sparsity_level {sparsity_level} --concrete_random_mask_proba {concrete_random_mask_proba} --lm_loss_max_value {lm_loss_max_value} --hcg_loss_max_value {hcg_loss_max_value} --prohibit_end_of_sentence_pruning {prohibit_end_of_sentence_pruning} --early_stopping_for_pretraining {early_stopping_for_pretraining} --gradient_accumulation_steps {gradient_accumulation_steps} --pretrain_fan_out_projection {pretrain_fan_out_projection} --eval_strategy {eval_strategy} --concrete_uniform_pruning {concrete_uniform_pruning} --concrete_stop_word_pruning {concrete_stop_word_pruning} --single_layer_hopping {single_layer_hopping}"

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
                "PYTHONPATH": f"{workdir_prefix}/src:/workspace-SR004.nfs2/d.tarasov/lighteval/src",
                "HF_HOME": "/workspace-SR004.nfs2/.cache/huggingface"
            },
        )

        if dry:
            print("JOB WAS NOT LAUNCHED")
        else:
            print(output_dir, job_w_args.submit())

    return


# ==
# == SmallLM 360 Rule based Pruning
# ==


def run_hcg_smollm2_360M_hcg_rule_based(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_360M_rule_based"

    common_params = {
        "freeze_lm_backbone": 0,
        "select_train_dataset_items": 1200000,
        "model_type": "pretrained",
        "llama_checkpoint": "HuggingFaceTB/SmolLM2-360M",

        "learning_rate": 0.0005,
        "gradient_accumulation_steps": 1,
        "per_device_train_batch_size": 16,
        "instance_type": "a100.1gpu",

        "hcg_loss_weight": 0,
    }

    hcg_experiments = []

    for freeze_lm_backbone in [ 0, 1 ]:
        current_hcg_experiments = [
            {
                "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
                "output_dir": f"{experiment_prefix_base_name}_random_0.1_freeze_{freeze_lm_backbone}",
                "concrete_random_mask_proba": 0.1,
                **common_params,
            },
            {
                "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
                "output_dir": f"{experiment_prefix_base_name}_random_0.2_freeze_{freeze_lm_backbone}",
                "concrete_random_mask_proba": 0.2,
                **common_params,
            },
            {
                "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
                "output_dir": f"{experiment_prefix_base_name}_uniform_0.1_freeze_{freeze_lm_backbone}",
                "concrete_uniform_pruning": 10, # 10% of the tokens
                **common_params,
            },
            {
                "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
                "output_dir": f"{experiment_prefix_base_name}_uniform_0.2_freeze_{freeze_lm_backbone}",
                "concrete_uniform_pruning": 5, # 20% of the tokens
                **common_params,
            },
            {
                "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1",
                "output_dir": f"{experiment_prefix_base_name}_stop_words_pruning_freeze_{freeze_lm_backbone}",
                "concrete_stop_word_pruning": 1,
                **common_params,
            },
        ]

        for exp in current_hcg_experiments:
            exp["freeze_lm_backbone"] = freeze_lm_backbone

        hcg_experiments += current_hcg_experiments

    run_experiments(hcg_experiments, job_description_prefix="HCG Rule Based: ", **kwargs)

    return


# ==
# == SmallLM 360
# ==

def run_hcg_smollm2_360M_pretrain_fan_out_projection(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_360M_pretrain_fan_out_projection"

    common_params = {
        # Model
        "model_type": "pretrained",
        "llama_checkpoint": "HuggingFaceTB/SmolLM2-360M",
        "freeze_lm_backbone": 1,

        # Data
        "select_train_dataset_items": 1200000,
        "per_device_train_batch_size": 16,

        # Training
        "learning_rate": 0.0005,
        "hcg_learning_rate": 0.0,

        "hcg_loss_weight": 0.0,
        "max_grad_norm": 1,

        'instance_type': 'a100.2gpu',

        # Training type
        "pretrain_fan_out_projection": "1",
    }

    hcg_experiments = [
        # Fan out projection
        {
            "dummy_adaptive_fan_in_layers_str": "1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1",
            "output_dir": f"{experiment_prefix_base_name}_4",
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


def run_hcg_smollm2_360M_hcg(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_360M"

    common_params = {
        # Model
        "model_type": "pretrained_checkpoint",
        # "llama_checkpoint": # will be overriden in cycle later,
        "freeze_lm_backbone": "1",
        "init_hcg_a": 1.0,

        # Data
        "select_train_dataset_items": 1000000,
        "per_device_train_batch_size": 4,

        # Training
        "learning_rate": 0.02,
        "hcg_learning_rate": 0.02,
        "lr_scheduler_type": "constant_with_warmup",

        # "hcg_loss_weight": # will be overriden in cycle later,
        "max_grad_norm": 0,

        'instance_type': 'a100.2gpu',
        'num_train_epochs': 2,
        "hcg_loss_weight": '0.01',

        # Training type
        "pretrain_fan_out_projection": "0",
        "single_layer_hopping": 0,
    }

    hcg_experiments = []

    extra_params_from_scratch = [
        (
            "8",
            "1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1",
        ),
        (
            "12",
            "1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1",
        ),
    ]

    for suffix, in_layers_str in extra_params_from_scratch:
        exp_config = {
            "dummy_adaptive_fan_in_layers_str": in_layers_str,
            **common_params,
        }
        weight = exp_config['hcg_loss_weight']
        exp_config["output_dir"] = f"{experiment_prefix_base_name}_w_{float(weight):.3f}_l_{suffix}_no_self_attn"

        exp_config['model_type'] = 'pretrained'
        exp_config['llama_checkpoint'] = 'HuggingFaceTB/SmolLM2-360M'

        hcg_experiments.append(exp_config)


    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return

def run_hcg_smollm2_1_7B_hcg(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_1.7B"

    common_params = {
        # Model
        "model_type": "pretrained_checkpoint",
        # "llama_checkpoint": # will be overriden in cycle later,
        "freeze_lm_backbone": "1",
        "init_hcg_a": 1.0,

        # Data
        "select_train_dataset_items": 1000000,
        "per_device_train_batch_size": 4,

        # Training
        "learning_rate": 0.02,
        "hcg_learning_rate": 0.02,
        "lr_scheduler_type": "constant_with_warmup",

        # "hcg_loss_weight": # will be overriden in cycle later,
        "max_grad_norm": 0,

        'instance_type': 'a100.2gpu',
        'num_train_epochs': 2,
        "hcg_loss_weight": '0.01',

        # Training type
        "pretrain_fan_out_projection": "0",
        "single_layer_hopping": 0,
    }

    hcg_experiments = []

    extra_params_from_scratch = [
        (
            "8",
            "1,1,1,1,1,1,1,0,1,1,1,1",
        ),
        (
            "10",
            "1,1,1,1,1,1,1,1,1,0,1,1",
        ),
    ]

    for suffix, in_layers_str in extra_params_from_scratch:
        exp_config = {
            "dummy_adaptive_fan_in_layers_str": in_layers_str,
            **common_params,
        }
        weight = exp_config['hcg_loss_weight']
        exp_config["output_dir"] = f"{experiment_prefix_base_name}_w_{float(weight):.3f}_l_{suffix}_no_self_attn"

        exp_config['model_type'] = 'pretrained'
        exp_config['llama_checkpoint'] = 'HuggingFaceTB/SmolLM2-1.7B'

        hcg_experiments.append(exp_config)


    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return


def run_hcg_llama31_8B_hcg(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_llama31_8B"

    common_params = {
        # Model
        "model_type": "pretrained",
        "llama_checkpoint": "unsloth/Meta-Llama-3.1-8B",

        "freeze_lm_backbone": "1",
        "init_hcg_a": 1.0,

        # Data
        "select_train_dataset_items": 1000000,
        "per_device_train_batch_size": 4,

        # Training
        "learning_rate": 0.02,
        "hcg_learning_rate": 0.02,
        "lr_scheduler_type": "constant_with_warmup",

        # "hcg_loss_weight": # will be overriden in cycle later,
        "max_grad_norm": 0,

        'instance_type': 'a100.2gpu',
        'num_train_epochs': 1,
        "hcg_loss_weight": '0.01',

        # Training type
        "pretrain_fan_out_projection": "0",
        "single_layer_hopping": 0,
    }

    hcg_experiments = []

    extra_params_from_scratch = [
        (
            "12",
            "1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1",
        ),
        (
            "8",
            "1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1",
        ),
    ]

    for suffix, in_layers_str in extra_params_from_scratch:
        exp_config = {
            "dummy_adaptive_fan_in_layers_str": in_layers_str,
            **common_params,
        }
        weight = exp_config['hcg_loss_weight']
        exp_config["output_dir"] = f"{experiment_prefix_base_name}_w_{float(weight):.3f}_l_{suffix}_no_self_attn"

        hcg_experiments.append(exp_config)


    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return



def run_hcg_smollm2_360M_hcg_post_training(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_slm2_360M_post_training"

    common_params = {
        # Model
        "model_type": "pretrained_checkpoint",
        # "llama_checkpoint": # will be overriden in cycle later,
        "freeze_lm_backbone": "1",
        "hard_hcg_log_a": 1,

        # Data
        "select_train_dataset_items": 1000000,
        "per_device_train_batch_size": 4,

        # Training
        "learning_rate": 0.02,
        "hcg_learning_rate": 0.02,
        "lr_scheduler_type": "cosine",

        # "hcg_loss_weight": # will be overriden in cycle later,
        "max_grad_norm": 0,

        'instance_type': 'a100.2gpu',

        # Training type
        "pretrain_fan_out_projection": "0",
    }

    hcg_experiments = []
    extra_params = [
        (
            "12",
            "1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1",
            "./adaptive_hcg_slm2_360M_w_0.001_l_12_MPKM3VH1/checkpoint-120000/",
        ),
        # (
        #     "4",
        #     "1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1",
        #     "adaptive_hcg_slm2_360M_pretrain_fan_out_projection_4_RG0781E8/checkpoint-37496",
        # ),
    ]

    for hcg_loss_weight in [ 0.001, ]:
    # , 0.01,
    # for hcg_loss_weight in [ 0.0 ]:
    # for hcg_loss_weight in [ 0.1 ]:
        for suffix, in_layers_str, llama_checkpoint in extra_params:
            exp_config = {
                "dummy_adaptive_fan_in_layers_str": in_layers_str,
                "output_dir": f"{experiment_prefix_base_name}_w_{hcg_loss_weight}_l_{suffix}",
                "llama_checkpoint": f"{workdir_prefix}/{llama_checkpoint}",
                "hcg_loss_weight": hcg_loss_weight,

                **common_params,
            }
            hcg_experiments.append(exp_config)


    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

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


    # SLM360M
    # run_hcg_smollm2_360M_pretrain_fan_out_projection(dry=dry)
    # run_hcg_smollm2_360M_hcg(dry=dry)

    # run_hcg_smollm2_1_7B_hcg(dry=dry)
    run_hcg_llama31_8B_hcg(dry=dry)

    # run_hcg_smollm2_360M_hcg_post_training(dry=dry)

    # Rule based
    # run_hcg_smollm2_360M_hcg_rule_based(dry=dry)

