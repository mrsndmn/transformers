import time
import string
import random
import client_lib # импортируем библиотеку для работы с ML Space

from rich.console import Console

import os

assert os.environ.get("WANDB_API_KEY", "") != "", "WANDB_API_KEY is required" 

from copy import deepcopy

from transformers.models.llama.extra_types import AVAILABLE_OPTIMIZED_PARAMS

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

        optimized_params = exp.pop('optimized_params')
        for param in optimized_params.split(','):
            assert param in AVAILABLE_OPTIMIZED_PARAMS, f'{param} is not in {AVAILABLE_OPTIMIZED_PARAMS}'

        warmup_steps = exp.pop('warmup_steps', 2000)
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

        force_train_on_trimmed_embeddings = exp.pop('force_train_on_trimmed_embeddings', '0')

        dataset = exp.pop('dataset', '')
        if dataset is None:
            dataset = ''

        if dataset != '':
            dataset = f"--dataset {dataset}"

        optim = exp.pop('optim', '')
        if optim != '':
            optim = f"--optim {optim}"

        gradient_checkpointing = exp.pop('gradient_checkpointing', '0')
        hcg_fan_in_from = exp.pop('hcg_fan_in_from', '')
        if hcg_fan_in_from is None:
            hcg_fan_in_from = ''
        if hcg_fan_in_from != '':
            if hcg_fan_in_from.startswith('./'):
                hcg_fan_in_from = os.path.join(workdir_prefix, hcg_fan_in_from)
            hcg_fan_in_from = f"--hcg_fan_in_from {hcg_fan_in_from}"

        fan_out_projection_mlp_intermediate_size = exp.pop('fan_out_projection_mlp_intermediate_size', '')
        if fan_out_projection_mlp_intermediate_size is None:
            fan_out_projection_mlp_intermediate_size = ''
        if fan_out_projection_mlp_intermediate_size != '':
            fan_out_projection_mlp_intermediate_size = f"--fan_out_projection_mlp_intermediate_size {fan_out_projection_mlp_intermediate_size}"

        hard_hcg_log_a = exp.pop('hard_hcg_log_a', 0)
        init_hcg_a = exp.pop('init_hcg_a', '')
        if init_hcg_a != '':
            init_hcg_a = f"--init_hcg_a {init_hcg_a}"

        logging_steps = exp.pop('logging_steps', "")
        if logging_steps == "":
            logging_steps = '1'

        logging_steps = f"--logging_steps {logging_steps}"

        per_device_train_batch_size = exp.pop('per_device_train_batch_size', 32)
        gradient_accumulation_steps = exp.pop('gradient_accumulation_steps', '')
        if gradient_accumulation_steps != '':
            gradient_accumulation_steps = f"--gradient_accumulation_steps {gradient_accumulation_steps}"

        save_steps = exp.pop('save_steps', 5000)
        save_total_limit = exp.pop('save_total_limit', "")
        if save_total_limit != "":
            save_total_limit = f"--save_total_limit {save_total_limit}"
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

        add_end_of_sentence_token = exp.pop('add_end_of_sentence_token', 0)

        eval_steps = exp.pop('eval_steps', 500)
        init_fan_out_mlp = exp.pop('init_fan_out_mlp', 0)

        each_layer_pruning = exp.pop('each_layer_pruning', 0)

        forward_residuals = exp.pop('forward_residuals', 0)
        fan_in_idx = exp.pop('fan_in_idx', '')
        if fan_in_idx != '':
            fan_in_idx = f"--fan_in_idx {fan_in_idx}"
        fan_out_idx = exp.pop('fan_out_idx', '')
        if fan_out_idx != '':
            fan_out_idx = f"--fan_out_idx {fan_out_idx}"

        if len(exp.keys()) > 0:
            raise ValueError(f"unknown parsms:{exp}")

        do_eval_on_save = 0

        is_fsdp = instance_type.endswith('fsdp')
        instance_type = instance_type.removesuffix('_fsdp')

        save_only_model = ""

        if instance_type == 'a100.8gpu':
            if is_fsdp:
                accelerate_config = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/accelerate_config_8gpu_fsdp.yaml'
            else:
                accelerate_config = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/accelerate_config_8gpu.yaml'
        elif instance_type == 'a100.6gpu':
            accelerate_config = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/accelerate_config_6gpu.yaml'
        elif instance_type == 'a100.4gpu':
            accelerate_config = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/accelerate_config_4gpu.yaml'
        elif instance_type == 'a100.2gpu':
            accelerate_config = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/accelerate_config_2gpu.yaml'
        elif instance_type == 'a100.1gpu':
            # do_eval_on_save = 1
            accelerate_config = '/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/accelerate_config_1gpu.yaml'
        else:
            raise ValueError(f"unknown instance_type:{instance_type}")

        seed = SEED

        script_str = f"/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/python /workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/accelerate launch --config_file {accelerate_config} {workdir_prefix}/src/transformers/models/llama/train_adaptive_llama.py --save_strategy steps --save_steps {save_steps} --generate_merges_transform_impl {generate_merges_transform_impl} --per_device_train_batch_size {per_device_train_batch_size} --learning_rate {learning_rate} --hcg_learning_rate {hcg_learning_rate} {init_hcg_a} --hard_hcg_log_a {hard_hcg_log_a} --max_grad_norm {max_grad_norm} --num_train_epochs {num_train_epochs} --seed {seed} --model_type {model_type} --llama_checkpoint {llama_checkpoint} --dummy_adaptive_fan_in_layers_str {dummy_adaptive_fan_in_layers_str} --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type {lr_scheduler_type} --merging_type {merging_type} --temperature_schedule 0 --optimized_params {optimized_params} --fan_out_projection {fan_out_projection} --warmup_steps {warmup_steps} --output_dir {output_dir_full_path} --learnt_temperature 0 --select_train_dataset_items {select_train_dataset_items} --weight_decay 0.1 --scale_not_pruned_gradients {scale_not_pruned_gradients} --hcg_loss_weight {hcg_loss_weight} --hcg_loss_weight_dynamic {hcg_loss_weight_dynamic} --bf16 1 --torch_compile {torch_compile} --sparsity_level {sparsity_level} --concrete_random_mask_proba {concrete_random_mask_proba} --lm_loss_max_value {lm_loss_max_value} --hcg_loss_max_value {hcg_loss_max_value} --prohibit_end_of_sentence_pruning {prohibit_end_of_sentence_pruning} --early_stopping_for_pretraining {early_stopping_for_pretraining} {gradient_accumulation_steps} --pretrain_fan_out_projection {pretrain_fan_out_projection} --eval_strategy {eval_strategy} --concrete_uniform_pruning {concrete_uniform_pruning} --concrete_stop_word_pruning {concrete_stop_word_pruning} --each_layer_pruning {each_layer_pruning} {fan_in_idx} {fan_out_idx} --forward_residuals {forward_residuals} --eval_steps {eval_steps} --init_fan_out_mlp {init_fan_out_mlp} {hcg_fan_in_from} {fan_out_projection_mlp_intermediate_size} --do_eval_on_save {do_eval_on_save} --gradient_checkpointing {gradient_checkpointing} {save_total_limit} {optim} {save_only_model} {logging_steps} {dataset} --add_end_of_sentence_token {add_end_of_sentence_token} --disable_tqdm 1 --force_train_on_trimmed_embeddings {force_train_on_trimmed_embeddings}"

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
            job_desc=f"{job_description_prefix}{output_dir} #rnd #multimodality #tarasov @mrsndmn",
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
    }

    hcg_experiments = []

    extra_params_from_scratch = [
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
        exp_config["output_dir"] = f"{experiment_prefix_base_name}_w_{float(weight):.3f}_l_{suffix}"

        exp_config['model_type'] = 'pretrained'
        exp_config['llama_checkpoint'] = 'HuggingFaceTB/SmolLM2-1.7B'

        hcg_experiments.append(exp_config)


    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return


def run_hcg_llama31_8B_hcg(
        hcg_loss_weight=1.0,
        fan_in_idx=None,
        num_train_epochs=1,
        add_end_of_sentence_token=0,
        fan_out_idx=None,
        optimized_params='fan_in,fan_out',
        fan_out_projection_mlp_intermediate_size=None,
        hcg_fan_in_from=None,
        learning_rate=0.08,
        hcg_learning_rate=0.04,
        model_type="pretrained",
        llama_checkpoint="unsloth/Meta-Llama-3.1-8B",
        fan_out_projection=False,
        select_train_dataset_items=500000,
        lr_scheduler_type="constant_with_warmup",
        instance_type="a100.1gpu",
        experiment_prefix_base_name = "adaptive_hcg_llama31_8B",
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        init_hcg_a=1.0,
        save_steps=5000,
        save_total_limit=3,
        per_device_train_batch_size=4,
        optim="adamw_torch_fused",
        torch_compile='1',
        logging_steps='',
        warmup_steps=2000,
        dataset=None,
        force_train_on_trimmed_embeddings='0',
        **kwargs,
    ):

    # assert fan_in_idx is not None, "fan_in_idx is required"
    # assert fan_out_idx is not None, "fan_out_idx is required"

    common_params = {
        # Model
        "model_type": model_type,
        "llama_checkpoint": llama_checkpoint,
        "hcg_fan_in_from": hcg_fan_in_from,

        "force_train_on_trimmed_embeddings": force_train_on_trimmed_embeddings,
        "optimized_params": optimized_params,

        "dataset": dataset,
        "num_train_epochs": num_train_epochs,

        "init_hcg_a": init_hcg_a,
        "gradient_checkpointing": gradient_checkpointing,
        "warmup_steps": warmup_steps,
        "add_end_of_sentence_token": add_end_of_sentence_token,

        "fan_out_projection_mlp_intermediate_size": fan_out_projection_mlp_intermediate_size,

        # Data
        "select_train_dataset_items": select_train_dataset_items,
        "per_device_train_batch_size": per_device_train_batch_size,

        "logging_steps": logging_steps,

        "optim": optim,
        "torch_compile": torch_compile,

        # Training
        "learning_rate": learning_rate,
        "hcg_learning_rate": hcg_learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,

        # "hcg_loss_weight": # will be overriden in cycle later,
        "max_grad_norm": 0,

        "gradient_accumulation_steps": gradient_accumulation_steps,

        'instance_type': instance_type,

        # Training type
        "pretrain_fan_out_projection": "0",
        "fan_out_projection": "1" if fan_out_projection else "0",
    }

    hcg_experiments = []

    extra_params_from_scratch = [
        (
            f"{fan_in_idx}-{fan_out_idx}",
            f"{fan_in_idx}",
            f"{fan_out_idx}",
        ),
    ]

    # for hcg_loss_weight in [ 0.1, 1.0 ]:
    for suffix, _fan_in_idx, _fan_out_idx in extra_params_from_scratch:
        in_layers_str = "1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1"
        exp_config = {
            "dummy_adaptive_fan_in_layers_str": in_layers_str,
            "fan_in_idx": _fan_in_idx,
            "fan_out_idx": _fan_out_idx,
            "hcg_loss_weight": hcg_loss_weight,
            **common_params,
        }
        exp_config["output_dir"] = f"{experiment_prefix_base_name}_w_{float(hcg_loss_weight):.3f}_l_{suffix}"

        hcg_experiments.append(exp_config)


    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return


def run_hcg_llama31_8B_hcg_forward_residuals(**kwargs):

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
        "learning_rate": 0.04,
        "hcg_learning_rate": 0.04,
        "lr_scheduler_type": "constant_with_warmup",

        # "hcg_loss_weight": # will be overriden in cycle later,
        "max_grad_norm": 0,

        'instance_type': 'a100.2gpu',
        'num_train_epochs': 1,

        # Training type
        "pretrain_fan_out_projection": "0",
    }

    hcg_experiments = []

    extra_params_from_scratch = [
        # (
        #     "16_26_forward_residuals",
        #     "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0", # crutch
        #     16, # fan_in_idx
        #     26, # fan_out_idx
        #     '1.0', # hcg_loss_weight
        #     True, # forward_residuals
        # ),
        # (
        #     "16_26_no_forward_residuals",
        #     "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0", # crutch
        #     16, # fan_in_idx
        #     26, # fan_out_idx
        #     '1.0', # hcg_loss_weight
        #     False, # forward_residuals
        # ),
        (
            "8_15_forward_residuals",
            "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0", # crutch
            8, # fan_in_idx
            15, # fan_out_idx
            '0.1', # hcg_loss_weight
            True, # forward_residuals
        ),
        (
            "8_15_no_forward_residuals",
            "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0", # crutch
            8, # fan_in_idx
            15, # fan_out_idx
            '0.1', # hcg_loss_weight
            False, # forward_residuals
        ),
    ]

    for suffix, in_layers_str, fan_in_idx, fan_out_idx, hcg_loss_weight, forward_residuals in extra_params_from_scratch:
        exp_config = {
            "dummy_adaptive_fan_in_layers_str": in_layers_str,
            "fan_in_idx": fan_in_idx,
            "fan_out_idx": fan_out_idx,
            "forward_residuals": forward_residuals,
            "hcg_loss_weight": hcg_loss_weight,
            **common_params,
        }
        exp_config["output_dir"] = f"{experiment_prefix_base_name}_w_{float(hcg_loss_weight):.3f}_l_{suffix}"

        hcg_experiments.append(exp_config)


    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return

def run_hcg_llama31_8B_hcg_each_layer(**kwargs):

    experiment_prefix_base_name = "adaptive_hcg_llama31_8B_each_layer_pruning"

    common_params = {
        # Model
        "model_type": "pretrained",
        "llama_checkpoint": "unsloth/Meta-Llama-3.1-8B",
        "dummy_adaptive_fan_in_layers_str": "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0",

        "freeze_lm_backbone": "1",
        "init_hcg_a": 2.85,

        # Data
        "select_train_dataset_items": 1000000,
        "per_device_train_batch_size": 4,

        # Training
        "learning_rate": 0.04,
        "hcg_learning_rate": 0.04,
        "lr_scheduler_type": "constant_with_warmup",

        "max_grad_norm": 0,

        'instance_type': 'a100.1gpu',
        'num_train_epochs': 1,

        # Training type
        "pretrain_fan_out_projection": "0",
        "fan_out_projection": "0",
        "each_layer_pruning": 1,
    }

    hcg_experiments = []

    for hcg_loss_weight in [ '0.1' ]:
    # for hcg_loss_weight in [ '1.0' ]:
        exp_config = {
            "hcg_loss_weight": hcg_loss_weight,
            **common_params,
        }
        exp_config["output_dir"] = f"{experiment_prefix_base_name}_w_{float(hcg_loss_weight):.3f}"

        hcg_experiments.append(exp_config)


    run_experiments(hcg_experiments, job_description_prefix="HCG: ", **kwargs)

    return

def run_hcg_qwen25_7B_hcg(
    hcg_loss_weight=1.0,
    fan_in_idx=None,
    fan_out_idx=None,
    optimized_params='fan_in,fan_out',
    learning_rate=0.08,
    model_type="pretrained",
    hcg_fan_in_from=None,
    instance_type="a100.1gpu",
    llama_checkpoint="Qwen/Qwen2.5-7B",
    fan_out_projection=False,
    select_train_dataset_items=500000,
    experiment_prefix_base_name="adaptive_hcg_qwen25_7B",
    fan_out_projection_mlp_intermediate_size=18944,
    gradient_accumulation_steps=4,
    lr_scheduler_type="constant_with_warmup",
    init_hcg_a=1.0,
    **kwargs,
):

    assert fan_in_idx is not None, "fan_in_idx is required"
    assert fan_out_idx is not None, "fan_out_idx is required"

    common_params = {
        # Model
        "model_type": model_type,
        "llama_checkpoint": llama_checkpoint,

        "optimized_params": optimized_params,

        "init_hcg_a": init_hcg_a,
        "hcg_fan_in_from": hcg_fan_in_from,

        "instance_type": instance_type,

        # Data
        "select_train_dataset_items": select_train_dataset_items,
        "per_device_train_batch_size": 4,

        "fan_out_projection_mlp_intermediate_size": fan_out_projection_mlp_intermediate_size,

        # Training
        "learning_rate": learning_rate,
        "hcg_learning_rate": 0.08,
        "lr_scheduler_type": lr_scheduler_type,

        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_steps": 5000,

        "max_grad_norm": 0,

        'num_train_epochs': 1,

        # Training type
        "pretrain_fan_out_projection": "0",
        "fan_out_projection": "1" if fan_out_projection else "0",
    }

    hcg_experiments = []

    extra_params_from_scratch = [
        (
            f"{fan_in_idx}-{fan_out_idx}",
            f"{fan_in_idx}",
            f"{fan_out_idx}",
        ),
    ]

    for suffix, fan_in_idx, fan_out_idx in extra_params_from_scratch:
        in_layers_str = "1,1,1,1,1,1,1,1,1,1,1,0,1,1"
        exp_config = {
            "dummy_adaptive_fan_in_layers_str": in_layers_str,
            "fan_in_idx": fan_in_idx,
            "fan_out_idx": fan_out_idx,
            **common_params,
        }
        exp_config['hcg_loss_weight'] = hcg_loss_weight
        exp_config["output_dir"] = f"{experiment_prefix_base_name}_w_{hcg_loss_weight:.3f}_l_{suffix}"

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

    # if not dry:
    #     tests_run = subprocess.run(["pytest", "src/transformers/models/llama/tests/"])
    #     if tests_run.returncode != 0:
    #         print("Tests failed")
    #         exit(1)


    # Pretrain Fain Out on Tiny Datasets
    # NGPUS = 1
    NGPUS = 1
    num_train_epochs = 1
    per_device_train_batch_size = 32
    gradient_accumulation_steps = 1
    save_steps = 5000

    # add_end_of_sentence_token
    # if True:
    if False:
        run_hcg_llama31_8B_hcg(
            hcg_loss_weight=0.1,
            hcg_learning_rate=0.01,
            learning_rate=0.0003,
            model_type='pretrained_checkpoint',
            optimized_params='full',
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # select_train_dataset_items=1510000 * NGPUS,
            num_train_epochs=num_train_epochs,
            save_total_limit=13,
            save_steps=save_steps,
            instance_type=f'a100.{NGPUS}gpu',
            llama_checkpoint=f'{workdir_prefix}/paper_checkpoints/pretrain/adaptive_slm2_135M_random_init',
            dataset='tiny',
            select_train_dataset_items=0,
            fan_in_idx=10,
            fan_out_idx=20,
            warmup_steps=1000,
            dry=dry,
            lr_scheduler_type='cosine',
            experiment_prefix_base_name="adaptive_slm2_135M_pretrain_with_end_of_sentence_token",
            add_end_of_sentence_token=1,
        )

    # if True:
    if False:
        run_hcg_llama31_8B_hcg(
            hcg_loss_weight=0.1,
            hcg_learning_rate=0.01,
            learning_rate=0.0003,
            model_type='pretrained_checkpoint',
            optimized_params='full',
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # select_train_dataset_items=1510000 * NGPUS,
            num_train_epochs=num_train_epochs,
            save_total_limit=13,
            save_steps=save_steps,
            instance_type=f'a100.{NGPUS}gpu',
            llama_checkpoint=f'{workdir_prefix}/paper_checkpoints/pretrain/adaptive_slm2_135M_random_init',
            dataset='tiny',
            select_train_dataset_items=0,
            fan_in_idx=10,
            fan_out_idx=20,
            warmup_steps=1000,
            dry=dry,
            lr_scheduler_type='cosine',
            experiment_prefix_base_name="adaptive_slm2_135M_pretrain",
        )

    # if True:
    if False:
        run_hcg_llama31_8B_hcg(
            hcg_loss_weight=0.0,
            hcg_learning_rate=0.00,
            learning_rate=0.0003,
            model_type='SmolLM2-135M',
            optimized_params='full',
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # select_train_dataset_items=1510000 * NGPUS,
            num_train_epochs=num_train_epochs,
            save_total_limit=13,
            save_steps=save_steps,
            instance_type=f'a100.{NGPUS}gpu',
            llama_checkpoint=f"{workdir_prefix}/paper_checkpoints/pretrain/slm2_135M_random_init/",
            dataset='tiny',
            select_train_dataset_items=0,
            fan_in_idx='',
            fan_out_idx='',
            dry=dry,
            warmup_steps=1000,
            lr_scheduler_type='cosine',
            experiment_prefix_base_name="vanilla_slm2_135M_pretrain",
        )

    if True:
    # if False:
        run_hcg_llama31_8B_hcg(
            hcg_loss_weight=0.0,
            hcg_learning_rate=0.00,
            learning_rate=0.0003,
            model_type='SmolLM2-135M',
            optimized_params='full',
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # select_train_dataset_items=1510000 * NGPUS,
            num_train_epochs=num_train_epochs,
            save_total_limit=13,
            save_steps=save_steps,
            instance_type=f'a100.{NGPUS}gpu',
            llama_checkpoint=f"{workdir_prefix}/paper_checkpoints/pretrain/slm2_127M_random_init_20L/",
            dataset='tiny',
            select_train_dataset_items=0,
            fan_in_idx='',
            fan_out_idx='',
            dry=dry,
            warmup_steps=1000,
            lr_scheduler_type='cosine',
            experiment_prefix_base_name="vanilla_slm2_127M_20L_pretrain",
        )


    if False:
        for i in range(2, 30):
            run_hcg_llama31_8B_hcg(
                hcg_loss_weight=0.1,
                fan_in_idx=i,
                fan_out_idx=i+1,
                dry=dry,
                select_train_dataset_items=85000,
                experiment_prefix_base_name="adaptive_hcg_llama31_8B_one"
            )

        sys.exit(0)

    # Pretrain Learned Vocab
    if False:
    # if True:
        for hcg_loss_weight in [ 0.1, 1.0 ]:
            run_hcg_llama31_8B_hcg(
                fan_in_idx=18,
                fan_out_idx=26,
                hcg_loss_weight=hcg_loss_weight,
                model_type="pretrained",
                llama_checkpoint='unsloth/Meta-Llama-3.1-8B',
                fan_out_projection_mlp_intermediate_size=16384,
                fan_out_projection=False,
                experiment_prefix_base_name='adaptive_hcg_llama31_8B_learned_vocab',
                select_train_dataset_items=85000,
                dry=dry,
            )

            run_hcg_qwen25_7B_hcg(
                fan_in_idx=9,
                fan_out_idx=20,
                hcg_loss_weight=hcg_loss_weight,
                model_type="pretrained",
                llama_checkpoint='Qwen/Qwen2.5-7B',
                fan_out_projection=False,
                select_train_dataset_items=85000,
                fan_out_projection_mlp_intermediate_size=18944,
                experiment_prefix_base_name='adaptive_hcg_qwen25_7B_learned_vocab',
                dry=dry,
            )

        sys.exit(0)


    llama_calibrated_checkpoints = [
        # (18, 26, './paper_checkpoints/base_thshld_0.6/adaptive_hcg_llama31_8B_learned_vocab_w_1.000_l_18-26_M0K275CH/checkpoint-5306_calib40'),
        (18, 26, './paper_checkpoints/base_thshld_0.6/adaptive_hcg_llama31_8B_learned_vocab_w_1.000_l_18-26_M0K275CH/checkpoint-5306_calib90'),
    ]

    qwen_calibrated_checkpoints = [
        (9, 20, './paper_checkpoints/base_thshld_0.6/adaptive_hcg_qwen25_7B_learned_vocab_w_1.000_l_9-20_J3BTODV3/checkpoint-5306_calib40'),
        # (9, 20, './paper_checkpoints/base_thshld_0.6/adaptive_hcg_qwen25_7B_learned_vocab_w_1.000_l_9-20_J3BTODV3/checkpoint-5306_calib90'),
    ]

    # Fan out projection
    # if True:
    if False:

        gradient_accumulation_steps = 4
        n_gpus = 8
        batch_size = 4
        optim_steps = 11000
        save_total_limit = 4

        for fan_in_idx, fan_out_idx, fan_in_from in llama_calibrated_checkpoints:
            run_hcg_llama31_8B_hcg(
                fan_in_idx=fan_in_idx,
                fan_out_idx=fan_out_idx,
                learning_rate=0.0003,
                optimized_params="fan_out",
                model_type="pretrained",
                fan_out_projection_mlp_intermediate_size=16384,
                hcg_fan_in_from=fan_in_from,
                llama_checkpoint='unsloth/Meta-Llama-3.1-8B',
                fan_out_projection=True,
                experiment_prefix_base_name='adaptive_hcg_llama31_8B_fan_out_projection_trimmed_embeddings',
                select_train_dataset_items=optim_steps*batch_size*n_gpus*gradient_accumulation_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                per_device_train_batch_size=batch_size,
                force_train_on_trimmed_embeddings='1',
                torch_compile='0',
                lr_scheduler_type="cosine",
                instance_type=f"a100.{n_gpus}gpu",
                save_steps=int(optim_steps // save_total_limit),
                save_total_limit=save_total_limit,
                logging_steps=1,
                init_hcg_a='',
                dry=dry,
                warmup_steps=1000,
            )

    if False:
    # if True:
        for fan_in_idx, fan_out_idx, fan_in_from in qwen_calibrated_checkpoints:
            run_hcg_qwen25_7B_hcg(
                fan_in_idx=fan_in_idx,
                fan_out_idx=fan_out_idx,
                learning_rate=0.0003,
                model_type="pretrained",
                hcg_fan_in_from=fan_in_from,
                llama_checkpoint='Qwen/Qwen2.5-7B',
                fan_out_projection=True,
                experiment_prefix_base_name='adaptive_hcg_qwen25_7B_fan_out_projection',
                select_train_dataset_items=250000,
                gradient_accumulation_steps=1,
                lr_scheduler_type="cosine",
                init_hcg_a='',
                dry=dry,
            )


    # Finetune LLM
    if False:
    # if True:

        gradient_accumulation_steps = 64
        n_gpus = 4
        batch_size = 2
        optim_steps = 11000

        for fan_in_idx, fan_out_idx, llama_checkpoint in llama_calibrated_checkpoints:
            run_hcg_llama31_8B_hcg(
                fan_in_idx=fan_in_idx,
                fan_out_idx=fan_out_idx,
                learning_rate=0.0003,
                optimized_params="inner_layers",
                model_type="pretrained",
                hcg_fan_in_from=llama_checkpoint,
                init_hcg_a='',
                llama_checkpoint='unsloth/Meta-Llama-3.1-8B',
                fan_out_projection=False,
                select_train_dataset_items=optim_steps*batch_size*n_gpus*gradient_accumulation_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                per_device_train_batch_size=batch_size,
                instance_type=f"a100.{n_gpus}gpu",
                lr_scheduler_type="cosine",
                experiment_prefix_base_name='adaptive_hcg_llama31_8B_finetune',
                save_steps=2500,
                dry=dry,
                fan_out_projection_mlp_intermediate_size=16384,
                warmup_steps=1000,
            )

    if False:
    # if True:
        for fan_in_idx, fan_out_idx, qwen_checkpoint in qwen_calibrated_checkpoints:
            run_hcg_qwen25_7B_hcg(
                fan_in_idx=fan_in_idx,
                fan_out_idx=fan_out_idx,
                learning_rate=0.00001,
                unfreeze_inner_layers='1',
                model_type="pretrained",
                hcg_fan_in_from=qwen_checkpoint,
                init_hcg_a='',
                llama_checkpoint='Qwen/Qwen2.5-7B',
                fan_out_projection=False,
                select_train_dataset_items=50000*4*4,
                lr_scheduler_type="cosine",
                experiment_prefix_base_name='adaptive_hcg_qwen25_7B_finetune',
                dry=dry,
                instance_type="a100.4gpu",
            )

        # sys.exit(0)

    # Finetune Full LLM
    # if True:
    if False:
        gradient_accumulation_steps = 64
        n_gpus = 4
        batch_size = 2
        optim_steps = 11000

        for fan_in_idx, fan_out_idx, llama_checkpoint in llama_calibrated_checkpoints:
            run_hcg_llama31_8B_hcg(
                fan_in_idx=fan_in_idx,
                fan_out_idx=fan_out_idx,
                learning_rate=0.0003,
                optimized_params='full',
                instance_type=f"a100.{n_gpus}gpu_fsdp",
                model_type="pretrained",
                hcg_fan_in_from=llama_checkpoint,
                init_hcg_a='',
                llama_checkpoint='unsloth/Meta-Llama-3.1-8B',
                fan_out_projection=False,
                optim="adamw_torch_fused",
                gradient_checkpointing='0',
                torch_compile='0',
                gradient_accumulation_steps=gradient_accumulation_steps,
                per_device_train_batch_size=batch_size,
                select_train_dataset_items=optim_steps*batch_size*n_gpus*gradient_accumulation_steps,
                lr_scheduler_type="cosine",
                experiment_prefix_base_name='adaptive_hcg_llama31_8B_full_finetune',
                dry=dry,
                fan_out_projection_mlp_intermediate_size=16384,
                # save_steps=1,
                save_steps=2500,
                save_total_limit=10,
                logging_steps=10,
                warmup_steps=1000,
            )


    # Finetune Lora LLM
    # if True:
    # if False:
        gradient_accumulation_steps = 64
        n_gpus = 4
        batch_size = 2
        optim_steps = 11000

        for fan_in_idx, fan_out_idx, llama_checkpoint in llama_calibrated_checkpoints:
            run_hcg_llama31_8B_hcg(
                fan_in_idx=fan_in_idx,
                fan_out_idx=fan_out_idx,
                learning_rate=0.0005,
                optimized_params='lora_lm_head_embed_tokens',
                instance_type=f"a100.{n_gpus}gpu",
                model_type="pretrained",
                hcg_fan_in_from=llama_checkpoint,
                init_hcg_a='',
                llama_checkpoint='unsloth/Meta-Llama-3.1-8B',
                fan_out_projection=False,
                optim="adamw_torch_fused",
                gradient_checkpointing='0',
                torch_compile='0',
                gradient_accumulation_steps=gradient_accumulation_steps,
                per_device_train_batch_size=batch_size,
                select_train_dataset_items=optim_steps*batch_size*n_gpus*gradient_accumulation_steps,
                lr_scheduler_type="cosine",
                experiment_prefix_base_name='adaptive_hcg_llama31_8B_lora_finetune',
                dry=dry,
                fan_out_projection_mlp_intermediate_size=16384,
                # save_steps=1,
                save_steps=2500,
                save_total_limit=10,
                warmup_steps=1000,
            )


    # run_hcg_llama31_8B_hcg(dry=dry)
    # run_hcg_qwen25_7B_hcg(dry=dry)

