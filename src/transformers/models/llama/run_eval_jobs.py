import glob
import time
import client_lib # импортируем библиотеку для работы с ML Space
import json
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

        script_str = f'bash -c \'date && cd {workdir_prefix} && {env_bin_path}/python {env_bin_path}/lighteval accelerate --override-batch-size 32 --output-dir {output_dir} --custom-tasks /workspace-SR004.nfs2/d.tarasov/cosmopedia/evaluation/lighteval_tasks.py "pretrained={pretrained_model},dtype=bfloat16,device=cuda" "custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|arc|0|1,custom|piqa|0|1,custom|wikitext_103|0|1"\''

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
            job_desc=f"{job_description_prefix} {pretrained_model} #rnd #multimodality",
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

def eval_hcg_no_self_attention(**kwargs):

    checkpoints = [
        "adaptive_hcg_slm2_360M_w_0.010_l_12_no_self_attn_G8Z0KI2B/checkpoint-240000",
    ]

    hcg_experiments = [ { "pretrained_model": x } for x in checkpoints ]

    if kwargs.pop('extract_metrics', False):
        run_extract_metrics(checkpoints)
    else:
        run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG: ", **kwargs)


    return

def saturday_morninig_eval(**kwargs):

    checkpoints = [
        # Strange 8 layer
        # "./run_hcg_smollm1dot7B_layer_8_8_lmv_1.25/_backup_checkpoint-35000",
        # "./run_hcg_smollm1dot7B_layer_8_8_lmv_1.1/_backup_checkpoint-35000",

        # Strange 8 layer
        # Fixed percent pruning slm2 1.7b 2 layer
        # TODO
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_2_pr_pct20/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_2_pr_pct30/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_2_pr_pct40/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_2_pr_pct60/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_2_pr_pct80/checkpoint-4993",

        # Fixed percent pruning slm2 1.7b 4 layer
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_4_pr_pct20/checkpoint-4993",
        # Посчиталось
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_4_pr_pct30/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_4_pr_pct40/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_4_pr_pct60/checkpoint-4993",
        # "./adaptive_hcg_slm2_1.7B_fixed_pruning_percent_v2_4_pr_pct80/checkpoint-4993",

        # Fan out ablations
        # TODO

        # Random 20% tokens
        # "./random_hcg_4_random0.2_1.7B/checkpoint-24993",
        # "./random_hcg_8_random0.2_1.7B/checkpoint-24993",

        "./adaptive_hcg_slm2_1.7B_nofoutproj_4/checkpoint-18743",
        "./adaptive_hcg_slm2_1.7B_nofoutproj_8/checkpoint-18743",
    ]

    hcg_experiments = [ { "pretrained_model": x } for x in checkpoints ]

    run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG: ", **kwargs)

    return


def no_crutch_loss_eval(**kwargs):

    checkpoints = [
        "./adaptive_hcg_slm2_1.7B_hcg_lambda_iterate_3/checkpoint-351000",
        "./adaptive_hcg_slm2_1.7B_hcg_lambda_iterate_3/_backup_checkpoint-37000",
        "./adaptive_hcg_slm2_1.7B_hcg_lambda_iterate_3/_backup_checkpoint-52000",
        "./adaptive_hcg_slm2_1.7B_hcg_lambda_iterate_3/_backup_checkpoint-61000",
        "./adaptive_hcg_slm2_1.7B_hcg_lambda_iterate_3/_backup_checkpoint-77000",

        # 2.75
        "./adaptive_hcg_slm2_1.7B_hcg_lambda_iterate_2.75/checkpoint-219000",

        # 2.5
        "./adaptive_hcg_slm2_1.7B_hcg_lambda_iterate_2.5/checkpoint-217000"

        # 2.25
        "./adaptive_hcg_slm2_1.7B_hcg_lambda_iterate_2.25/checkpoint-218000",
    ]

    hcg_experiments = [ { "pretrained_model": x } for x in checkpoints ]

    run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG: ", **kwargs)

    return


def no_crutch_loss_normalize_token_frequenct_eval(**kwargs):

    checkpoints = [
        "./adaptive_hcg_slm2_1.7B_hcg_scale_token_frequency_0.5_no_eossp/checkpoint-58793/",
        "./adaptive_hcg_slm2_1.7B_hcg_scale_token_frequency_0.75_no_eossp/checkpoint-58793/",
        "./adaptive_hcg_slm2_1.7B_hcg_scale_token_frequency_1.0_no_eossp/checkpoint-58793/",
        "./adaptive_hcg_slm2_1.7B_hcg_scale_token_frequency_1.5_no_eossp/checkpoint-58793/",
        "./adaptive_hcg_slm2_1.7B_hcg_scale_token_frequency_2.0_no_eossp/checkpoint-58793/",

        "./adaptive_hcg_slm2_360M_hcg_scale_token_frequency_0.75_no_eossp/checkpoint-58793/",
        "./adaptive_hcg_slm2_360M_hcg_scale_token_frequency_1_no_eossp/checkpoint-58793/",
        "./adaptive_hcg_slm2_360M_hcg_scale_token_frequency_1.5_no_eossp/checkpoint-58793/",
        "./adaptive_hcg_slm2_360M_hcg_scale_token_frequency_2.0_no_eossp/checkpoint-58793/",
        "./adaptive_hcg_slm2_360M_hcg_scale_token_frequency_2.5_no_eossp/checkpoint-58793/",
        "./adaptive_hcg_slm2_360M_hcg_scale_token_frequency_3.0_no_eossp/checkpoint-58793/",
        "./adaptive_hcg_slm2_360M_hcg_scale_token_frequency_3.5_no_eossp/checkpoint-58793/",
    ]

    hcg_experiments = [ { "pretrained_model": x } for x in checkpoints ]


    if kwargs.pop('extract_metrics', False):
        run_extract_metrics(checkpoints)
    else:
        run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG: ", **kwargs)

    return


def no_crutch_loss_normalize_token_frequenct_eval_bs_1m(**kwargs):

    checkpoints = [
        "./adaptive_hcg_slm2_360M_hcg_scale_token_frequency_es_fix_pretrain_1.5_no_eossp/checkpoint-14698",
        "./adaptive_hcg_slm2_360M_hcg_scale_token_frequency_es_fix_pretrain_1.25_no_eossp/checkpoint-14698",
    ]

    hcg_experiments = [ { "pretrained_model": x } for x in checkpoints ]

    if kwargs.pop('extract_metrics', False):
        run_extract_metrics(checkpoints)
    else:
        run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG: ", **kwargs)

    return


def baselines_rule_based_eval(**kwargs):

    checkpoints = [
        # Freezed
        "./adaptive_hcg_slm2_360M_rule_based_random_0.1_freeze_1_WXBG8TRK/checkpoint-37487/",
        "./adaptive_hcg_slm2_360M_rule_based_uniform_0.1_freeze_1_W87CVQV5/checkpoint-37487/",
        "./adaptive_hcg_slm2_360M_rule_based_stop_words_pruning_freeze_1_MQRNRNRJ/checkpoint-37487/",

        "./adaptive_hcg_slm2_360M_rule_based_uniform_0.2_freeze_1_Y4Q7B7JW/checkpoint-37487/",
        "./adaptive_hcg_slm2_360M_rule_based_random_0.2_freeze_1_3YS0AJM1/checkpoint-37487/",

        # Finetuned
        "./adaptive_hcg_slm2_360M_rule_based_random_0.1_freeze_0_5X68TMDE/checkpoint-37487/",
        "./adaptive_hcg_slm2_360M_rule_based_uniform_0.1_freeze_0_M04D44LF/checkpoint-37487/",
        "./adaptive_hcg_slm2_360M_rule_based_stop_words_pruning_freeze_0_JEXQ1YJA/checkpoint-37487/",

        "./adaptive_hcg_slm2_360M_rule_based_random_0.2_freeze_0_2GTV5AUE/checkpoint-37487/",
        "./adaptive_hcg_slm2_360M_rule_based_uniform_0.2_freeze_0_Z3182L60/checkpoint-37487/",
    ]

    hcg_experiments = [ { "pretrained_model": x } for x in checkpoints ]

    if kwargs.pop('extract_metrics', False):
        run_extract_metrics(checkpoints)
    else:
        run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG Rule Based: ", **kwargs)

    return



def run_extract_metrics(checkpoints: list[str]):

    bench_keys = [
        'custom|arc:_average|0',
        'custom|piqa|0',
        # 'custom|trivia_qa|0',
        'custom|mmlu_cloze:_average|0',
        'custom|mmlu_pro_cloze|0',
        # 'custom|gsm8k|5',
        'custom|wikitext_103|0',
    ]

    print(" & ".join([ 'checkpoint', 'pruned', 'MaxLM Loss' ] + bench_keys), " \\\\")

    for checkpoint in checkpoints:
        checkpoint_norm = checkpoint.replace('/', '_')
        metrics_mask = os.path.join('exps_evaluation', 'results', checkpoint_norm, '*.json')
        metrics_paths = glob.glob(metrics_mask)
        assert len(metrics_paths) >= 1, f"No metrics files found for {checkpoint}"
        # Sort by creation time and take the most recent one
        metrics_path = sorted(metrics_paths, key=lambda x: os.path.getctime(x))[-1]

        with open(metrics_path, "r") as f:
            json_data = json.load(f)

        max_len = max(map(len, bench_keys))

        checkpoint_metrics = []
        for key in bench_keys:

            metric_dict = json_data['results'].get(key, {})

            metric = 0
            metric_stderr = 0

            if 'acc_norm' in metric_dict:
                metric = metric_dict['acc_norm']
                metric_stderr = metric_dict['acc_norm_stderr']
            elif 'qem' in metric_dict:
                metric = metric_dict['qem']
                metric_stderr = metric_dict['qem_stderr']
            elif 'ppl' in metric_dict:
                metric = metric_dict['ppl']
                metric_stderr = metric_dict['ppl_stderr']
            elif len(metric_dict.keys()) > 0:
                raise ValueError("unknown metrics:", metric_dict)

            space = " " * (max_len - len(key) + 1)
            # print(key, space, "\t", f"{metric*100:.2f}", '\tstderr', f"{metric_stderr*100:.2f}")

            checkpoint_metrics.append(f"{metric*100:.2f}")

        print(" & ".join([ checkpoint, '\%', '\-'] + checkpoint_metrics), " \\\\")


if __name__ == "__main__":

    import sys

    dry = len(sys.argv) > 1 and sys.argv[1] == 'dry'
    extract_metrics = len(sys.argv) > 1 and sys.argv[1] == 'extract_metrics'

    print("dry", dry)

    # eval_hcg_adaptive_pretrain(dry=dry)
    # eval_hcg_fixed_percent(dry=dry)

    # eval_hcg_strange_8layer(dry=dry)
    eval_hcg_no_self_attention(dry=dry, extract_metrics=extract_metrics)

    # no_crutch_loss_eval(dry=dry)
    # no_crutch_loss_normalize_token_frequenct_eval(dry=dry, extract_metrics=extract_metrics)
    # no_crutch_loss_normalize_token_frequenct_eval_bs_1m(dry=dry, extract_metrics=extract_metrics)
    # baselines_rule_based_eval(dry=dry, extract_metrics=extract_metrics)