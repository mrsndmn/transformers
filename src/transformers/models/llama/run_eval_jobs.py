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

console = Console()
console.print(client_lib.get_instance_types(regions="SR004"))

INSTANCE_TYPE = "a100.1gpu"
N_WORKERS = 1
BASE_IMAGE = "cr.ai.cloud.ru/f51af5b1-d43b-4db4-938d-569d7cfffb7a/cuda12.1-torch2-py310-adaptive_attention:0.0.3"

workdir_prefix = "/workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out"

def run_eval_experiments(experiments, job_description_prefix="eval", dry=False, tasks=None):

    experiments = copy.deepcopy(experiments)

    env_bin_path = "/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin"

    print("len experiments:", len(experiments))

    for exp in experiments:

        pretrained_model = exp.pop('pretrained_model')
        output_dir = exp.pop('output_dir', './exps_evaluation')

        if len(exp.keys()) > 0:
            raise ValueError("Invalid exp values!")

        default_tasks = "custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|arc|0|1,custom|wikitext_103|0|1,custom|winogrande|0|1,custom|hellaswag|0|1,custom|siqa|0|1,custom|openbookqa|0|1,custom|piqa|0|1"

        if tasks is None:
            tasks = default_tasks

        script_str = f'bash -c \'date && cd {workdir_prefix} && {env_bin_path}/python {env_bin_path}/lighteval accelerate --output-dir {output_dir} --custom-tasks /workspace-SR004.nfs2/d.tarasov/cosmopedia/evaluation/lighteval_tasks.py "pretrained={pretrained_model},dtype=bfloat16,device=cuda" "{tasks}"\''
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
            # conda_env="test_client_lib",
            processes_per_worker=1,
            job_desc=f"{job_description_prefix} {pretrained_model} {tasks} #rnd #multimodality",
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

def run_extract_metrics(checkpoints: list[str], tasks=None):

    if tasks is None:
        tasks = [
            'custom|arc:_average|0',
            'custom|piqa|0',
            'custom|mmlu_cloze:_average|0',
            'custom|mmlu_pro_cloze|0',
            'custom|wikitext_103|0',
        ]
    else:
        tasks_mapping = {
            'custom|arc|0|1': 'custom|arc:_average|0',
            'custom|openbookqa|0|1': 'custom|openbookqa|0',
            'custom|mmlu_cloze|0|1': 'custom|mmlu_cloze:_average|0',
            'custom|mmlu_pro_cloze|0|1': 'custom|mmlu_pro_cloze|0',
            'custom|wikitext_103|0|1': 'custom|wikitext_103|0',
            'custom|siqa|0|1': 'custom|siqa|0',
            'custom|piqa|0|1': 'custom|piqa|0',
            'custom|hellaswag|0|1': 'custom|hellaswag|0',
            'custom|winogrande|0|1': 'custom|winogrande|0',
        }
        tasks = list(map(lambda x: tasks_mapping[x], tasks))

    print(" & ".join([ 'checkpoint', 'pruned', 'MaxLM Loss' ] + tasks), " \\\\")

    for checkpoint in checkpoints:

        if 'unsloth' in checkpoint or 'Qwen' in checkpoint:
            # exps_evaluation/results/unsloth/Meta-Llama-3.1-8B/
            checkpoint_norm = checkpoint.split('/')
        else:
            checkpoint_norm = [ checkpoint.replace('/', '_') ]
        metrics_mask = os.path.join('exps_evaluation', 'results', *checkpoint_norm, '*.json')
        metrics_paths = glob.glob(metrics_mask)
        # print("metrics_paths", metrics_paths)
        assert len(metrics_paths) >= 1, f"No metrics files found for {checkpoint}"
        # Sort by creation time and take the most recent one
        metrics_paths = sorted(metrics_paths, key=lambda x: os.path.getctime(x))[-9:]
        # print("")
        # print("checkpoint", checkpoint)
        # print("metrics_paths", metrics_paths)

        checkpoint_metrics = []

        metrics_dict = {}

        for metrics_path in metrics_paths:

            with open(metrics_path, "r") as f:
                json_data = json.load(f)

            for key in tasks:

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
                    metric = metric_dict['ppl'] / 100
                    metric_stderr = metric_dict['ppl_stderr']
                elif len(metric_dict.keys()) > 0:
                    raise ValueError("unknown metrics:", metric_dict)

                if metric != 0:
                    metrics_dict[key] = f"{metric*100:.2f}"

        for key in tasks:
            checkpoint_metrics.append(metrics_dict.get(key, ""))

        print(" & ".join([ checkpoint, '\%', '\-'] + checkpoint_metrics), " \\\\")



def eval_hcg_adaptive_pretrain_all_tasks_parallel(**kwargs):

    checkpoints = [
        # a lot of jobs ...
        # /mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/exps_evaluation/results/unsloth/Meta-Llama-3.1-8B/
        # results_2025-06-16T17-17-36.190191.json
        # results_2025-06-16T17-20-07.497466.json
        # results_2025-06-16T17-20-20.107840.json
        # results_2025-06-16T17-20-33.343409.json
        # results_2025-06-16T17-20-49.794687.json
        # results_2025-06-16T17-21-40.487502.json
        # results_2025-06-16T17-21-50.828092.json
        # results_2025-06-16T17-26-02.538177.json
        # results_2025-06-16T17-43-04.025151.json
        #
        # "unsloth/Meta-Llama-3.1-8B",
        "Qwen/Qwen2.5-7B",

        # "./adaptive_hcg_llama31_8B_w_1.000_l_22-26_NTWKRP0G/checkpoint-5000",
        # "./adaptive_hcg_llama31_8B_w_1.000_l_22-26_NTWKRP0G/calibr60_checkpoint-5000_left_padding/",
        # "./adaptive_hcg_llama31_8B_l22-26_analytical_pruning",

        # "./adaptive_hcg_qwen25_7B_w_1.000_l_12-17_HXIGMJTI/checkpoint-5000",
        # "./adaptive_hcg_qwen25_7B_w_1.000_l_12-17_HXIGMJTI/calib40_checkpoint-5000",
        # "./adaptive_hcg_qwen25_7B_l12-17_analytical_pruning",

        # TODO fan out projection + llm layers finetuned
        # TODO fan out projection checkpoint

        # Analytical Pruning
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_llama31_8B_l18-26_analytical_pruning_q0.25/",
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_llama31_8B_l18-26_analytical_pruning_q0.5/",
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_llama31_8B_l18-26_analytical_pruning_q0.75/",
        "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_llama31_8B_l18-26_analytical_pruning_q1/",
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_qwen25_7B_l9-20_analytical_pruning_q0.25/",
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_qwen25_7B_l9-20_analytical_pruning_q0.5/",
        # "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_qwen25_7B_l9-20_analytical_pruning_q0.75/",
        "./paper_checkpoints/analytical_thshld_0.6/adaptive_hcg_qwen25_7B_l9-20_analytical_pruning_q1/",
    ]

    checkpoints = [
        "./paper_checkpoints/base_thshld_0.6/adaptive_hcg_llama31_8B_learned_vocab_w_1.000_l_18-26_M0K275CH/checkpoint-5306_calib40",
        "./paper_checkpoints/base_thshld_0.6/adaptive_hcg_qwen25_7B_learned_vocab_w_1.000_l_9-20_J3BTODV3/checkpoint-5306_calib40",

        './paper_checkpoints/base_thshld_0.6/adaptive_hcg_llama31_8B_learned_vocab_w_1.000_l_18-26_M0K275CH/checkpoint-5306_calib90',
        './paper_checkpoints/base_thshld_0.6/adaptive_hcg_qwen25_7B_learned_vocab_w_1.000_l_9-20_J3BTODV3/checkpoint-5306_calib90',
    ]

    # temporary checkpoints for finetuned models
    checkpoints = [
        './adaptive_hcg_llama31_8B_finetune_w_1.000_l_18-26_HLTUKB25/checkpoint-5000',
        './adaptive_hcg_qwen25_7B_finetune_w_1.000_l_9-20_N31CKBUP/checkpoint-5000',
    ]

    # checkpoints for fan out projection tuned models
    checkpoints = [
        './adaptive_hcg_llama31_8B_fan_out_projection_w_1.000_l_18-26_GHLL4UA3/checkpoint-60000/',
        './adaptive_hcg_qwen25_7B_fan_out_projection_w_1.000_l_9-20_8VWXTSOA/checkpoint-60000/',
    ]


    hcg_experiments = [ { "pretrained_model": x } for x in checkpoints ]

    print('len hcg_experiments:', len(hcg_experiments))

    # task = None
    # tasks = [ 'custom|wikitext_103|0|1' ]
    tasks = "custom|arc|0|1,custom|openbookqa|0|1,custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|wikitext_103|0|1,custom|siqa|0|1,custom|piqa|0|1,custom|hellaswag|0|1,custom|winogrande|0|1".split(",")

    if kwargs.pop('extract_metrics', False):
        run_extract_metrics(checkpoints, tasks=tasks)
    else:
        for task in tasks:
            run_eval_experiments(hcg_experiments, job_description_prefix="Eval HCG: ", tasks=task, **kwargs)


    return



if __name__ == "__main__":

    import sys

    dry = len(sys.argv) > 1 and sys.argv[1] == 'dry'
    extract_metrics = len(sys.argv) > 1 and sys.argv[1] == 'extract_metrics'

    print("dry", dry, 'extract_metrics', extract_metrics)

    eval_hcg_adaptive_pretrain_all_tasks_parallel(dry=dry, extract_metrics=extract_metrics)