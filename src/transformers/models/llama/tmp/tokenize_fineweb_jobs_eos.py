
import os

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":

    client, extra_options = training_job_api_from_profile('default')

    workdir = os.getcwd()

    author_name = "d.tarasov"

    for i in range(6):
        for j in range(10):
            data_file = f"sample/100BT/{i:03}_{j:05}.parquet"

            result = client.run_job(
                payload={
                    'script': f"bash -c 'cd {workdir} && /workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/python src/transformers/models/llama/tmp/tokenize_fineweb_edu.py --data_file {data_file} --tokenizer gpt2_eos --output_dir ./fineweb_edu_tokenized_gpt2_eos'",
                    'job_desc': f'Tokenize fineweb edu gpt2_eos {data_file} #{author_name} #rnd #multimodal @mrsndmn',
                    'instance_type': 'a100.1gpu',
                    'region': extra_options['region'],
                    'env_variables': {
                        'PYTHONPATH': './src',
                    },
                    'type': 'binary_exp',
                    'shm_size_class': 'medium',
                    'base_image': 'cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36',
                    'n_workers': 1,              # Количество воркеров.
                    'processes_per_worker': 1,   # Количество процессов на воркер. Для accelerate нужно запускать 1 процесс на воркер. Для torchrun лучше не заполнять этот параметр. По умолчанию запускается по количеству GPU на одном воркере - это подходит для torchrun.
                }
            )

            print(data_file, result)

