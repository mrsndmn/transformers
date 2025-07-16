
import os

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":

    client, extra_options = training_job_api_from_profile('default')

    workdir = os.getcwd()

    author_name = 'd.tarasov'

    total_shards = 4

    for shard_num in range(total_shards):
            result = client.run_job(
                payload={
                    'script': f"bash {workdir}/src/transformers/models/llama/tmp/jobs/tokenize_fineweb.sh {shard_num} {total_shards}",
                    'job_desc': f'Tokenize FineWebEdu dataset #{author_name} #rnd #multimodal @mrsndmn',
                    'instance_type': 'a100.1gpu',
                    'region': extra_options['region'],
                    'type': 'binary_exp',
                    'shm_size_class': 'medium',
                    'base_image': 'cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36',
                    'n_workers': 1,              # Количество воркеров.
                    'processes_per_worker': 1,   # Количество процессов на воркер. Для accelerate нужно запускать 1 процесс на воркер. Для torchrun лучше не заполнять этот параметр. По умолчанию запускается по количеству GPU на одном воркере - это подходит для torchrun.
                }
            )

            print(shard_num, result)

