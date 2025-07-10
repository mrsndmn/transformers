from collections import Counter
import glob
import time
import client_lib # импортируем библиотеку для работы с ML Space
import json
from rich.console import Console
import requests
from tqdm import tqdm

from client_lib import environment

if __name__ == "__main__":

    import sys
    region = "SR004"

    resp = requests.post(
        f"http://{environment.GW_API_ADDR}/job_list",
        headers={"X-Api-Key": environment.GW_API_KEY, "X-Namespace": environment.NAMESPACE},
        json={"region": region},
    )

    jobs = resp.json()

    print("len jobs:", len(jobs))

    cnt = Counter()

    # jobs[]
    for job in jobs:
        cnt[job['status']] += 1

    print("job keys:", jobs[0].keys())

    print(cnt)

    # for job in tqdm(jobs):
    #     print("job name", job['job_name'])
    #     resp = client_lib.kill(job['job_name'], region=region)
    #     print(resp)

