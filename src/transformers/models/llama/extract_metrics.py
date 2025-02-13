import sys
import json

if __name__ == '__main__':

    file_name = sys.argv[1]

    with open(file_name, 'r') as f:
        json_data = json.load(f)

    bench_keys = [
        'custom|mmlu_pro_cloze|0',
        'custom|mmlu_cloze:_average|0',
        'custom|hellaswag|0',
        'custom|arc:_average|0',
        'custom|piqa|0',
        'custom|trivia_qa|0',
    ]
    max_len = max(map(len, bench_keys))

    for key in bench_keys:

        if 'acc_norm' in json_data['results'][key]:
            metric = json_data['results'][key]['acc_norm']
            metric_stderr = json_data['results'][key]['acc_norm_stderr']
        elif 'qem' in json_data['results'][key]:
            metric = json_data['results'][key]['qem']
            metric_stderr = json_data['results'][key]['qem_stderr']

        space = " " * (max_len - len(key) + 1)
        print(key, space, "\t", f"{metric*100:.2f}", '\tstderr', f"{metric_stderr*100:.2f}")