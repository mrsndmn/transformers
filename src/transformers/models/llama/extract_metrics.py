import sys
import json

if __name__ == '__main__':

    file_name = sys.argv[1]

    with open(file_name, 'r') as f:
        json_data = json.load(f)

    bench_keys = [
        'custom|arc:_average|0',
        'custom|piqa|0',
        'custom|trivia_qa|0',
        'custom|mmlu_cloze:_average|0',
        'custom|mmlu_pro_cloze|0',
        'custom|gsm8k|5',
    ]
    max_len = max(map(len, bench_keys))

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
        elif len(metric_dict.keys()) > 0:
            raise ValueError("unknown metrics:", metric_dict)

        space = " " * (max_len - len(key) + 1)
        print(key, space, "\t", f"{metric*100:.2f}", '\tstderr', f"{metric_stderr*100:.2f}")
