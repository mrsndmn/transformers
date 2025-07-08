import json
import datasets
from transformers import AutoTokenizer

def get_metrics_scores(ds):

    acc_norms = []

    metrics = ds['metrics']
    for m in metrics:
        m_json = json.loads(m.replace("'", '"'))
        acc_norm = m_json['acc_norm']
        acc_norms.append(acc_norm)

    return acc_norms


if __name__ == "__main__":


    adaptive_dataset_path = '/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/exps_evaluation/details/._adaptive_slm2_1.7B_pretrain_with_end_of_sentence_token_ftte_w_0.100_l_8-16_8MTABS8F_checkpoint-8000/2025-07-08T10-11-44.017325/details_custom|hellaswag|0_2025-07-08T10-11-44.017325.parquet'

    vanilla_dataset_path = '/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out/exps_evaluation/details/._vanilla_slm2_1.7B_pretrain_16L_w_0.000_l_-_F0UCD5QW_checkpoint-8000_/2025-07-08T10-14-45.571105/details_custom|hellaswag|0_2025-07-08T10-14-45.571105.parquet'

    adaptive_ds = datasets.load_dataset("parquet", data_files=adaptive_dataset_path)
    adaptive_ds = adaptive_ds['train']

    vanilla_ds = datasets.load_dataset("parquet", data_files=vanilla_dataset_path)
    vanilla_ds = vanilla_ds['train']

    adaptive_acc_norms = get_metrics_scores(adaptive_ds)
    vanilla_acc_norms = get_metrics_scores(vanilla_ds)

    tokenizer = AutoTokenizer.from_pretrained("./adaptive_slm2_1.7B_pretrain_with_end_of_sentence_token_ftte_w_0.100_l_8-16_8MTABS8F/checkpoint-8000")

    for i in range(len(adaptive_acc_norms)):
        adaptive_acc_norm = adaptive_acc_norms[i]
        vanilla_acc_norm = vanilla_acc_norms[i]

        vanilla_details = vanilla_ds[i]
        adaptive_details = adaptive_ds[i]

        vanilla_input_tokens = tokenizer.batch_decode(json.loads(vanilla_details['input_tokens']))
        vanilla_cont_tokens = tokenizer.batch_decode(json.loads(vanilla_details['cont_tokens']))

        adaptive_input_tokens = tokenizer.batch_decode(json.loads(adaptive_details['input_tokens']))
        adaptive_cont_tokens = tokenizer.batch_decode(json.loads(adaptive_details['cont_tokens']))

        if adaptive_acc_norm != vanilla_acc_norm:
            print(f"Adaptive acc norm: {adaptive_acc_norm}, Vanilla acc norm: {vanilla_acc_norm}")
            breakpoint()

    breakpoint()


