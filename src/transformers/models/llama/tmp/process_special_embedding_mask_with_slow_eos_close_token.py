import torch
import os
from tqdm.auto import tqdm
import datasets
from datasets import Dataset
import sys
from transformers.models.llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow


special_token_id = 49152

def process_dataset_item(dataset_item):
    input_ids = torch.tensor(dataset_item['input_ids']).unsqueeze(0)
    special_embeddings_mask = input_ids == special_token_id
    clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask)

    return {
        'input_ids': input_ids[0].numpy().tolist(),
        'attention_mask': dataset_item['attention_mask'],
        'special_embeddings_mask': special_embeddings_mask[0].numpy().tolist(),
        'clothest_end_of_sentence_token_idx': clothest_end_of_sentence_token_idx[0].numpy().tolist(),
    }

if __name__ == "__main__":

    dataset_path = f'./fineweb_edu_tokenized_gpt2_eos/'
    targer_dir = f'./fineweb_edu_tokenized_gpt2_with_special_embedding_mask_clothest_eos_token_idx'
    dataset_shard = sorted(os.listdir(dataset_path))[:10]
    print(dataset_shard)

    columns_to_keep = ['input_ids', 'attention_mask', 'special_embeddings_mask', 'clothest_end_of_sentence_token_idx']


    for data_file in tqdm(dataset_shard, desc='Loading datasets'):

        shard_targer_dir = f'{targer_dir}/{data_file}'
        if os.path.exists(shard_targer_dir):
            continue

        dataset = Dataset.load_from_disk(f'{dataset_path}/{data_file}')

        dataset = dataset.remove_columns( list(set(dataset.column_names) - set(columns_to_keep)) )

        # dataset = dataset.select(range(1000))
        dataset = dataset.map(process_dataset_item, num_proc=32)

        dataset.save_to_disk(shard_targer_dir)



