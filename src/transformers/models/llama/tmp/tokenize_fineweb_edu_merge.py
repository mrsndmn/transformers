import argparse
import os
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from datasets import load_dataset, Dataset, concatenate_datasets

if __name__ == "__main__":

    output_dir = os.listdir('./fineweb_edu_tokenized_gpt2_eos/')
    print(output_dir)

    all_datasets = []
    for data_file in output_dir:
        dataset = Dataset.load_from_disk(f'./fineweb_edu_tokenized_gpt2_eos/{data_file}')
        all_datasets.append(dataset)

    all_datasets = concatenate_datasets(all_datasets)
    all_datasets.save_to_disk('./fineweb_edu_tokenized_gpt2_eos/all_datasets.parquet')
    print(f"Saved to ./fineweb_edu_tokenized_gpt2_eos/all_datasets.parquet")