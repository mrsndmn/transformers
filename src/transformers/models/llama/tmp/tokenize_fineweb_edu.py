import argparse
import os
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from datasets import load_dataset

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--data_file", type=str)
    args.add_argument("--num_proc", type=int, default=48)
    args = args.parse_args()

    tokenizer = GPT2TokenizerFastEOS.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")

    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    data_files = [ args.data_file ]
    # for i in range(6):
    #     for j in range(10):
    #         data_files.append(f"sample/100BT/{i:03}_{j:05}.parquet")

    smollm_corpus = load_dataset("HuggingFaceFW/fineweb-edu", data_files=data_files, num_proc=args.num_proc)
    smollm_corpus = smollm_corpus['train']


    def tokenize_function(examples):
        text = examples['text']

        tokenized_inputs = tokenizer(text, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')

        return tokenized_inputs

    smollm_corpus = smollm_corpus.map(tokenize_function, batched=True, num_proc=48)

    output_dir = f'./fineweb_edu_tokenized_gpt2_eos/{os.path.basename(args.data_file)}'
    smollm_corpus.save_to_disk(output_dir)
    print(f"Saved to {output_dir}")
