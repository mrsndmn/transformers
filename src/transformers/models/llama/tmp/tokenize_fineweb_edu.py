from datasets import load_dataset
import sys
from transformers.models.llama.modeling_sentence_llama import special_token_mask_to_clothest_token_idx_slow

from transformers.tokenization_utils_fast import PreTrainedTokenizerFastEOS
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFastEOS
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFastEOS
from transformers import AutoTokenizer


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_num', type=int, required=True)
    parser.add_argument('--total_shards', type=int, required=True)
    parser.add_argument('--pretrained_model_name', type=str, required=True)
    parser.add_argument('--with_eos_token', action='store_true')
    args = parser.parse_args()

    shard_num = args.shard_num
    total_shards = args.total_shards
    pretrained_model_name = args.pretrained_model_name # HuggingFaceTB/SmolLM2-1.7B / unsloth/Llama-3.2-1B

    pretrained_model_name_short = pretrained_model_name.split('/')[-1]

    print(f'pretrained_model_name_short: {pretrained_model_name_short}')

    suffix = '_with_eos_token' if args.with_eos_token else ''

    targer_dir = f'./fineweb_edu_tokenized_{pretrained_model_name_short}{suffix}'

    shard_targer_dir = f'{targer_dir}/shard_{shard_num}'


    dataset_name = 'HuggingFaceFW/fineweb-edu'
    dataset = load_dataset(dataset_name, 'sample-10BT', num_proc=16, split='train').shard(total_shards, shard_num)

    dataset = dataset.select(range(1000))

    # dataset = dataset.remove_columns( list(set(dataset.column_names) - set(columns_to_keep)) )

    columns_to_keep = ['input_ids', 'attention_mask', 'special_embeddings_mask', 'clothest_end_of_sentence_token_idx']
    columns_to_keep = list(set(dataset.column_names) - set(columns_to_keep))

    tokenizer_class = type(AutoTokenizer.from_pretrained(pretrained_model_name)).__name__

    if args.with_eos_token:
        if tokenizer_class == 'GPT2TokenizerFast':
            tokenizer_class = GPT2TokenizerFastEOS
        elif tokenizer_class == 'PreTrainedTokenizerFast':
            tokenizer_class = PreTrainedTokenizerFastEOS
        elif tokenizer_class == 'Qwen2TokenizerFast':
            tokenizer_class = Qwen2TokenizerFastEOS
        else:
            raise ValueError(f"Invalid tokenizer class: {tokenizer_class}")

    tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)

    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    special_token_id = tokenizer.end_of_sentence_token_id

    def process_dataset_item(dataset_item):
        text = dataset_item['text']

        tokenized_inputs = tokenizer(text, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')

        input_ids = tokenized_inputs['input_ids']
        special_embeddings_mask = input_ids == special_token_id
        clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask)

        return {
            'input_ids': input_ids[0].numpy().tolist(),
            'attention_mask': tokenized_inputs['attention_mask'],
            'special_embeddings_mask': special_embeddings_mask[0].numpy().tolist(),
            'clothest_end_of_sentence_token_idx': clothest_end_of_sentence_token_idx[0].numpy().tolist(),
        }


    dataset = dataset.map(process_dataset_item, num_proc=16, remove_columns=columns_to_keep)

    dataset.save_to_disk(shard_targer_dir)

