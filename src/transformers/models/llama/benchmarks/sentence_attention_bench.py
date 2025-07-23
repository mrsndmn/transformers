from datasets import Dataset
from tqdm.auto import tqdm

import torch
from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM, special_token_mask_to_clothest_token_idx_slow

from transformers import AutoTokenizer, DynamicCache

LIPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."


def scrooge_prefill(model, input_ids, special_embeddings_mask, clothest_end_of_sentence_token_idx):

    prev_sentence_i = 0

    assert clothest_end_of_sentence_token_idx.shape[0] == 1, 'only single size batch is supported'

    eos_tokens_idxs = set(clothest_end_of_sentence_token_idx.cpu().numpy().tolist()[0])
    eos_tokens_idxs.remove(0)
    eos_tokens_idxs = sorted(eos_tokens_idxs)

    # TODO Sentence Cache - saves only the last sentence embedding
    past_key_values = DynamicCache()

    # TODO Calculate Last Chunk with possibly no sentence id

    # model.model.

    for i, sentence_i in enumerate(eos_tokens_idxs):
        assert past_key_values.get_seq_length() == i

        print("prev_sentence_i, sentence_i", prev_sentence_i, sentence_i)

        outputs = model(
            input_ids=input_ids[:, prev_sentence_i:sentence_i],
            special_embeddings_mask=special_embeddings_mask[:, prev_sentence_i:sentence_i],
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx[:, prev_sentence_i:sentence_i],
            past_key_values=past_key_values,
        )
        prev_sentence_i = sentence_i

        # Leave only sentence attention cache
        for idx in range(len(past_key_values.key_cache)):
            if past_key_values.key_cache[idx] != []:
                past_key_values.key_cache[idx]   = past_key_values.key_cache[idx][..., -(i + 1):, :]
                past_key_values.value_cache[idx] = past_key_values.value_cache[idx][..., -(i + 1):, :]

        assert past_key_values.get_seq_length() == i + 1, 'cache seq len should be equal to number of sentences'

    return past_key_values


if __name__ == "__main__":

    model_class = SentenceLlamaForCausalLM
    checkpoint_dir = "./sentence_Llama-3.2-1B_pretrain_with_end_of_sentence_full_BTLCR6IG/checkpoint-2000"

    model = model_class.from_pretrained(checkpoint_dir)
    model.eval()
    model.to("cuda")

    model.config._attn_implementation = "sentence_attention"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # Prefill
    # Peak Memory Usage, Time

    dataset = Dataset.load_from_disk('./fineweb_edu_tokenized_Llama-3.2-1B_with_eos_token/shard_9')
    dataset = dataset.select(range(100))

    sum_tokens = 0
    sum_special_tokens = 0

    with torch.no_grad():
        for item in tqdm(dataset):

            torch.cuda.reset_peak_memory_stats()

            event = torch.cuda.Event(enable_timing=True)
            event_2 = torch.cuda.Event(enable_timing=True)

            input_ids = torch.tensor(item["input_ids"], device="cuda").unsqueeze(0)
            special_embeddings_mask = torch.tensor(item["special_embeddings_mask"], device="cuda").unsqueeze(0)

            sum_tokens += input_ids.shape[1]
            sum_special_tokens += special_embeddings_mask.sum().item()

            clothest_end_of_sentence_token_idx = torch.tensor(item["clothest_end_of_sentence_token_idx"], device="cuda").unsqueeze(0)

            event.record()

            # model(
            #     input_ids=input_ids,
            #     special_embeddings_mask=special_embeddings_mask,
            #     clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
            # )

            event_2.record()

            event.synchronize()
            event_2.synchronize()

            # print(f"Prefill Time: {event.elapsed_time(event_2)}")
            print(f"Peak Memory Usage: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")

            torch.cuda.empty_cache()

            torch.cuda.reset_peak_memory_stats()

            scrooge_prefill(model, input_ids, special_embeddings_mask, clothest_end_of_sentence_token_idx)

            print(f"Scrooge Peak Memory Usage: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")



    print(f"Average tokens: {sum_tokens / len(dataset)}")
    print(f"Average special tokens: {sum_special_tokens / len(dataset)}")

    print(f"Average tokens per special token (compression ratio): {sum_tokens / sum_special_tokens}")