import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm.auto import tqdm

import torch
from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM, special_token_mask_to_clothest_token_idx_slow

from transformers import AutoTokenizer, DynamicCache

LIPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."


def scrooge_prefill(model, input_ids, attention_mask, special_embeddings_mask, clothest_end_of_sentence_token_idx, trim_kv_cache=True):

    prev_sentence_i = 0 # (attention_mask == 0).sum().item()

    assert clothest_end_of_sentence_token_idx.shape[0] == 1, 'only single size batch is supported'

    eos_tokens_idxs = set(clothest_end_of_sentence_token_idx.cpu().numpy().tolist()[0])
    eos_tokens_idxs.remove(0)
    eos_tokens_idxs = sorted(eos_tokens_idxs)
    eos_tokens_idxs = eos_tokens_idxs + [input_ids.shape[1]]

    # TODO Sentence Cache - saves only the last sentence embedding
    past_key_values = DynamicCache()

    # TODO Calculate Last Chunk with possibly no sentence id

    # model.model.
    # full_ones_attention_mask = torch.ones_like(input_ids)

    hidden_states = []

    for i, sentence_i in enumerate(eos_tokens_idxs):
        kv_length = past_key_values.get_seq_length()
        if trim_kv_cache:
            assert past_key_values.get_seq_length() == i, 'cache seq len should be equal to number of sentences'

        # print("prev_sentence_i, sentence_i", prev_sentence_i, sentence_i)
        # current_attention_mask = full_ones_attention_mask[:, :(sentence_i-prev_sentence_i + past_key_values.get_seq_length())]

        outputs = model(
            input_ids=input_ids[:, prev_sentence_i:sentence_i],
            attention_mask=attention_mask[:, (prev_sentence_i-kv_length):sentence_i],
            special_embeddings_mask=special_embeddings_mask[:, (prev_sentence_i-kv_length):sentence_i],
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx[:, (prev_sentence_i-kv_length):sentence_i],
            past_key_values=past_key_values,
            cache_position=torch.arange(prev_sentence_i, sentence_i, device=input_ids.device),
            is_sentence_chunked_prefill=trim_kv_cache,
            output_hidden_states=True,
        )
        prev_sentence_i = sentence_i

        hidden_states.append(outputs.hidden_states)

        # Leave only sentence attention cache
        if i != len(eos_tokens_idxs) - 1 and trim_kv_cache:
            for idx in range(len(past_key_values.key_cache)):
                if past_key_values.key_cache[idx] != []:
                    past_key_values.key_cache[idx]   = torch.cat([past_key_values.key_cache[idx][..., :i, :], past_key_values.key_cache[idx][..., -1:, :]], dim=-2)
                    past_key_values.value_cache[idx] = torch.cat([past_key_values.value_cache[idx][..., :i, :], past_key_values.value_cache[idx][..., -1:, :]], dim=-2)

            assert past_key_values.get_seq_length() == i + 1, 'cache seq len should be equal to number of sentences'

    last_outputs = outputs

    return {
        "last_outputs": last_outputs,
        "past_key_values": past_key_values,
        "hidden_states": hidden_states,
    }


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
    dataset = dataset.select(range(10))

    sum_tokens = 0
    sum_special_tokens = 0

    special_token_id = tokenizer.end_of_sentence_token_id


    base_seq_len = 1024
    sequence_lengths = []
    mean_scroodge_peak_memory = []
    mean_base_peak_memory = []

    with torch.no_grad():

        for sequence_scaling in [ 1, 4, 8, 16, 32 ]:

            base_peak_memory = []
            scroodge_peak_memory = []

            for item in tqdm(dataset, desc=f"Sequence Scaling: {sequence_scaling}"):

                torch.cuda.reset_peak_memory_stats()

                event = torch.cuda.Event(enable_timing=True)
                event_2 = torch.cuda.Event(enable_timing=True)

                input_ids = torch.tensor(item["input_ids"], device="cuda").unsqueeze(0)
                input_ids = input_ids.repeat(1, sequence_scaling)
                special_embeddings_mask = input_ids == special_token_id
                clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask)

                sum_tokens += input_ids.shape[1]
                sum_special_tokens += special_embeddings_mask.sum().item()

                event.record()

                rich_prefill_outputs = model(
                    input_ids=input_ids,
                    special_embeddings_mask=special_embeddings_mask,
                    clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
                    use_cache=True,
                )

                event_2.record()

                event.synchronize()
                event_2.synchronize()

                # print(f"Prefill Time: {event.elapsed_time(event_2)}")
                base_peak_memory.append(torch.cuda.max_memory_allocated() / 1024 ** 2)

                rich_prefill_outputs_logits = rich_prefill_outputs.logits.detach().cpu()
                del rich_prefill_outputs


                torch.cuda.empty_cache()

                torch.cuda.reset_peak_memory_stats()

                scroodge_last_outputs, _ = scrooge_prefill(model, input_ids, special_embeddings_mask, clothest_end_of_sentence_token_idx)

                # TODO validate scroodge prefill
                # assert torch.allclose(rich_prefill_outputs_logits[0, -1, :], scroodge_last_outputs.logits.cpu()[0, -1, :])
                scroodge_peak_memory.append(torch.cuda.max_memory_allocated() / 1024 ** 2)


            sequence_lengths.append(base_seq_len * sequence_scaling)
            mean_scroodge_peak_memory.append(sum(scroodge_peak_memory) / len(scroodge_peak_memory))
            mean_base_peak_memory.append(sum(base_peak_memory) / len(base_peak_memory))

    print(f"Average tokens: {sum_tokens / len(dataset)}")
    print(f"Average special tokens: {sum_special_tokens / len(dataset)}")

    print(f"Average tokens per special token (compression ratio): {sum_tokens / sum_special_tokens}")

    df = pd.DataFrame({
        "sequence_lengths": sequence_lengths,
        "mean_scroodge_peak_memory": mean_scroodge_peak_memory,
        "mean_base_peak_memory": mean_base_peak_memory,
    })
    df.to_csv("src/transformers/models/llama/benchmarks/plots/sentence_attention_bench_memory_kv_cache.csv", index=False)
    print('saved csv to src/transformers/models/llama/benchmarks/plots/sentence_attention_bench_memory_kv_cache.csv')

    plt.plot(sequence_lengths, mean_scroodge_peak_memory, label="Scrooge", color="red")
    plt.plot(sequence_lengths, mean_base_peak_memory, label="Base", color="blue")
    plt.xlabel("Sequence Length")
    plt.ylabel("Peak Memory Usage (MB)")
    plt.title("Peak Memory Usage for Sentence Attention")
    plt.legend()
    plt.show()
    plt.savefig("src/transformers/models/llama/benchmarks/plots/sentence_attention_bench_memory_kv_cache.png")
    print('saved plot to src/transformers/models/llama/benchmarks/plots/sentence_attention_bench_memory_kv_cache.png')
