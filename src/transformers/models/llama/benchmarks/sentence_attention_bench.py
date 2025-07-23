import torch
from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM, special_token_mask_to_clothest_token_idx_slow

from transformers import AutoTokenizer

LIPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."


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

    input_ids = tokenizer.encode(LIPSUM * 10, return_tensors="pt")
    input_ids = input_ids.to("cuda")

    special_embeddings_mask = torch.zeros_like(input_ids).to("cuda")

    if model.config.end_of_sentence_token_id is not None:
        special_embeddings_mask[input_ids == model.config.end_of_sentence_token_id] = 1

    print("sum special tokens", special_embeddings_mask.sum().item())
    print("total tokens      ", input_ids.shape[1])

    clothest_end_of_sentence_token_idx = special_token_mask_to_clothest_token_idx_slow(special_embeddings_mask).to("cuda")

    with torch.no_grad():
        event = torch.cuda.Event(enable_timing=True)
        event_2 = torch.cuda.Event(enable_timing=True)

        event.record()

        model(
            input_ids=input_ids,
            special_embeddings_mask=special_embeddings_mask,
            clothest_end_of_sentence_token_idx=clothest_end_of_sentence_token_idx,
        )

        event_2.record()

        event.synchronize()
        event_2.synchronize()

        print(f"Prefill Time: {event.elapsed_time(event_2)}")
        print(f"Peak Memory Usage: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")

