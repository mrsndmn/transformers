
import time

import torch

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_adaptive_llama import AdaptiveFanInGumbel, AdaptiveFanOut, AdaptiveFanInOutput, AdaptiveFanOutOutput, AdaptiveLlamaModel

import torch
from transformers.models.llama.merges_transform.generate_merges import generate_merges_transform, fan_out_restore_residuals, prune_tokens_concrete


if __name__ == "__main__":

    config_cuda_kernel = LlamaConfig(hidden_size=1024, num_hidden_layers=2, attn_implementation='eager', generate_merges_transform_impl="cuda_kernel")

    cuda_adaptive_fan_in_gumbel = AdaptiveFanInGumbel(config_cuda_kernel)

    batch_size = 100
    seq_len = 128

    merging_map_1 = torch.zeros([batch_size, seq_len, 2])
    merging_map_1[:, :, 0] = 1.

    special_tokens_mask = torch.zeros([batch_size, seq_len], dtype=torch.bool)
    special_tokens_mask[:, 0] = True
    special_tokens_mask[:, -1] = True

    test_cases = [
        {
            "name": "dummy no merging",
            "merging_map": merging_map_1,
            "attention_mask": torch.ones([batch_size, seq_len], dtype=torch.bool),
            "special_tokens_mask": special_tokens_mask,
        },
    ]

    # todo make fixtures not golang-style tests
    for test_case in test_cases:
        test_case_name = test_case['name']
        merging_map = test_case['merging_map']
        attention_mask = test_case['attention_mask']
        special_tokens_mask = test_case['special_tokens_mask']

        cuda_merging_map = merging_map.to('cuda').to(dtype=torch.float32)
        cuda_attention_mask = attention_mask.to('cuda').to(dtype=torch.bool)
        special_tokens_mask = special_tokens_mask.to('cuda').to(dtype=torch.bool)

        n_runs = 100

        cuda_time_start = time.time()
        for _ in range(n_runs):
            cuda_aggregated_embeddings_transform, cuda_merged_embeddings_counts, cuda_merged_attention_mask = cuda_adaptive_fan_in_gumbel.generate_merges_transform(cuda_merging_map, cuda_attention_mask, special_tokens_mask)
            cuda_aggregated_embeddings_transform.sum().item()
        cuda_duration = (time.time() - cuda_time_start) / n_runs

        # py_cuda_time_start = time.time()
        # # for _ in range(n_runs):
        # #     py_aggregated_embeddings_transform, py_merged_embeddings_counts, py_merged_attention_mask = py_adaptive_fan_in_gumbel.generate_merges_transform(cuda_merging_map, cuda_attention_mask, special_tokens_mask)
        # #     py_aggregated_embeddings_transform.sum().item()
        # py_cuda_duration = (time.time() - py_cuda_time_start) / n_runs

        print(f"bs={batch_size} seq_len={seq_len} cuda_duration", cuda_duration)
