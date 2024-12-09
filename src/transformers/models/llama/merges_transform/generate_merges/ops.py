import torch
from torch import Tensor

import time

__all__ = ["generate_merges_transform"]


def generate_merges_transform(
        merging_map: Tensor,
        attention_mask: Tensor
        ) -> Tensor:
    """Generates Merging map with custom cuda kernel

    Args:
        merging_map (Tensor): _description_
        attention_mask (Tensor): _description_
    """
    device = merging_map.device
    batch_size = merging_map.shape[0]
    seq_len = merging_map.shape[1]
    
    torch._check(len(merging_map.shape) == 3)
    torch._check(merging_map.shape[-1] == 2)
    torch._check(merging_map.shape[:2] == attention_mask.shape[:2])
    torch._check(merging_map.dtype == torch.float)
    torch._check(attention_mask.dtype == torch.bool)
    torch._check(merging_map.device == attention_mask.device)

    # TODO crop length
    # aggregated_embeddings_transform = torch.zeros([batch_size, seq_len, seq_len], device=device)
    # merged_embeddings_counts = torch.zeros([batch_size, seq_len], dtype=torch.long, device=device)
    # merged_attention_mask = torch.zeros([batch_size, seq_len], dtype=torch.bool, device=device)

    aggregated_embeddings_transform_output, merged_embeddings_counts, merged_attention_mask = torch.ops.generate_merges.generate_merges_transform.default(
        merging_map,
        attention_mask
    )
    
    max_new_seq_len = merged_attention_mask.sum(dim=-1).max()
    if max_new_seq_len < merged_attention_mask.shape[1]:
        aggregated_embeddings_transform_output = aggregated_embeddings_transform_output[:, :max_new_seq_len]
        merged_embeddings_counts = merged_embeddings_counts[:, :max_new_seq_len]
        merged_attention_mask = merged_attention_mask[:, :max_new_seq_len]
    
    return aggregated_embeddings_transform_output, merged_embeddings_counts, merged_attention_mask

def _backward(ctx, output_merging_map_grad, merged_embeddings_counts, merged_attention_mask):
    # [bs, seq_len, 2] [bs, new_seq_len, seq_len]
    saved_merging_map, output_transform_matrix, merged_embeddings_counts = ctx.saved_tensors
    
    batch_size = saved_merging_map.shape[0]
    seq_len = saved_merging_map.shape[1]
    
    # [bs, new_seq_len, seq_len]
    # output_merging_map_grad
    output_merging_map_grad_zeroed = output_merging_map_grad
    output_merging_map_grad_zeroed[~output_transform_matrix.bool()] = 0
    
    # expected to be [ bs, seq_len, 2 ]
    grad_merging_map_output = None
    if ctx.needs_input_grad[0]:
        # [ bs, seq_len, 2 ]
        grad_merging_map = torch.bmm(output_merging_map_grad_zeroed, saved_merging_map)
        # breakpoint()
        
        # grad_merging_map = grad_merging_map.flatten(0, 1)
        # merged_embeddings_counts = merged_embeddings_counts.flatten(0, 1)

        # start_repeat_interleaved = time.time()

        # 10 it = 01:07
        # grad_merging_map_output = torch.zeros_like(grad_merging_map)
        # merged_embeddings_counts_sum = merged_embeddings_counts.sum(dim=-1)
        # for batch_i in range(batch_size):
        #     repeat_mask = merged_embeddings_counts[batch_i]
        #     total_tokens = merged_embeddings_counts_sum[batch_i].item()
        #     grad_merging_map_output[batch_i, :total_tokens] = grad_merging_map[batch_i].repeat_interleave(repeat_mask, dim=0)

        # 10 it = 58 sec
        grad_merging_map_output = torch.ops.generate_merges.batch_repeat_interleave_for_merges_count.default(grad_merging_map, merged_embeddings_counts)
        # assert (grad_merging_map_output_cuda == grad_merging_map_output).all()
        # grad_merging_map_output = grad_merging_map_output_cuda

        # print("backward grad compute:", time.time() - start_repeat_interleaved)

    return grad_merging_map_output, None


def _setup_context(ctx, inputs, output):
    merging_map, attention_mask = inputs
    output_transform_matrix, merged_embeddings_counts, merged_attention_mask = output
    saved_merging_map = None
    if ctx.needs_input_grad[0]:
        saved_merging_map = merging_map

    ctx.save_for_backward(saved_merging_map, output_transform_matrix, merged_embeddings_counts)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "generate_merges::generate_merges_transform", _backward, setup_context=_setup_context)


# @torch.library.register_fake("generate_merges::generate_merges_transform")
# def _(merging_map, attention_mask):
    
#     batch_size = merging_map.shape[0]
#     seq_len = merging_map.shape[1]
#     device = merging_map.device
    
#     output_merging_map = torch.empty([batch_size, seq_len, seq_len], device=device)
#     merged_embeddings_counts = torch.zeros([batch_size, seq_len], dtype=torch.long, device=device)
#     merged_attention_mask = torch.zeros([batch_size, seq_len], dtype=torch.bool, device=device)

#     return output_merging_map, merged_embeddings_counts, merged_attention_mask
