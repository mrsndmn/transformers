import torch
from torch import Tensor

import time

__all__ = ["generate_merges_transform", "fan_out_restore_residuals"]


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
    
    assert (merging_map.sum(dim=-1).bool() == attention_mask).all()

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


def _backward_generate_merges_transform(ctx, output_merging_map_grad, merged_embeddings_counts, merged_attention_mask):
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
        # assert ((grad_merging_map_output == 0).sum(dim=-1) == 1).all()
        # breakpoint()
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
    "generate_merges::generate_merges_transform", _backward_generate_merges_transform, setup_context=_setup_context)


def fan_out_restore_residuals(
        merged_embeddings_counts: Tensor, # [ bs, seq_len ]
        hidden_states: Tensor, # [ bs, new_seq_len, hidden_dim ]
        residual_hidden_states: Tensor, # [ bs, seq_len, hidden_dim ]
        ) -> Tensor:

    torch._check(len(merged_embeddings_counts.shape) == 2)
    torch._check(merged_embeddings_counts.shape[:2] == hidden_states.shape[:2])
    torch._check(merged_embeddings_counts.dtype == torch.long)
    torch._check(hidden_states.dtype == torch.float)
    torch._check(residual_hidden_states.dtype == torch.float)
    torch._check(residual_hidden_states.device == residual_hidden_states.device)
    torch._check(merged_embeddings_counts.device == residual_hidden_states.device)

    restored_hidden_states = torch.ops.generate_merges.fan_out_restore_residuals.default(
        merged_embeddings_counts,
        hidden_states,
        residual_hidden_states,
    )

    return restored_hidden_states


def _backward_fan_out_restore_residuals(ctx, restored_hidden_states_grad):
    # restored_hidden_states_grad ~ [ bs, seq_len, hidden_dim ]
    (merged_embeddings_counts,) = ctx.saved_tensors
    
    hidden_states_grad = None
    residual_hidden_states_grad = None
    if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
        hidden_states_grad, residual_hidden_states_grad = torch.ops.generate_merges.backward_fan_out_restore_residuals.default(
            merged_embeddings_counts,
            restored_hidden_states_grad,
        )

    assert hidden_states_grad is not None
    assert residual_hidden_states_grad is not None
    assert residual_hidden_states_grad.shape == restored_hidden_states_grad.shape
    
    return None, hidden_states_grad, residual_hidden_states_grad


def _setup_context_fan_out_restore_residuals(ctx, inputs, output):
    merged_embeddings_counts, hidden_states, residual_hidden_states = inputs

    # if ctx.needs_input_grad[1]:
    #     saved_merging_map = merging_map
    # if ctx.needs_input_grad[2]:
    #     saved_merging_map = merging_map

    ctx.save_for_backward(merged_embeddings_counts)


torch.library.register_autograd(
    "generate_merges::fan_out_restore_residuals", _backward_fan_out_restore_residuals, setup_context=_setup_context_fan_out_restore_residuals)

