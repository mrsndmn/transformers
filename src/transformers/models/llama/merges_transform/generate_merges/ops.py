import torch
from torch import Tensor

import time

__all__ = ["fan_out_restore_residuals", "prune_tokens_concrete"]


def fan_out_restore_residuals(
        merged_embeddings_counts: Tensor, # [ bs, seq_len ]
        hidden_states: Tensor, # [ bs, new_seq_len, hidden_dim ]
        residual_hidden_states: Tensor, # [ bs, seq_len, hidden_dim ]
        residual_hidden_states_attention_mask: Tensor, # [ bs, seq_len ]
        ) -> Tensor:

    torch._check(len(merged_embeddings_counts.shape) == 2)
    torch._check(merged_embeddings_counts.shape[:2] == hidden_states.shape[:2])
    torch._check(merged_embeddings_counts.dtype == torch.long)
    torch._check(hidden_states.dtype == torch.float or hidden_states.dtype == torch.float16 or hidden_states.dtype == torch.bfloat16)
    # torch._check(residual_hidden_states.dtype == torch.float)
    torch._check(residual_hidden_states.device == residual_hidden_states.device)
    torch._check(merged_embeddings_counts.device == residual_hidden_states.device)

    orig_dtype = residual_hidden_states.dtype
    residual_hidden_states = residual_hidden_states.to(torch.float32)
    hidden_states = hidden_states.to(torch.float32)

    restored_hidden_states = torch.ops.generate_merges.fan_out_restore_residuals.default(
        merged_embeddings_counts,
        hidden_states,
        residual_hidden_states,
        residual_hidden_states_attention_mask,
    )

    restored_hidden_states = restored_hidden_states.to(orig_dtype)

    return restored_hidden_states


def _backward_fan_out_restore_residuals(ctx, restored_hidden_states_grad):
    # restored_hidden_states_grad ~ [ bs, seq_len, hidden_dim ]
    
    (merged_embeddings_counts, residual_hidden_states_attention_mask) = ctx.saved_tensors
        # [ bs ]
    restored_hidden_states_seq_lengths = residual_hidden_states_attention_mask.sum(dim=-1)

    # print("restored_hidden_states_seq_lengths", restored_hidden_states_seq_lengths)
    # print("merged_embeddings_counts", merged_embeddings_counts)

    hidden_states_grad = None
    residual_hidden_states_grad = None
    if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
        hidden_states_grad, residual_hidden_states_grad = torch.ops.generate_merges.backward_fan_out_restore_residuals.default(
            merged_embeddings_counts,
            restored_hidden_states_grad,
            restored_hidden_states_seq_lengths,
        )
        
        # print("_backward_fan_out_restore_residuals hidden_states_grad", hidden_states_grad)
        # breakpoint()

    assert hidden_states_grad is not None
    assert residual_hidden_states_grad is not None
    assert residual_hidden_states_grad.shape == restored_hidden_states_grad.shape
    
    return None, hidden_states_grad, residual_hidden_states_grad, None


def _setup_context_fan_out_restore_residuals(ctx, inputs, output):
    merged_embeddings_counts, hidden_states, residual_hidden_states, residual_hidden_states_attention_mask = inputs

    # if ctx.needs_input_grad[1]:
    #     saved_merging_map = merging_map
    # if ctx.needs_input_grad[2]:
    #     saved_merging_map = merging_map

    ctx.save_for_backward(merged_embeddings_counts, residual_hidden_states_attention_mask)


torch.library.register_autograd(
    "generate_merges::fan_out_restore_residuals", _backward_fan_out_restore_residuals, setup_context=_setup_context_fan_out_restore_residuals)


def prune_tokens_concrete(
        hidden_state: Tensor, # [ bs, seq_len, hidden_dim ]
        concrete_bool: Tensor, # [ bs, seq_len ]
        attention_mask: Tensor, # [ bs, seq_len ]
        special_embeddings_mask: Tensor = None, # [ bs, seq_len ]
        concrete: Tensor = None, # [ bs, seq_len ]
        ):
    """Prunes tokens based on concrete boolean mask and reorders them.
    
    Args:
        hidden_state: Input hidden states [batch_size, seq_len, hidden_dim]
        concrete_bool: Boolean mask for important tokens [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        special_embeddings_mask: Optional mask for special embeddings [batch_size, seq_len]
        concrete: Optional concrete values [batch_size, seq_len]
    
    Returns:
        Tuple containing:
        - Reordered hidden states [batch_size, new_seq_len, hidden_dim]
        - Merged embeddings counts [batch_size, new_seq_len]
        - Merged attention mask [batch_size, new_seq_len]
        - Merged special embeddings mask [batch_size, new_seq_len] (if special_embeddings_mask provided)
        - Merged concrete values [batch_size, new_seq_len] (if concrete provided)
    """
    
    hidden_state_dtype = hidden_state.dtype
    hidden_state = hidden_state.to(torch.float32)

    # Handle optional inputs
    device = hidden_state.device

    # Input validation
    torch._check(len(hidden_state.shape) == 3)
    torch._check(len(concrete_bool.shape) == 2)
    torch._check(concrete_bool.shape == attention_mask.shape)
    torch._check(concrete_bool.shape == special_embeddings_mask.shape)
    torch._check(concrete_bool.shape == concrete.shape)
    torch._check(concrete_bool.dtype == torch.bool)
    torch._check(concrete.dtype in (torch.float32, torch.bfloat16, torch.float16))
    torch._check(attention_mask.dtype == torch.long)
    torch._check(special_embeddings_mask.dtype == torch.long)

    outputs = torch.ops.generate_merges.prune_tokens_concrete_cuda.default(
        hidden_state,
        concrete_bool,
        attention_mask,
        special_embeddings_mask,
        concrete,
    )

    hidden_state_m, merged_embeddings_counts, merged_attention_mask, merged_special_embeddings_mask, merged_concrete = outputs
    hidden_state_m = hidden_state_m.to(hidden_state_dtype)

    return hidden_state_m, merged_embeddings_counts, merged_attention_mask, merged_special_embeddings_mask, merged_concrete


def _backward_prune_tokens_concrete(ctx, grad_hidden_state, grad_merged_embeddings_counts, grad_merged_attention_mask, grad_merged_special_embeddings_mask, grad_merged_concrete):
    # Unpack saved tensors
    concrete_bool, attention_mask = ctx.saved_tensors
    
    # Initialize gradients
    grad_hidden_state_output = None
    grad_concrete_output = None
    
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[4]:  # If we need gradients for hidden_state or concrete
        grad_hidden_state_output, grad_concrete_output = torch.ops.generate_merges.backward_prune_tokens_concrete_cuda.default(
            grad_hidden_state,
            grad_merged_concrete,
            concrete_bool,
            attention_mask,
        )
    
    # Return gradients for all inputs in order (None for those that don't need gradients)
    return grad_hidden_state_output, None, None, None, grad_concrete_output

def _setup_context_prune_tokens_concrete(ctx, inputs, output):
    hidden_state, concrete_bool, attention_mask, special_embeddings_mask, concrete = inputs
    
    # Save tensors needed for backward
    ctx.save_for_backward(concrete_bool, attention_mask)

# Register the autograd function
torch.library.register_autograd(
    "generate_merges::prune_tokens_concrete_cuda", 
    _backward_prune_tokens_concrete, 
    setup_context=_setup_context_prune_tokens_concrete
)


