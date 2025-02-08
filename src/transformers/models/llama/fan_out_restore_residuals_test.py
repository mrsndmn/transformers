
import torch
from transformers.models.llama.merges_transform.generate_merges import generate_merges_transform, fan_out_restore_residuals, prune_tokens_concrete

def test_fan_out_restore_residuals():

    device = 'cuda'
    merged_embeddings_counts = torch.ones([ 3, 5 ], device=device, dtype=torch.long)
    hidden_states = torch.rand([3, 5, 960], device=device)
    hidden_states_clone = hidden_states.clone()
    residual_hidden_states_projection = torch.zeros_like(hidden_states)
    attention_mask = torch.ones([ hidden_states.shape[0], hidden_states.shape[1] ], device=device, dtype=torch.long)

    restored_hidden_states = fan_out_restore_residuals(merged_embeddings_counts, hidden_states, residual_hidden_states_projection, attention_mask)

    assert (restored_hidden_states == hidden_states_clone).all()


def test_fan_out_restore_residuals_with_merging_map():

    device = 'cuda'

    merging_map = torch.tensor([ [ [ 0, 1 ], [ 0, 1 ], [ 1, 0 ], [ 0, 1 ] ] ], dtype=torch.bfloat16, device=device)
    attention_mask = torch.ones([ 1, merging_map.shape[1] ], device=device, dtype=torch.bool)
    special_embeddings_mask = torch.zeros([ 1, merging_map.shape[1] ], device=device, dtype=torch.bool)
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    aggregated_embeddings_transform_output, merged_embeddings_counts, merged_attention_mask = generate_merges_transform(merging_map, attention_mask, special_embeddings_mask)

    assert (merged_embeddings_counts == torch.tensor([ [ 1, 1, 2 ] ], dtype=torch.long, device=device)).all()

    hidden_states = torch.rand([1, merging_map[0, :, 1].sum().long().item(), 64], device=device, requires_grad=True)
    hidden_states_clone = hidden_states.clone()
    residual_hidden_states_projection = torch.rand([1, merging_map.shape[1], 64], device=device, requires_grad=True)

    restored_hidden_states = fan_out_restore_residuals(merged_embeddings_counts, hidden_states, residual_hidden_states_projection, attention_mask)

    assert (restored_hidden_states[:, :2] == hidden_states_clone[:, :2]).all()
    assert (restored_hidden_states[:, 3:] == hidden_states_clone[:, 2:]).all()

    assert (restored_hidden_states[:, 2] == residual_hidden_states_projection[:, 2]).all()

    grad_init = torch.rand_like(restored_hidden_states)

    restored_hidden_states.backward(grad_init)

    assert (grad_init[:, 2, :] == residual_hidden_states_projection.grad[:, 2, :]).all()
    assert (grad_init[:, :2, :] == hidden_states.grad[:, :2, :]).all()
    assert (grad_init[:, 3, :] == hidden_states.grad[:, 2, :]).all()

    breakpoint()


def test_skip_last_tokens():
    device = 'cuda'

    merging_map = torch.tensor([ [ [ 0, 1 ], [ 0, 1 ], [ 1, 0 ], [ 1, 0 ] ] ], dtype=torch.bfloat16, device=device)
    attention_mask = torch.ones([ 1, merging_map.shape[1] ], device=device, dtype=torch.bool)
    special_embeddings_mask = torch.zeros([ 1, merging_map.shape[1] ], device=device, dtype=torch.bool)
    special_embeddings_mask[:, 0] = 1

    aggregated_embeddings_transform_output, merged_embeddings_counts, merged_attention_mask = generate_merges_transform(merging_map, attention_mask, special_embeddings_mask)

    assert (merged_embeddings_counts == torch.tensor([ [ 1, 1 ] ], dtype=torch.long, device=device)).all()

    hidden_states = torch.rand([1, merging_map[0, :, 1].sum().long().item(), 64], device=device, requires_grad=True)
    hidden_states_clone = hidden_states.clone()
    residual_hidden_states_projection = torch.rand([1, merging_map.shape[1], 64], device=device, requires_grad=True)

    restored_hidden_states = fan_out_restore_residuals(merged_embeddings_counts, hidden_states, residual_hidden_states_projection, attention_mask)

    assert (restored_hidden_states[:, :2] == hidden_states_clone[:, :2]).all()
    assert (restored_hidden_states[:, 2:] == residual_hidden_states_projection[:, 2:]).all()

    grad_init = torch.rand_like(restored_hidden_states)

    restored_hidden_states.backward(grad_init)

    assert (grad_init[:, :2, :] == hidden_states.grad[:, :2, :]).all()
    assert (grad_init[:, 2:] == residual_hidden_states_projection.grad[:, 2:]).all()

    breakpoint()
