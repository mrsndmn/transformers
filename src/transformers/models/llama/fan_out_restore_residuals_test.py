
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

    assert (restored_hidden_states[:, :3] == hidden_states_clone[:, :3]).all()

    assert (restored_hidden_states[:, -1] == residual_hidden_states_projection[:, -1]).all()

    grad_init = torch.rand_like(restored_hidden_states)

    restored_hidden_states.backward(grad_init)

    assert (grad_init[:, -1, :] == residual_hidden_states_projection.grad[:, -1, :]).all()
    assert (grad_init[:, :3, :] == hidden_states.grad[:, :3, :]).all()


def test_prune_tokens_concrete_simple():

    batch_size = 1
    seq_len = 7
    hidden_dim = 64

    device = 'cuda'

    hidden_state = torch.rand([ batch_size, seq_len, hidden_dim ], device=device)
    concrete = torch.tensor([[ 1, 1, 1, 1, 1, 1, 1 ]], dtype=torch.float32, device=device)
    concrete_bool = concrete.bool()
    attention_mask = torch.ones_like(concrete_bool).long()
    special_embeddings_mask = torch.zeros_like(concrete_bool).long()
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    merged_hidden_state, merged_embeddings_counts, merged_attention_mask, merged_special_embeddings_mask, merged_concrete = prune_tokens_concrete(
        hidden_state,
        concrete_bool,
        attention_mask,
        concrete=concrete,
        special_embeddings_mask=special_embeddings_mask,
    )

    assert (hidden_state == merged_hidden_state).all()
    assert (merged_embeddings_counts == attention_mask.long()).all()
    assert (merged_attention_mask == attention_mask).all()
    assert (merged_special_embeddings_mask == special_embeddings_mask).all()
    assert (merged_concrete == concrete).all()

    return

def test_prune_tokens_concrete_middle():

    batch_size = 1
    seq_len = 7
    hidden_dim = 64

    device = 'cuda'

    hidden_state = torch.rand([ batch_size, seq_len, hidden_dim ], device=device, requires_grad=True)
    concrete_bool = torch.tensor([[ 1, 1, 1, 1, 0, 0, 1 ]], dtype=torch.bool, device=device)
    attention_mask = torch.ones_like(concrete_bool).long()
    concrete = concrete_bool.float()
    special_embeddings_mask = torch.zeros_like(concrete_bool).long()
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    merged_hidden_state, merged_embeddings_counts, merged_attention_mask, merged_special_embeddings_mask, merged_concrete = prune_tokens_concrete(
        hidden_state,
        concrete_bool,
        attention_mask,
        concrete=concrete,
        special_embeddings_mask=special_embeddings_mask,
    )

    expected_merged_emb_counts = torch.tensor([[ 1, 1, 1, 3, 1 ]], dtype=torch.long, device=device)
    expected_attention_mask = torch.ones_like(expected_merged_emb_counts)

    assert merged_hidden_state.shape[1] == 5
    assert (hidden_state[:, :4] == merged_hidden_state[:, :4]).all()
    assert (hidden_state[:, -1:] == merged_hidden_state[:, -1:]).all()

    assert (merged_embeddings_counts == expected_merged_emb_counts).all()
    assert (merged_attention_mask == expected_attention_mask).all()
    assert merged_special_embeddings_mask.sum().item() == 2
    assert merged_concrete.shape[1] == 5
    assert merged_concrete.sum().item() == 5

    return


def test_prune_tokens_concrete_end():

    batch_size = 1
    seq_len = 7
    hidden_dim = 64

    device = 'cuda'

    hidden_state = torch.rand([ batch_size, seq_len, hidden_dim ], device=device)
    concrete_bool = torch.tensor([[ 1, 1, 1, 1, 0, 0, 0 ]], dtype=torch.bool, device=device)
    attention_mask = torch.ones_like(concrete_bool).long()
    concrete = concrete_bool.float()
    special_embeddings_mask = torch.zeros_like(concrete_bool).long()
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, 3] = 1

    merged_hidden_state, merged_embeddings_counts, merged_attention_mask, merged_special_embeddings_mask, merged_concrete = prune_tokens_concrete(
        hidden_state,
        concrete_bool,
        attention_mask,
        concrete=concrete,
        special_embeddings_mask=special_embeddings_mask,
    )


    expected_merged_emb_counts = torch.tensor([[ 1, 1, 1, 4 ]], dtype=torch.long, device=device)
    expected_attention_mask = torch.ones_like(expected_merged_emb_counts)

    assert merged_hidden_state.shape[1] == 4
    assert (hidden_state[:, :4] == merged_hidden_state[:, :4]).all()

    assert (merged_embeddings_counts == expected_merged_emb_counts).all()
    assert (merged_attention_mask == expected_attention_mask).all()

    assert merged_special_embeddings_mask.sum().item() == 2
    assert merged_concrete.shape[1] == 4
    assert merged_concrete.sum().item() == 4

    return

def test_prune_tokens_concrete_backward_basic():
    batch_size = 1
    seq_len = 7
    hidden_dim = 64

    device = 'cuda'

    hidden_state = torch.rand([batch_size, seq_len, hidden_dim], device=device, requires_grad=True)
    concrete = torch.ones([batch_size, seq_len], dtype=torch.float32, device=device, requires_grad=True)
    concrete_bool = concrete.bool()
    attention_mask = torch.ones_like(concrete_bool).long()
    special_embeddings_mask = torch.zeros_like(concrete_bool).long()

    merged_hidden_state, merged_embeddings_counts, merged_attention_mask, merged_special_embeddings_mask, merged_concrete = prune_tokens_concrete(
        hidden_state,
        concrete_bool,
        attention_mask,
        special_embeddings_mask=special_embeddings_mask,
        concrete=concrete,
    )

    # Verify shapes
    assert merged_hidden_state.shape == hidden_state.shape
    assert merged_concrete.shape == concrete.shape

    # Test gradient flow
    grad_output = torch.rand_like(merged_hidden_state)
    grad_concrete_output = torch.rand_like(merged_concrete)

    loss = (merged_hidden_state * grad_output).sum() + (merged_concrete * grad_concrete_output).sum()
    loss.backward()

    # Verify gradients exist
    assert hidden_state.grad is not None
    assert concrete.grad is not None
    assert hidden_state.grad.shape == hidden_state.shape
    assert concrete.grad.shape == concrete.shape

    assert (hidden_state.grad == grad_output).all()
    assert (concrete.grad == grad_concrete_output).all()


def test_prune_tokens_concrete_backward_with_masking():
    batch_size = 1
    seq_len = 7
    hidden_dim = 64

    device = 'cuda'

    hidden_state = torch.rand([batch_size, seq_len, hidden_dim], device=device, requires_grad=True)
    concrete = torch.ones([batch_size, seq_len], dtype=torch.float32, device=device, requires_grad=True)
    concrete_bool = torch.tensor([[1, 1, 1, 1, 0, 0, 1]], dtype=torch.bool, device=device)
    attention_mask = torch.ones_like(concrete_bool).long()
    special_embeddings_mask = torch.zeros_like(concrete_bool).long()
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    merged_hidden_state, merged_embeddings_counts, merged_attention_mask, merged_special_embeddings_mask, merged_concrete = prune_tokens_concrete(
        hidden_state,
        concrete_bool,
        attention_mask,
        special_embeddings_mask=special_embeddings_mask,
        concrete=concrete,
    )

    # Verify expected shapes after pruning
    assert merged_hidden_state.shape[1] == 5  # Should have 5 tokens after pruning
    assert merged_concrete.shape[1] == 5

    # Test gradient flow
    grad_output = torch.rand_like(merged_hidden_state)
    grad_concrete_output = torch.rand_like(merged_concrete)
    
    loss = (merged_hidden_state * grad_output).sum() + (merged_concrete * grad_concrete_output).sum()
    loss.backward()

    # Verify gradients exist and have correct shapes
    assert hidden_state.grad is not None
    assert concrete.grad is not None
    assert hidden_state.grad.shape == hidden_state.shape
    assert concrete.grad.shape == concrete.shape

    # Verify gradient flow for special tokens
    assert hidden_state.grad[:, 0].abs().sum() > 0  # First token (special) should have gradient
    assert hidden_state.grad[:, -1].abs().sum() > 0  # Last token (special) should have gradient

    assert hidden_state.grad[:, 4:6].sum() == 0, 'no grads for not concrete hidden states'

    assert (hidden_state.grad[:, :4] == grad_output[:, :4]).all()
    assert (hidden_state.grad[:, 6] == grad_output[:, 4]).all()

    assert (concrete.grad[:, :4] == grad_concrete_output[:, :4]).all()
    assert (concrete.grad[:, 6] == grad_concrete_output[:, 4]).all()

def test_prune_tokens_concrete_backward_attention_mask():
    batch_size = 1
    seq_len = 7
    hidden_dim = 64

    device = 'cuda'

    hidden_state = torch.rand([batch_size, seq_len, hidden_dim], device=device, requires_grad=True)
    concrete = torch.ones([batch_size, seq_len], dtype=torch.float32, device=device, requires_grad=True)
    concrete_bool = torch.tensor([[0, 0, 1, 1, 0, 0, 0]], dtype=torch.bool, device=device)
    attention_mask = torch.ones([batch_size, seq_len], device=device, dtype=torch.long)
    attention_mask[:, :2] = 0  # Mask out first 2 tokens
    special_embeddings_mask = torch.zeros_like(concrete_bool).long()
    special_embeddings_mask[:, 2] = 1
    special_embeddings_mask[:, 3] = 1

    merged_hidden_state, merged_embeddings_counts, merged_attention_mask, merged_special_embeddings_mask, merged_concrete = prune_tokens_concrete(
        hidden_state,
        concrete_bool,
        attention_mask,
        special_embeddings_mask=special_embeddings_mask,
        concrete=concrete,
    )

    # Verify expected shapes after pruning
    assert merged_hidden_state.shape[1] == 2  # Should have 2 tokens after pruning
    assert merged_concrete.shape[1] == 2

    # Test gradient flow
    grad_output = torch.rand_like(merged_hidden_state)
    grad_concrete_output = torch.rand_like(merged_concrete)

    loss = (merged_hidden_state * grad_output).sum() + (merged_concrete * grad_concrete_output).sum()
    loss.backward()

    # Verify gradients exist and have correct shapes
    assert hidden_state.grad is not None
    assert concrete.grad is not None
    assert hidden_state.grad.shape == hidden_state.shape
    assert concrete.grad.shape == concrete.shape

    # Verify no gradients for masked tokens
    assert (hidden_state.grad[:, 5:] == 0).all()  # Masked tokens should have zero gradient
    assert (concrete.grad[:, 5:] == 0).all()  # Masked tokens should have zero gradient

    assert (hidden_state.grad[:, 2:4] == grad_output).all()

    assert (concrete.grad[:, 2:4] == grad_concrete_output).all()


def test_prune_tokens_concrete_backward_dtype_consistency():
    batch_size = 1
    seq_len = 7
    hidden_dim = 64

    device = 'cuda'
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    for dtype in dtypes:
        hidden_state = torch.rand([batch_size, seq_len, hidden_dim], device=device, dtype=dtype, requires_grad=True)
        concrete = torch.ones([batch_size, seq_len], dtype=torch.float32, device=device, requires_grad=True)
        concrete_bool = concrete.bool()
        attention_mask = torch.ones_like(concrete_bool).long()
        special_embeddings_mask = torch.zeros_like(concrete_bool).long()

        merged_hidden_state, merged_embeddings_counts, merged_attention_mask, merged_special_embeddings_mask, merged_concrete = prune_tokens_concrete(
            hidden_state,
            concrete_bool,
            attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            concrete=concrete,
        )

        # Verify output dtype matches input
        assert merged_hidden_state.dtype == dtype

        # Test gradient flow
        grad_output = torch.rand_like(merged_hidden_state)
        grad_concrete_output = torch.rand_like(merged_concrete)
        
        loss = (merged_hidden_state * grad_output).sum() + (merged_concrete * grad_concrete_output).sum()
        loss.backward()

        # Verify gradients exist and have correct dtype
        assert hidden_state.grad is not None
        assert concrete.grad is not None
        assert hidden_state.grad.dtype == dtype
        assert concrete.grad.dtype == torch.float32  # Concrete always uses float32

