
import torch
from transformers.models.llama.merges_transform.generate_merges import generate_merges_transform, fan_out_restore_residuals, prune_tokens_concrete

def test_fan_out_restore_residuals():

    device = 'cuda'
    merged_embeddings_counts = torch.ones([ 3, 5 ], device=device, dtype=torch.long)
    hidden_states = torch.rand([3, 5, 960], device=device)
    hidden_states_clone = hidden_states.clone()
    residual_hidden_states_projection = torch.zeros_like(hidden_states)

    restored_hidden_states = fan_out_restore_residuals(merged_embeddings_counts, hidden_states, residual_hidden_states_projection)

    assert (restored_hidden_states == hidden_states_clone).all()
