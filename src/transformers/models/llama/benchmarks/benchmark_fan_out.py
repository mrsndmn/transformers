import time
import torch
from transformers.models.llama.merges_transform.generate_merges import fan_out_restore_residuals, prune_tokens_concrete

if __name__ == "__main__":

    device = 'cuda'
    merged_embeddings_counts = torch.ones([ 16, 2048 ], device=device, dtype=torch.long)
    hidden_states = torch.rand([16, 2048, 1024], device=device)
    hidden_states_clone = hidden_states.clone()
    residual_hidden_states_projection = torch.zeros_like(hidden_states)
    attention_mask = torch.ones([ hidden_states.shape[0], hidden_states.shape[1] ], device=device, dtype=torch.long)

    restored_hidden_states = fan_out_restore_residuals(merged_embeddings_counts, hidden_states, residual_hidden_states_projection, attention_mask)

    torch.cuda.synchronize()
    num_trials = 500
    start_time = time.time()
    for _ in range(num_trials):
        restored_hidden_states = fan_out_restore_residuals(merged_embeddings_counts, hidden_states, residual_hidden_states_projection, attention_mask)
        torch.cuda.synchronize()

    print(f"Time taken: {(time.time() - start_time) / num_trials} seconds")

    torch.cuda.synchronize()

    assert (restored_hidden_states == hidden_states_clone).all()



