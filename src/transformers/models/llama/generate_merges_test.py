import torch
from transformers.models.llama.merges_transform.generate_merges import generate_merges_transform

# # Example inputs (these should be tensors with appropriate shapes)
merging_map = torch.zeros(4, 10, 2, dtype=torch.long, device='cuda')  # [bs, seq_len, 2]
attention_mask = torch.ones(4, 10, device='cuda', dtype=torch.bool)  # [bs, seq_len]

# # Prepare output tensors
aggregated_embeddings_transform = torch.zeros(4, 10, 10, device='cuda')  # [bs, new_seq_len, seq_len]
merged_embeddings_counts = torch.zeros(4, 10, device='cuda', dtype=torch.int64)  # [bs, new_seq_len]
merged_attention_mask = torch.zeros(4, 10, device='cuda', dtype=torch.bool)  # [bs, new_seq_len]

# # Call the CUDA extension function
aggregated_embeddings_transform, merged_embeddings_counts, merged_attention_mask = generate_merges_transform( merging_map, attention_mask )

print("aggregated_embeddings_transform", aggregated_embeddings_transform)
print("merged_embeddings_counts", merged_embeddings_counts)
print("merged_attention_mask", merged_attention_mask)
