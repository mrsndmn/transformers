import torch
from torch import Tensor

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

    # TODO crop length
    aggregated_embeddings_transform = torch.zeros([batch_size, seq_len, seq_len], device=device)
    
    merged_embeddings_counts = torch.zeros([batch_size, seq_len], dtype=torch.long, device=device)
    merged_attention_mask = torch.zeros([batch_size, seq_len], dtype=torch.bool, device=device)

    torch.ops.generate_merges.generate_merges_transform.default(
        merging_map,
        attention_mask,
        # outputs tensors
        aggregated_embeddings_transform,
        merged_embeddings_counts,
        merged_attention_mask
    )
    
    max_new_seq_len = merged_attention_mask.sum(dim=-1).max()
    if max_new_seq_len < merged_attention_mask.shape[1]:
        aggregated_embeddings_transform = aggregated_embeddings_transform[:, :max_new_seq_len]
        merged_embeddings_counts = merged_embeddings_counts[:, :max_new_seq_len]
        merged_attention_mask = merged_attention_mask[:, :max_new_seq_len]
    
    return aggregated_embeddings_transform, merged_embeddings_counts, merged_attention_mask

