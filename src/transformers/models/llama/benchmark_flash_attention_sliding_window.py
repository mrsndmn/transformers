# ## How to use FlashAttention

# The main functions implement scaled dot product attention (softmax(Q @ K^T *
# softmax_scale) @ V):
# ```python
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
# ```

# ```python
# flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
#                           window_size=(-1, -1), alibi_slopes=None, deterministic=False):
# """dropout_p should be set to 0.0 during evaluation
# If Q, K, V are already stacked into 1 tensor, this function will be faster than
# calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
# of the gradients of Q, K, V.
# If window_size != (-1, -1), implements sliding window local attention. Query at position i
# will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.
# Arguments:
#     qkv: (batch_size, seqlen, 3, nheads, headdim)
#     dropout_p: float. Dropout probability.
#     softmax_scale: float. The scaling of QK^T before applying softmax.
#         Default to 1 / sqrt(headdim).
#     causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
#     window_size: (left, right). If not (-1, -1), implements sliding window local attention.
#     alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
#         the attention score of query i and key j.
#     deterministic: bool. Whether to use the deterministic implementation of the backward pass,
#         which is slightly slower and uses more memory. The forward pass is always deterministic.
# Return:
#     out: (batch_size, seqlen, nheads, headdim).
# """
# ```

# ```python
# flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
#                 window_size=(-1, -1), alibi_slopes=None, deterministic=False):
# """dropout_p should be set to 0.0 during evaluation
# Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
# than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
# For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
# 0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
# If window_size != (-1, -1), implements sliding window local attention. Query at position i
# will only attend to keys between
# [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

# Arguments:
#     q: (batch_size, seqlen, nheads, headdim)
#     k: (batch_size, seqlen, nheads_k, headdim)
#     v: (batch_size, seqlen, nheads_k, headdim)
#     dropout_p: float. Dropout probability.
#     softmax_scale: float. The scaling of QK^T before applying softmax.
#         Default to 1 / sqrt(headdim).
#     causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
#     window_size: (left, right). If not (-1, -1), implements sliding window local attention.
#     alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
#         (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
#         is added to the attention score of query i and key j.
#     deterministic: bool. Whether to use the deterministic implementation of the backward pass,
#         which is slightly slower and uses more memory. The forward pass is always deterministic.
# Return:
#     out: (batch_size, seqlen, nheads, headdim).
# """
# ```

import torch
import time
from flash_attn import flash_attn_func
import numpy as np
import matplotlib.pyplot as plt

def generate_random_tensors(batch_size, seq_len, hidden_dim, num_heads, device="cuda"):
    """Generate random query, key, value tensors for testing."""
    head_dim = hidden_dim // num_heads
    shape = (batch_size, seq_len, num_heads, head_dim)
    
    q = torch.randn(shape, device=device, dtype=torch.float16)
    k = torch.randn(shape, device=device, dtype=torch.float16)
    v = torch.randn(shape, device=device, dtype=torch.float16)
    
    return q, k, v

def benchmark_flash_attention(batch_size, seq_len, hidden_dim, num_heads, num_trials=10, sliding_window=None):
    """Benchmark flash attention with optional sliding window."""
    device = "cuda"
    q, k, v = generate_random_tensors(batch_size, seq_len, hidden_dim, num_heads, device)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = flash_attn_func(q, k, v, causal=True)
            if sliding_window is not None:
                _ = flash_attn_func(q, k, v, causal=True, window_size=(sliding_window, 0))
    
    # Benchmark standard causal flash attention
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_trials):
        with torch.no_grad():
            _ = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    standard_time = (time.time() - start_time) / num_trials
    
    # Benchmark sliding window flash attention if window size provided
    sliding_time = None
    if sliding_window is not None:
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_trials):
            with torch.no_grad():
                _ = flash_attn_func(q, k, v, causal=True, window_size=(sliding_window, 0))
        torch.cuda.synchronize()
        sliding_time = (time.time() - start_time) / num_trials
    
    return standard_time, sliding_time

def run_benchmarks():
    """Run benchmarks for different sequence lengths and window sizes."""
    batch_size = 64
    hidden_dim = 2048
    num_heads = 16
    sequence_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    window_sizes = [16, 32, 64, 128, 256]
    
    # Dictionary to store results for each window size
    results = {'standard': []}
    for window_size in window_sizes:
        results[f'window_{window_size}'] = []
    
    print("\nRunning benchmarks...")
    print(f"{'Seq Length':<12}", end='')
    print(f"{'Standard (ms)':<15}", end='')
    for window_size in window_sizes:
        print(f"Window {window_size} (ms)".ljust(20), end='')
    print()
    print("-" * (12 + 15 + 20 * len(window_sizes)))
    
    for seq_len in sequence_lengths:
        times = []
        # Get standard time first
        standard_time, _ = benchmark_flash_attention(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            sliding_window=None
        )
        results['standard'].append(standard_time * 1000)
        
        print(f"{seq_len:<12}{standard_time * 1000:>8.2f}ms    ", end='')
        
        # Benchmark each window size
        for window_size in window_sizes:
            _, sliding_time = benchmark_flash_attention(
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                sliding_window=window_size
            )
            results[f'window_{window_size}'].append(sliding_time * 1000)
            print(f"{sliding_time * 1000:>8.2f}ms         ", end='')
        print()
    
    # Create line plot
    plt.figure(figsize=(12, 8))
    
    # Plot standard flash attention
    plt.plot(sequence_lengths, results['standard'], marker='o', label='Standard Flash Attention', linewidth=2)
    
    # Plot each window size
    for window_size in window_sizes:
        plt.plot(sequence_lengths, results[f'window_{window_size}'], 
                marker='o', label=f'Window Size = {window_size}', linewidth=2)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Flash Attention Performance Comparison')
    plt.xscale('log', base=2)  # Use log scale for better visualization
    plt.yscale('log')  # Use log scale for better visualization
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig('flash_attention_benchmark.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    run_benchmarks()


