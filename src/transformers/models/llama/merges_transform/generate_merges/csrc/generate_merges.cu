#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <iostream>
using namespace std;

void check_cuda_errors() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

__global__ void generate_merges_transform_kernel(
    const torch::PackedTensorAccessor64<float, 3> merging_map,
    const torch::PackedTensorAccessor64<bool, 2> attention_mask,
    const torch::PackedTensorAccessor64<bool, 2> special_embeddings_mask,
    torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_counts,
    torch::PackedTensorAccessor64<bool, 2> merged_attention_mask,
    torch::PackedTensorAccessor64<float, 3> aggregated_embeddings_transform,
    int batch_size, int seq_len
) {
    int batch_i = blockIdx.x; // Batch index
    // int seq_len_i = threadIdx.x; // Index for the new sequence length

    if (batch_i >= batch_size) {
        return; // Out of bounds check
    }

    int new_seq_len_i = 0;
    for (int seq_len_i = 0; seq_len_i < seq_len; ++seq_len_i) {
        if (!attention_mask[batch_i][seq_len_i]) {
            break;
        }

        bool is_token_important = merging_map[batch_i][seq_len_i][1] > merging_map[batch_i][seq_len_i][0];

        if (is_token_important) {
            merged_embeddings_counts[batch_i][new_seq_len_i] += 1;
            aggregated_embeddings_transform[batch_i][new_seq_len_i][seq_len_i] = 1.0;
            merged_attention_mask[batch_i][new_seq_len_i] = true;
            new_seq_len_i += 1;
        } else {
            merged_embeddings_counts[batch_i][new_seq_len_i] += 1;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> generate_merges_transform_cuda(
    const torch::Tensor& merging_map,
    const torch::Tensor& attention_mask,
    const torch::Tensor& special_embeddings_mask
) {
    const int batch_size = merging_map.size(0);
    const int seq_len = merging_map.size(1);

    // Launch the kernel
    const dim3 block_size(1, 1, 1);  // One thread per sequence element
    const dim3 grid_size(batch_size, 1, 1);   // One block per batch element

    auto options = merging_map.options();
    auto device = options.device();

    auto aggregated_embeddings_transform_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto merged_embeddings_counts_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto mask_options = torch::TensorOptions().dtype(torch::kBool).device(device);

    torch::Tensor aggregated_embeddings_transform = torch::zeros({batch_size, seq_len, seq_len}, aggregated_embeddings_transform_options);
    torch::Tensor merged_embeddings_counts = torch::zeros({batch_size, seq_len}, merged_embeddings_counts_options);
    torch::Tensor merged_attention_mask = torch::zeros({batch_size, seq_len}, mask_options);

    generate_merges_transform_kernel<<<grid_size, block_size>>>(
        merging_map.packed_accessor64<float, 3>(),
        attention_mask.packed_accessor64<bool, 2>(),
        special_embeddings_mask.packed_accessor64<bool, 2>(),
        merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
        merged_attention_mask.packed_accessor64<bool, 2>(),
        aggregated_embeddings_transform.packed_accessor64<float, 3>(),
        batch_size, seq_len
    );

    // Error checking
    // cudaDeviceSynchronize();
    // check_cuda_errors();

    return std::make_tuple(aggregated_embeddings_transform, merged_embeddings_counts, merged_attention_mask);
}


__global__ void batch_repeat_interleave_for_merges_count_kernel(
    const torch::PackedTensorAccessor64<float, 3> grad_merging_map,
    const torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_counts,
    torch::PackedTensorAccessor64<float, 3> merging_map_output,
    int batch_size, int seq_len
) {
    int batch_i = blockIdx.x; // Batch index
    // int seq_len_i = threadIdx.x; // Index for the new sequence length

    if (batch_i >= batch_size) {
        return; // Out of bounds check
    }

    int output_seq_len_i = 0;
    for (int seq_len_i = 0; seq_len_i < seq_len; ++seq_len_i) {
        auto current_repeats_num = merged_embeddings_counts[batch_i][seq_len_i];
        if (current_repeats_num == 0) {
            break;
        }

        output_seq_len_i = output_seq_len_i + current_repeats_num - 1;
        merging_map_output[batch_i][output_seq_len_i][0] = grad_merging_map[batch_i][seq_len_i][0];
        merging_map_output[batch_i][output_seq_len_i][1] = grad_merging_map[batch_i][seq_len_i][1];
        ++output_seq_len_i;
    }
}

torch::Tensor batch_repeat_interleave_for_merges_count(
    const torch::Tensor& grad_merging_map,
    const torch::Tensor& merged_embeddings_counts
) {
    const int batch_size = merged_embeddings_counts.size(0);
    const int seq_len = merged_embeddings_counts.size(1);

    // Launch the kernel
    const dim3 block_size(1, 1, 1);  // One thread per sequence element
    const dim3 grid_size(batch_size, 1, 1);   // One block per batch element

    torch::Tensor grad_merging_map_output = torch::zeros_like(grad_merging_map, grad_merging_map.options());

    batch_repeat_interleave_for_merges_count_kernel<<<grid_size, block_size>>>(
        grad_merging_map.packed_accessor64<float, 3>(),
        merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
        grad_merging_map_output.packed_accessor64<float, 3>(),
        batch_size, seq_len
    );

    // Error checking
    // cudaDeviceSynchronize();
    // check_cuda_errors();

    return grad_merging_map_output;
}

const int HIDDEN_DIM_BLOCK_SIZE_FAN_OUT = 64;

__global__ void fan_out_restore_residuals_kernel(
    const torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_counts,
    const torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_cumsum_flipped,
    const torch::PackedTensorAccessor64<float, 3> hidden_states,
    torch::PackedTensorAccessor64<float, 3> restored_hidden_states,
    int batch_size, int seq_len, int residual_seq_len, int hidden_dim
) {
    int batch_i = blockIdx.x; // Batch index
    int seq_len_i = blockIdx.y; // Sequence index
    int thread_idx = threadIdx.x;
    
    // Each thread gets its own portion of shared memory
    extern __shared__ float shared_hidden_states[];
    
    if (batch_i >= batch_size || seq_len_i >= seq_len) {
        return; // Out of bounds check
    }
    
    int hidden_dim_start = thread_idx * HIDDEN_DIM_BLOCK_SIZE_FAN_OUT;
    if (hidden_dim_start >= hidden_dim) {
        return;
    }

    int hidden_dim_end = hidden_dim_start + HIDDEN_DIM_BLOCK_SIZE_FAN_OUT;
    if (hidden_dim_end > hidden_dim) {
        hidden_dim_end = hidden_dim;
    }

    // Each thread uses its own portion of shared memory
    float* thread_shared_mem = &shared_hidden_states[thread_idx * HIDDEN_DIM_BLOCK_SIZE_FAN_OUT];

    auto restored_idx = residual_seq_len - merged_embeddings_cumsum_flipped[batch_i][seq_len_i];
    auto num_repeats = merged_embeddings_counts[batch_i][seq_len_i];
    if (num_repeats == 0) {
        return;
    }

    __syncthreads();
    // Load hidden states into thread's portion of shared memory
    for (int hi = hidden_dim_start; hi < hidden_dim_end; ++hi) {
        thread_shared_mem[hi - hidden_dim_start] = hidden_states[batch_i][seq_len_i][hi];
    }

    __syncthreads();
    // Write from thread's portion of shared memory to global memory
    for (int hi = hidden_dim_start; hi < hidden_dim_end; ++hi) {
        restored_hidden_states[batch_i][restored_idx][hi] += thread_shared_mem[hi - hidden_dim_start];
    }
}

torch::Tensor fan_out_restore_residuals(
    const torch::Tensor& merged_embeddings_counts,
    const torch::Tensor& hidden_states,
    const torch::Tensor& residual_hidden_states_projection,
    const torch::Tensor& residual_hidden_states_attention_mask
) {
    const int batch_size = merged_embeddings_counts.size(0);
    const int seq_len = merged_embeddings_counts.size(1);
    const int residual_seq_len = residual_hidden_states_projection.size(1);
    const int hidden_dim = residual_hidden_states_projection.size(2);
    
    // Launch the kernel
    int num_threads = (hidden_dim + HIDDEN_DIM_BLOCK_SIZE_FAN_OUT - 1) / HIDDEN_DIM_BLOCK_SIZE_FAN_OUT;
    
    const dim3 block_size(num_threads, 1, 1);  // One thread per sequence element
    const dim3 grid_size(batch_size, seq_len, 1);   // One block per batch element
    
    // Calculate shared memory size (HIDDEN_DIM_BLOCK_SIZE_FAN_OUT floats per thread)
    size_t shared_mem_size = num_threads * HIDDEN_DIM_BLOCK_SIZE_FAN_OUT * sizeof(float);
    
    torch::Tensor restored_hidden_states = torch::clone(residual_hidden_states_projection);
    
    torch::Tensor merged_embeddings_counts_flipped = torch::flip(merged_embeddings_counts, {1});
    torch::Tensor merged_embeddings_cumsum = torch::cumsum(merged_embeddings_counts_flipped, 1);
    torch::Tensor merged_embeddings_cumsum_flipped = torch::flip(merged_embeddings_cumsum, {1});
    
    fan_out_restore_residuals_kernel<<<grid_size, block_size, shared_mem_size>>>(
        merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
        merged_embeddings_cumsum_flipped.packed_accessor64<int64_t, 2>(),
        hidden_states.packed_accessor64<float, 3>(),
        restored_hidden_states.packed_accessor64<float, 3>(),
        batch_size, seq_len, residual_seq_len, hidden_dim
    );
    
    // Error checking
    // cudaDeviceSynchronize();
    // check_cuda_errors();
    
    return restored_hidden_states;
}


__global__ void backward_fan_out_straight_hidden_states_kernel(
    const torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_counts,
    const torch::PackedTensorAccessor64<float, 3> restored_hidden_states_grad,
    const torch::PackedTensorAccessor64<int64_t, 1> restored_hidden_states_seq_lengths,
    torch::PackedTensorAccessor64<float, 3> hidden_states_grad,
    int batch_size, int seq_len, int hidden_dim
) {
    int batch_i = blockIdx.x; // Batch index
    int thread_idx = threadIdx.x; // Index for hidden dim

    if (batch_i >= batch_size) {
        return; // Out of bounds check
    }

    int hidden_dim_start = thread_idx * HIDDEN_DIM_BLOCK_SIZE_FAN_OUT;
    if (hidden_dim_start >= hidden_dim) {
        return;
    }

    int hidden_dim_end = hidden_dim_start + HIDDEN_DIM_BLOCK_SIZE_FAN_OUT;
    if (hidden_dim_end > hidden_dim) {
        hidden_dim_end = hidden_dim;
    }

    int output_seq_len_i = restored_hidden_states_grad.size(1) - 1;
    for (int seq_len_i = seq_len-1; seq_len_i >= 0; --seq_len_i) {
        auto num_repeats = merged_embeddings_counts[batch_i][seq_len_i];
        if (num_repeats == 0) {
            break;
        }

        // for hidden_states_grad
        int restored_idx = int(output_seq_len_i - num_repeats + 1);
        for (int hi = hidden_dim_start; hi < hidden_dim_end; ++hi) {
            hidden_states_grad[batch_i][seq_len_i][hi] = restored_hidden_states_grad[batch_i][restored_idx][hi];
        }

        output_seq_len_i -= num_repeats;
    }

}

__global__ void backward_fan_out_straight_residual_hidden_states_kernel(
    const torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_counts,
    const torch::PackedTensorAccessor64<float, 3> restored_hidden_states_grad,
    const torch::PackedTensorAccessor64<int64_t, 1> restored_hidden_states_seq_lengths,
    torch::PackedTensorAccessor64<float, 3> residual_hidden_states_grad,
    int batch_size, int seq_len, int hidden_dim
) {
    int batch_i = blockIdx.x; // Batch index
    int thread_idx = threadIdx.x; // Index for hidden dim

    if (batch_i >= batch_size) {
        return; // Out of bounds check
    }

    int hidden_dim_start = thread_idx * HIDDEN_DIM_BLOCK_SIZE_FAN_OUT;
    if (hidden_dim_start >= hidden_dim) {
        return;
    }

    int hidden_dim_end = hidden_dim_start + HIDDEN_DIM_BLOCK_SIZE_FAN_OUT;
    if (hidden_dim_end > hidden_dim) {
        hidden_dim_end = hidden_dim;
    }

    int output_seq_len_i = restored_hidden_states_grad.size(1) - 1;
    for (int seq_len_i = seq_len-1; seq_len_i >= 0; --seq_len_i) {
        auto num_repeats = merged_embeddings_counts[batch_i][seq_len_i];
        if (num_repeats == 0) {
            break;
        }

        // for residual_hidden_states_grad
        for (int residuals_grad_i = 0; residuals_grad_i < num_repeats - 1; ++residuals_grad_i) {
            int restore_idx = output_seq_len_i - residuals_grad_i;
            for (int hi = 0; hi < hidden_dim; ++hi) {
                residual_hidden_states_grad[batch_i][restore_idx][hi] = restored_hidden_states_grad[batch_i][restore_idx][hi];
            }
        }

        output_seq_len_i -= num_repeats;
    }

}


std::tuple<torch::Tensor, torch::Tensor> backward_fan_out_restore_residuals(
    const torch::Tensor& merged_embeddings_counts,
    const torch::Tensor& restored_hidden_states_grad,
    const torch::Tensor& restored_hidden_states_seq_lengths // [ bs ]
) {
    const int batch_size = merged_embeddings_counts.size(0);
    const int seq_len = merged_embeddings_counts.size(1);
    const int hidden_dim = restored_hidden_states_grad.size(2);

    int num_threads = (hidden_dim + HIDDEN_DIM_BLOCK_SIZE_FAN_OUT - 1) / HIDDEN_DIM_BLOCK_SIZE_FAN_OUT;

    // Launch the kernel
    const dim3 block_size(num_threads, 1, 1);  // One thread per sequence element
    const dim3 grid_size(batch_size, 1, 1);   // One block per batch element

    torch::Tensor hidden_states_grad = torch::zeros({batch_size, seq_len, hidden_dim}, restored_hidden_states_grad.options());
    torch::Tensor residual_hidden_states_grad = torch::zeros_like(restored_hidden_states_grad, restored_hidden_states_grad.options());

    backward_fan_out_straight_hidden_states_kernel<<<grid_size, block_size>>>(
        merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
        restored_hidden_states_grad.packed_accessor64<float, 3>(),
        restored_hidden_states_seq_lengths.packed_accessor64<int64_t, 1>(),
        hidden_states_grad.packed_accessor64<float, 3>(),
        batch_size, seq_len, hidden_dim
    );

    backward_fan_out_straight_residual_hidden_states_kernel<<<grid_size, block_size>>>(
        merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
        restored_hidden_states_grad.packed_accessor64<float, 3>(),
        restored_hidden_states_seq_lengths.packed_accessor64<int64_t, 1>(),
        residual_hidden_states_grad.packed_accessor64<float, 3>(),
        batch_size, seq_len, hidden_dim
    );

    // Error checking
    // cudaDeviceSynchronize();
    // check_cuda_errors();

    return std::make_tuple(hidden_states_grad, residual_hidden_states_grad);
}

const int HIDDEN_DIM_BLOCK_SIZE = 16;
// const int SEQ_LEN_BLOCK_SIZE = 64;
// avoid splitting block size!
// todo better optimize memory access in kernel
const int SEQ_LEN_BLOCK_SIZE = 100024;

__global__ void prune_tokens_concrete_kernel(
    const torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits> hidden_state,
    const torch::PackedTensorAccessor64<bool, 2, torch::RestrictPtrTraits> concrete_bool,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> attention_mask,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> special_embeddings_mask,
    const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> concrete,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> merged_embeddings_counts,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> merged_attention_mask,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> merged_special_embeddings_mask,
    torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> merged_concrete,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> seq_len_blocks_lengths,
    int batch_size, int seq_len, int hidden_dim
) {
    int batch_i = blockIdx.x; // Batch index

    if (batch_i >= batch_size) {
        return; // Out of bounds check
    }

    int current_seq_len = seq_len - 1;
    int current_merged_embeddings = 0;

    for (int seq_len_i = seq_len - 1; seq_len_i >= 0; --seq_len_i) {
        if (!attention_mask[batch_i][seq_len_i]) {
            break;
        }

        if (concrete_bool[batch_i][seq_len_i]) {
            merged_embeddings_counts[batch_i][current_seq_len] = current_merged_embeddings + 1;
            merged_special_embeddings_mask[batch_i][current_seq_len] = special_embeddings_mask[batch_i][seq_len_i];
            merged_concrete[batch_i][current_seq_len] = concrete[batch_i][seq_len_i];
            merged_attention_mask[batch_i][current_seq_len] = 1L;
            current_seq_len -= 1;
            current_merged_embeddings = 0;
        } else {
            current_merged_embeddings += 1;
        }
    }

    seq_len_blocks_lengths[batch_i][0] = seq_len - current_seq_len - 1;
}

__global__ void prune_tokens_concrete_copy_hidden_state_kernel(
    const torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits> hidden_state,
    const torch::PackedTensorAccessor64<bool, 2, torch::RestrictPtrTraits> concrete_bool,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> attention_mask,
    torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits> merged_hidden_state,
    int batch_size, int seq_len, int hidden_dim
) {
    int batch_i = blockIdx.x; // Batch index
    int thread_idx = threadIdx.x; // Index for hidden dim

    if (batch_i >= batch_size) {
        return; // Out of bounds check
    }

    int hidden_dim_start = thread_idx * HIDDEN_DIM_BLOCK_SIZE;
    if (hidden_dim_start >= hidden_dim) {
        return;
    }
    int hidden_dim_end = hidden_dim_start + HIDDEN_DIM_BLOCK_SIZE;
    if (hidden_dim_end > hidden_dim) {
        hidden_dim_end = hidden_dim;
    }

    int current_seq_len = seq_len - 1;
    for (int seq_len_i = seq_len - 1; seq_len_i >= 0; --seq_len_i) {
        if (attention_mask[batch_i][seq_len_i] == 0L) {
            break;
        }

        if (concrete_bool[batch_i][seq_len_i]) {
            for (int hi = hidden_dim_start; hi < hidden_dim_end; ++hi) {
                merged_hidden_state[batch_i][current_seq_len][hi] = hidden_state[batch_i][seq_len_i][hi];
            }
            current_seq_len -= 1;
        }
    }
}

__global__ void backward_prune_tokens_concrete_kernel(
    const torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits> grad_output_hidden_state,
    const torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> grad_output_concrete,
    const torch::PackedTensorAccessor64<bool, 2, torch::RestrictPtrTraits> concrete_bool,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> attention_mask,
    torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits> grad_hidden_state,
    torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> grad_concrete,
    int batch_size, int seq_len, int hidden_dim, int grad_output_seq_len
) {
    int batch_i = blockIdx.x; // Batch index
    int thread_idx = threadIdx.x; // Index for hidden dim

    if (batch_i >= batch_size) {
        return; // Out of bounds check
    }

    int hidden_dim_start = thread_idx * HIDDEN_DIM_BLOCK_SIZE;
    if (hidden_dim_start >= hidden_dim) {
        return;
    }
    int hidden_dim_end = hidden_dim_start + HIDDEN_DIM_BLOCK_SIZE;
    if (hidden_dim_end > hidden_dim) {
        hidden_dim_end = hidden_dim;
    }


    // Start from the correct position in grad_output
    int current_grad_pos = grad_output_seq_len - 1;
    
    // Iterate in reverse order to match forward pass
    for (int seq_len_i = seq_len - 1; seq_len_i >= 0; --seq_len_i) {
        if (attention_mask[batch_i][seq_len_i] == 0L) {
            break;
        }

        if (concrete_bool[batch_i][seq_len_i]) {
            if (current_grad_pos >= 0 && current_grad_pos < grad_output_seq_len) {
                // Copy gradients for hidden states
                for (int hi = hidden_dim_start; hi < hidden_dim_end; ++hi) {
                    grad_hidden_state[batch_i][seq_len_i][hi] = grad_output_hidden_state[batch_i][current_grad_pos][hi];
                }
                // Copy gradients for concrete values
                grad_concrete[batch_i][seq_len_i] = grad_output_concrete[batch_i][current_grad_pos];
            }
            current_grad_pos--;
        }
    }
}

void collapse_blocks(
    torch::Tensor seq_len_blocks_lengths,
    torch::Tensor merged_hidden_state,
    torch::Tensor merged_embeddings_counts,
    torch::Tensor merged_attention_mask
) {

    int seq_len = merged_hidden_state.size(1);
    int batch_size = merged_hidden_state.size(0);

    int seq_len_pow2 = SEQ_LEN_BLOCK_SIZE;
    int seq_len_pow2_exp = 0;
    while(seq_len_pow2 < seq_len) {
        seq_len_pow2 *= 2;
        seq_len_pow2_exp += 1;
    }

    int init_seq_len_threads_len = seq_len_blocks_lengths.size(1);

    // torch.cat [ [ i, 64, 512 ].slice, [ i, 64, 512 ].slice, [ i, 64, 512 ].slice ]

    // TODO Попробовать с перестановкой интдексов
    // [
    //   [ | 1, 1, 0, 0, 0 | 1, 1, 1, 0, 0 | ] 
    //   [ | 1, 1, 1, 0, 0 | 1, 1, 1, 1, 0 | ]
    // ]

    // [ b1, b2, b3, b4, b5, b6, b7, b8 ]
    // [ b12, b3, b4, b5, b6, b7, b8 ]
    // [ b123, b4, b5, b6, b7, b8 ]
    // [ b1234, b5, b6, b7, b8 ]

    // [ b1, b2, b3, b4, b5, b6, b7, b8 ]
    // [ b12, b34, b56, b78 ]
    // [ b1234, b5678 ]
    // [ b12345678 ]

    for (int batch_i = 0; batch_i < batch_size; ++batch_i) {
        // todo process each batch in separate cuda stream
        // cout << "seq_len_pow2_exp " << seq_len_pow2_exp << endl << flush;
        // cout << "seq_len_pow2     " << seq_len_pow2 << endl << flush;

        int seq_len_len = init_seq_len_threads_len * 2;

        for (int divide_and_conquer_i = 0; divide_and_conquer_i < seq_len_pow2_exp; ++divide_and_conquer_i) {
            int block_size = SEQ_LEN_BLOCK_SIZE * (1 << (divide_and_conquer_i));

            seq_len_len = (seq_len_len + 1) / 2;

            // cout << "divide_and_conquer_i " << divide_and_conquer_i << endl << flush;
            // cout << "block_size " << block_size << endl << flush;
            // cout << "seq_len_len " << seq_len_len << endl << flush;

            torch::Tensor merged_hidden_state_clone = torch::clone(merged_hidden_state);
            torch::Tensor merged_embeddings_counts_clone = torch::clone(merged_embeddings_counts);
            torch::Tensor merged_attention_mask_clone = torch::clone(merged_attention_mask);

            for (int seq_len_blocks_i = 0; seq_len_blocks_i < seq_len_len; seq_len_blocks_i += 2) {
                if (seq_len_blocks_i + 1 >= seq_len_len) {
                    // Neighbour block is out of range
                    // cout << "block_seq_len      "  << seq_len_blocks_lengths[batch_i][seq_len_blocks_i].item<int64_t>() << endl << flush;
                    seq_len_blocks_lengths[batch_i][int(seq_len_blocks_i / 2)] = seq_len_blocks_lengths[batch_i][seq_len_blocks_i];
                    break;
                }

                int64_t block_seq_len = seq_len_blocks_lengths[batch_i][seq_len_blocks_i].item<int64_t>();
                int64_t next_block_seq_len = seq_len_blocks_lengths[batch_i][seq_len_blocks_i+1].item<int64_t>();
                // cout << "block_seq_len      "  << block_seq_len << endl << flush;
                // cout << "next_block_seq_len "  << next_block_seq_len << endl << flush;

                int new_block_seq_len_start = block_size * seq_len_blocks_i + block_seq_len;
                int new_block_seq_len_end = new_block_seq_len_start + next_block_seq_len;

                int next_block_seq_len_start = block_size * (seq_len_blocks_i + 1);
                int next_block_seq_len_end = next_block_seq_len_start + next_block_seq_len;

                if (new_block_seq_len_start > seq_len) {
                    break;
                }

                seq_len_blocks_lengths[batch_i][int(seq_len_blocks_i / 2)] = block_seq_len;

                if (next_block_seq_len_start > seq_len) {
                    break;
                }

                // cout << "new_block_seq_len_start "  << new_block_seq_len_start << endl << flush;
                // cout << "new_block_seq_len_end "    << new_block_seq_len_end << endl << flush;
                // cout << "next_block_seq_len_start " << next_block_seq_len_start << endl << flush;
                // cout << "next_block_seq_len_end "   << next_block_seq_len_end << endl << flush;

                merged_hidden_state.slice(0, batch_i, batch_i + 1).slice(1, new_block_seq_len_start, new_block_seq_len_end)       = merged_hidden_state_clone.slice(0, batch_i, batch_i + 1).slice(1, next_block_seq_len_start, next_block_seq_len_end);
                merged_embeddings_counts.slice(0, batch_i, batch_i + 1).slice(1, new_block_seq_len_start, new_block_seq_len_end)  = merged_embeddings_counts_clone.slice(0, batch_i, batch_i + 1).slice(1, next_block_seq_len_start, next_block_seq_len_end);
                merged_attention_mask.slice(0, batch_i, batch_i + 1).slice(1, new_block_seq_len_start, new_block_seq_len_end)     = merged_attention_mask_clone.slice(0, batch_i, batch_i + 1).slice(1, next_block_seq_len_start, next_block_seq_len_end);

                seq_len_blocks_lengths[batch_i][int(seq_len_blocks_i / 2)] = block_seq_len + next_block_seq_len;
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> prune_tokens_concrete_cuda(
    const torch::Tensor& hidden_state,            // [ bs, seq_len, hidden_dim ]
    const torch::Tensor& concrete_bool,           // [ bs, seq_len ]
    const torch::Tensor& attention_mask,          // [ bs, seq_len ]
    const torch::Tensor& special_embeddings_mask, // [ bs, seq_len ]
    const torch::Tensor& concrete                 // [ bs, seq_len ]
) {
    const int batch_size = concrete_bool.size(0);
    const int seq_len = concrete_bool.size(1);
    const int hidden_dim = hidden_state.size(2);

    // Launch the kernel
    int num_threads = (hidden_dim + HIDDEN_DIM_BLOCK_SIZE - 1) / HIDDEN_DIM_BLOCK_SIZE;
    const dim3 block_size(num_threads, 1, 1);  // One thread per hidden_dim_span
    const dim3 grid_size(batch_size, 1, 1);   // One block per batch element

    const dim3 block_size_prune(1, 1, 1);  // One thread per hidden_dim_span
    const dim3 grid_size_prune(batch_size, 1, 1);   // One block per batch element

    auto options = concrete_bool.options();
    auto device = options.device();

    auto merged_embeddings_counts_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto mask_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto merged_hidden_state_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto concrete_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    torch::Tensor merged_hidden_state = torch::zeros({batch_size, seq_len, hidden_dim}, merged_hidden_state_options);
    torch::Tensor merged_embeddings_counts = torch::zeros({batch_size, seq_len}, merged_embeddings_counts_options);
    torch::Tensor merged_attention_mask = torch::zeros({batch_size, seq_len}, mask_options);
    torch::Tensor merged_special_embeddings_mask = torch::zeros({batch_size, seq_len}, mask_options);
    torch::Tensor merged_concrete = torch::zeros({batch_size, seq_len}, concrete_options);

    torch::Tensor seq_len_blocks_lengths = torch::zeros({batch_size, 1}, merged_embeddings_counts_options);

    // cudaStream_t stream1, stream2;
    // cudaStreamCreate(&stream1);
    // cudaStreamCreate(&stream2);

    prune_tokens_concrete_kernel<<<grid_size_prune, block_size_prune, 0>>>(
        hidden_state.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
        concrete_bool.packed_accessor64<bool, 2, torch::RestrictPtrTraits>(),
        attention_mask.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        special_embeddings_mask.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        concrete.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        merged_embeddings_counts.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        merged_attention_mask.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        merged_special_embeddings_mask.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        merged_concrete.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        seq_len_blocks_lengths.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        batch_size, seq_len, hidden_dim
    );

    prune_tokens_concrete_copy_hidden_state_kernel<<<grid_size, block_size, 0>>>(
        hidden_state.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
        concrete_bool.packed_accessor64<bool, 2, torch::RestrictPtrTraits>(),
        attention_mask.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        merged_hidden_state.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
        batch_size, seq_len, hidden_dim
    );

    // cudaStreamSynchronize(stream1);
    // cudaStreamDestroy(stream1);

    seq_len_blocks_lengths = seq_len_blocks_lengths.to(torch::kCPU);

    int64_t max_seq_len = seq_len_blocks_lengths.slice(1, 0, 1).max().item<int64_t>();
    int64_t slice_from = seq_len - max_seq_len;

    namespace index = torch::indexing;
    torch::Tensor sliced_merged_hidden_state = merged_hidden_state.index({index::Slice(), index::Slice(slice_from, seq_len), index::Slice()});
    torch::Tensor sliced_merged_embeddings_counts = merged_embeddings_counts.index({index::Slice(), index::Slice(slice_from, seq_len)});
    torch::Tensor sliced_merged_attention_mask = merged_attention_mask.index({index::Slice(), index::Slice(slice_from, seq_len)});
    torch::Tensor sliced_merged_special_embeddings_mask = merged_special_embeddings_mask.index({index::Slice(), index::Slice(slice_from, seq_len)});
    torch::Tensor sliced_merged_concrete = merged_concrete.index({index::Slice(), index::Slice(slice_from, seq_len)});

    // cudaStreamSynchronize(stream2);
    // cudaStreamDestroy(stream2);

    return std::make_tuple(sliced_merged_hidden_state, sliced_merged_embeddings_counts, sliced_merged_attention_mask, 
                          sliced_merged_special_embeddings_mask, sliced_merged_concrete);
}

std::tuple<torch::Tensor, torch::Tensor> backward_prune_tokens_concrete_cuda(
    const torch::Tensor& grad_output_hidden_state,
    const torch::Tensor& grad_output_concrete,
    const torch::Tensor& concrete_bool,
    const torch::Tensor& attention_mask
) {
    const int batch_size = concrete_bool.size(0);
    const int seq_len = concrete_bool.size(1);
    const int hidden_dim = grad_output_hidden_state.size(2);
    const int grad_output_seq_len = grad_output_hidden_state.size(1);

    // Launch the kernel
    int num_threads = (hidden_dim + HIDDEN_DIM_BLOCK_SIZE - 1) / HIDDEN_DIM_BLOCK_SIZE;
    const dim3 block_size(num_threads, 1, 1);  // One thread per hidden_dim_span
    const dim3 grid_size(batch_size, 1, 1);   // One block per batch element

    torch::Tensor grad_hidden_state = torch::zeros({batch_size, seq_len, hidden_dim}, grad_output_hidden_state.options());
    torch::Tensor grad_concrete = torch::zeros({batch_size, seq_len}, grad_output_concrete.options());

    backward_prune_tokens_concrete_kernel<<<grid_size, block_size>>>(
        grad_output_hidden_state.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
        grad_output_concrete.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        concrete_bool.packed_accessor64<bool, 2, torch::RestrictPtrTraits>(),
        attention_mask.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
        grad_hidden_state.packed_accessor64<float, 3, torch::RestrictPtrTraits>(),
        grad_concrete.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
        batch_size, seq_len, hidden_dim, grad_output_seq_len
    );

    return std::make_tuple(grad_hidden_state, grad_concrete);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(generate_merges, m) {
    m.def("generate_merges_transform(Tensor a, Tensor b, Tensor c) -> (Tensor, Tensor, Tensor)");
    m.def("batch_repeat_interleave_for_merges_count(Tensor a, Tensor b) -> Tensor");
    m.def("fan_out_restore_residuals(Tensor a, Tensor b, Tensor c, Tensor d) -> Tensor");
    m.def("backward_fan_out_restore_residuals(Tensor a, Tensor b, Tensor c) -> (Tensor, Tensor)");
    m.def("prune_tokens_concrete_cuda(Tensor a, Tensor b, Tensor c, Tensor d, Tensor e) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("backward_prune_tokens_concrete_cuda(Tensor a, Tensor b, Tensor c, Tensor d) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(generate_merges, CUDA, m) {
    m.impl("generate_merges_transform", &generate_merges_transform_cuda);
    m.impl("batch_repeat_interleave_for_merges_count", &batch_repeat_interleave_for_merges_count);
    m.impl("fan_out_restore_residuals", &fan_out_restore_residuals);
    m.impl("backward_fan_out_restore_residuals", &backward_fan_out_restore_residuals);
    m.impl("prune_tokens_concrete_cuda", &prune_tokens_concrete_cuda);
    m.impl("backward_prune_tokens_concrete_cuda", &backward_prune_tokens_concrete_cuda);
}
