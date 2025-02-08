#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

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

        if (is_token_important ||  seq_len_i == seq_len - 1) {
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

__global__ void fan_out_restore_residuals_kernel(
    const torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_counts,
    const torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_cumsum,
    const torch::PackedTensorAccessor64<float, 3> hidden_states,
    torch::PackedTensorAccessor64<float, 3> restored_hidden_states,
    int batch_size, int seq_len, int hidden_dim
) {
    int batch_i = blockIdx.x; // Batch index
    int seq_len_i = blockIdx.y; // Sequence index
    int thread_idx = threadIdx.x;

    if (batch_i >= batch_size || seq_len_i >= seq_len) {
        return; // Out of bounds check
    }

    int hidden_dim_start = thread_idx * 64;
    if (hidden_dim_start > hidden_dim) {
        return;
    }
    int hidden_dim_end = hidden_dim_start + 64;
    if (hidden_dim_end > hidden_dim) {
        hidden_dim_end = hidden_dim;
    }

    auto num_repeats = merged_embeddings_counts[batch_i][seq_len_i];
    if (num_repeats == 0) {
        return;
    }

    auto restored_idx = merged_embeddings_cumsum[batch_i][seq_len_i] - 1;

    for (int hi = hidden_dim_start; hi < hidden_dim_end; ++hi) {
        restored_hidden_states[batch_i][restored_idx][hi] = hidden_states[batch_i][seq_len_i][hi];
    }
}

torch::Tensor fan_out_restore_residuals(
    const torch::Tensor& merged_embeddings_counts,
    const torch::Tensor& hidden_states,
    const torch::Tensor& residual_hidden_states_projection
) {
    const int batch_size = merged_embeddings_counts.size(0);
    const int seq_len = merged_embeddings_counts.size(1);
    const int hidden_dim = residual_hidden_states_projection.size(2);

    // Launch the kernel

    int num_threads = (hidden_dim + 63) / 64;

    const dim3 block_size(num_threads, 1, 1);  // One thread per sequence element
    const dim3 grid_size(batch_size, seq_len, 1);   // One block per batch element

    torch::Tensor restored_hidden_states = torch::clone(residual_hidden_states_projection);

    torch::Tensor merged_embeddings_cumsum = torch::cumsum(merged_embeddings_counts, 1);

    fan_out_restore_residuals_kernel<<<grid_size, block_size>>>(
        merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
        merged_embeddings_cumsum.packed_accessor64<int64_t, 2>(),
        hidden_states.packed_accessor64<float, 3>(),
        restored_hidden_states.packed_accessor64<float, 3>(),
        batch_size, seq_len, hidden_dim
    );

    // Error checking
    // cudaDeviceSynchronize();
    // check_cuda_errors();

    return restored_hidden_states;
}


__global__ void backward_fan_out_straight_kernel(
    const torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_counts,
    const torch::PackedTensorAccessor64<float, 3> restored_hidden_states_grad,
    torch::PackedTensorAccessor64<float, 3> hidden_states_grad,
    torch::PackedTensorAccessor64<float, 3> residual_hidden_states_grad,
    int batch_size, int seq_len, int hidden_dim
) {
    int batch_i = blockIdx.x; // Batch index

    // TODO could be also parallelized by sequence dim!
    if (batch_i >= batch_size) {
        return; // Out of bounds check
    }

    int output_seq_len_i = 0;
    for (int seq_len_i = 0; seq_len_i < seq_len; ++seq_len_i) {
        auto num_repeats = merged_embeddings_counts[batch_i][seq_len_i];
        if (num_repeats == 0) {
            break;
        }

        // for hidden_states_grad
        int restored_idx = int(output_seq_len_i + num_repeats - 1);
        for (int hi = 0; hi < hidden_dim; ++hi) {
            hidden_states_grad[batch_i][seq_len_i][hi] = restored_hidden_states_grad[batch_i][restored_idx][hi];
        }

        // for residual_hidden_states_grad
        for (int residuals_grad_i = 0; residuals_grad_i < num_repeats - 1; ++residuals_grad_i) {
            for (int hi = 0; hi < hidden_dim; ++hi) {
                residual_hidden_states_grad[batch_i][output_seq_len_i + residuals_grad_i][hi] = restored_hidden_states_grad[batch_i][output_seq_len_i + residuals_grad_i][hi];
            }
        }

        output_seq_len_i += num_repeats;
    }
}

std::tuple<torch::Tensor, torch::Tensor> backward_fan_out_restore_residuals(
    const torch::Tensor& merged_embeddings_counts,
    const torch::Tensor& restored_hidden_states_grad
) {
    const int batch_size = merged_embeddings_counts.size(0);
    const int seq_len = merged_embeddings_counts.size(1);
    const int hidden_dim = restored_hidden_states_grad.size(2);

    // Launch the kernel
    const dim3 block_size(1, 1, 1);  // One thread per sequence element
    const dim3 grid_size(batch_size, 1, 1);   // One block per batch element

    torch::Tensor hidden_states_grad = torch::zeros({batch_size, seq_len, hidden_dim}, restored_hidden_states_grad.options());
    torch::Tensor residual_hidden_states_grad = torch::zeros_like(restored_hidden_states_grad, restored_hidden_states_grad.options());

    backward_fan_out_straight_kernel<<<grid_size, block_size>>>(
        merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
        restored_hidden_states_grad.packed_accessor64<float, 3>(),
        hidden_states_grad.packed_accessor64<float, 3>(),
        residual_hidden_states_grad.packed_accessor64<float, 3>(),
        batch_size, seq_len, hidden_dim
    );

    // Error checking
    // cudaDeviceSynchronize();
    // check_cuda_errors();

    return std::make_tuple(hidden_states_grad, residual_hidden_states_grad);
}

const int HIDDEN_DIM_BLOCK_SIZE = 128;
const int SEQ_LEN_BLOCK_SIZE = 64;

__global__ void prune_tokens_concrete_kernel(
    const torch::PackedTensorAccessor64<float, 3> hidden_state,
    const torch::PackedTensorAccessor64<bool, 2> concrete_bool,
    const torch::PackedTensorAccessor64<bool, 2> attention_mask,
    torch::PackedTensorAccessor64<float, 3> merged_hidden_state,
    torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_counts,
    torch::PackedTensorAccessor64<bool, 2> merged_attention_mask,
    torch::PackedTensorAccessor64<int64_t, 2> seq_len_blocks_lengths,
    int batch_size, int seq_len, int hidden_dim
) {
    int batch_i = blockIdx.x; // Batch index
    int seq_len_block_idx = blockIdx.y; // Index for the new sequence length
    int thread_idx = threadIdx.x; // Index for hidden dim

    if (batch_i >= batch_size || seq_len_block_idx * SEQ_LEN_BLOCK_SIZE >= seq_len + SEQ_LEN_BLOCK_SIZE) {
        return; // Out of bounds check
    }

    int hidden_dim_start = thread_idx * HIDDEN_DIM_BLOCK_SIZE;
    if (hidden_dim_start > hidden_dim) {
        return;
    }
    int hidden_dim_end = hidden_dim_start + HIDDEN_DIM_BLOCK_SIZE;
    if (hidden_dim_end > hidden_dim) {
        hidden_dim_end = hidden_dim;
    }

    int initial_new_seq_len_i = seq_len_block_idx * SEQ_LEN_BLOCK_SIZE;
    int new_seq_len_i = seq_len_block_idx * SEQ_LEN_BLOCK_SIZE;

    int seq_len_i_start = seq_len_block_idx * SEQ_LEN_BLOCK_SIZE;
    int seq_len_i_end = seq_len_block_idx * SEQ_LEN_BLOCK_SIZE + SEQ_LEN_BLOCK_SIZE;
    if (seq_len_i_end > seq_len) {
        seq_len_i_end = seq_len;
    }

    for (int seq_len_i = seq_len_i_start; seq_len_i < seq_len_i_end; ++seq_len_i) {
        if (!attention_mask[batch_i][seq_len_i]) {
            break;
        }

        bool is_token_important = concrete_bool[batch_i][seq_len_i];

        if (is_token_important ||  seq_len_i == seq_len - 1) {

            if (thread_idx == 0) {
                merged_embeddings_counts[batch_i][new_seq_len_i] += 1;
                merged_attention_mask[batch_i][new_seq_len_i] = true;
            }

            for (int i = hidden_dim_start; i < hidden_dim_end; ++i) {
                merged_hidden_state[batch_i][new_seq_len_i][i] = hidden_state[batch_i][seq_len_i][i];
            }

            new_seq_len_i += 1;
        } else {
            if (thread_idx == 0) {
                merged_embeddings_counts[batch_i][new_seq_len_i] += 1;
            }
        }
    }

    seq_len_blocks_lengths[batch_i][seq_len_block_idx] = new_seq_len_i - initial_new_seq_len_i;
}

__global__ void inplace_merge_pruned_tokens(
    const torch::PackedTensorAccessor64<int64_t, 2> seq_len_blocks_lengths,
    torch::PackedTensorAccessor64<float, 3> merged_hidden_state,
    torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_counts,
    torch::PackedTensorAccessor64<bool, 2> merged_attention_mask,
    int batch_size, int seq_len, int hidden_dim
) {
    int batch_i = blockIdx.x; // Batch index
    int thread_idx = threadIdx.x; // Index for hidden dim

    if (batch_i >= batch_size) {
        return; // Out of bounds check
    }

    int hidden_dim_start = thread_idx * HIDDEN_DIM_BLOCK_SIZE;
    if (hidden_dim_start > hidden_dim) {
        return;
    }

    int hidden_dim_end = hidden_dim_start + HIDDEN_DIM_BLOCK_SIZE;
    if (hidden_dim_end > hidden_dim) {
        hidden_dim_end = hidden_dim;
    }

    const int seq_len_blocks_len = seq_len_blocks_lengths.size(1); // [ bs, seq_len_blocks_len ]

    int current_new_seq_len_i = 0;

    for (int block_i = 0; block_i < seq_len_blocks_len; ++block_i) {
        int seq_len_start = block_i * SEQ_LEN_BLOCK_SIZE;
        int seq_len_end = block_i * SEQ_LEN_BLOCK_SIZE + seq_len_blocks_lengths[batch_i][block_i];

        for (int seq_len_i = seq_len_start; seq_len_i < seq_len_end; ++seq_len_i) {
            if (merged_embeddings_counts[batch_i][seq_len_i] == 0) {
                break;
            }

            for (int i = hidden_dim_start; i < hidden_dim_end; ++i) {
                merged_hidden_state[batch_i][current_new_seq_len_i][i] = merged_hidden_state[batch_i][seq_len_i][i];
            }

            merged_embeddings_counts[batch_i][current_new_seq_len_i] = merged_embeddings_counts[batch_i][seq_len_i];
            merged_attention_mask[batch_i][current_new_seq_len_i] = merged_attention_mask[batch_i][seq_len_i];

            current_new_seq_len_i += 1;
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

    int seq_len_pow2 = 1;
    int seq_len_pow2_exp = 1;
    while(seq_len_pow2 < seq_len) {
        seq_len_pow2 = seq_len_pow2 << 1;
        seq_len_pow2_exp += 1;
    }

    int init_seq_len_threads_len = seq_len_blocks_lengths.size(1);

    for (int batch_i = 0; batch_i < batch_size; ++batch_i) {
        // todo process each batch in separate cuda stream

        for (int divide_and_conquer_i = 1; divide_and_conquer_i < seq_len_pow2_exp; ++divide_and_conquer_i) {
            int block_size = SEQ_LEN_BLOCK_SIZE * divide_and_conquer_i;
            int seq_len_len = seq_len_pow2 / (1 << (divide_and_conquer_i - 1));
            if (seq_len_len > init_seq_len_threads_len) {
                seq_len_len = init_seq_len_threads_len;
            }

            torch::Tensor merged_hidden_state_clone = torch::clone(merged_hidden_state);
            torch::Tensor merged_embeddings_counts_clone = torch::clone(merged_embeddings_counts);
            torch::Tensor merged_attention_mask_clone = torch::clone(merged_attention_mask);

            for (int seq_len_blocks_i = 0; seq_len_blocks_i < seq_len_len - 1; seq_len_blocks_i += 2) {
                if (seq_len_blocks_i + 1 >= seq_len_len) {
                    // Neighbour block is out of range
                    seq_len_blocks_lengths[batch_i][int(seq_len_blocks_i / 2)] = seq_len_blocks_lengths[batch_i][seq_len_blocks_i];
                    break;
                }

                int64_t block_seq_len = seq_len_blocks_lengths[batch_i][seq_len_blocks_i].item<int64_t>();
                int64_t next_block_seq_len = seq_len_blocks_lengths[batch_i][seq_len_blocks_i+1].item<int64_t>();

                int new_block_seq_len_start = block_size * seq_len_blocks_i + block_seq_len;
                int new_block_seq_len_end = new_block_seq_len_start + next_block_seq_len;

                int next_block_seq_len_start = block_size * (seq_len_blocks_i + 1);
                int next_block_seq_len_end = next_block_seq_len_start + next_block_seq_len;

                merged_hidden_state.slice(0, batch_i, batch_i + 1).slice(1, new_block_seq_len_start, new_block_seq_len_end)       = merged_hidden_state_clone.slice(0, batch_i, batch_i + 1).slice(1, next_block_seq_len_start, next_block_seq_len_end);
                merged_embeddings_counts.slice(0, batch_i, batch_i + 1).slice(1, new_block_seq_len_start, new_block_seq_len_end)  = merged_embeddings_counts_clone.slice(0, batch_i, batch_i + 1).slice(1, next_block_seq_len_start, next_block_seq_len_end);
                merged_attention_mask.slice(0, batch_i, batch_i + 1).slice(1, new_block_seq_len_start, new_block_seq_len_end)     = merged_attention_mask_clone.slice(0, batch_i, batch_i + 1).slice(1, next_block_seq_len_start, next_block_seq_len_end);

                int new_seq_len = block_seq_len + next_block_seq_len;
                seq_len_blocks_lengths[batch_i][int(seq_len_blocks_i / 2)] = new_seq_len;
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> prune_tokens_concrete_cuda(
    const torch::Tensor& hidden_state,            // [ bs, seq_len, hidden_dim ]
    const torch::Tensor& concrete_bool,           // [ bs, seq_len ]
    const torch::Tensor& attention_mask          // [ bs, seq_len ]
) {
    const int batch_size = concrete_bool.size(0);
    const int seq_len = concrete_bool.size(1);
    const int hidden_dim = hidden_state.size(2);

    // Launch the kernel
    int num_threads = (hidden_dim + HIDDEN_DIM_BLOCK_SIZE - 1) / HIDDEN_DIM_BLOCK_SIZE;
    int seq_len_threads = (seq_len + SEQ_LEN_BLOCK_SIZE - 1) / SEQ_LEN_BLOCK_SIZE;
    const dim3 block_size(num_threads, 1, 1);  // One thread per hidden_dim_span
    const dim3 grid_size(batch_size, seq_len_threads, 1);   // One block per batch element

    auto options = concrete_bool.options();
    auto device = options.device();

    auto merged_embeddings_counts_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto mask_options = torch::TensorOptions().dtype(torch::kBool).device(device);
    auto merged_hidden_state_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    torch::Tensor merged_hidden_state = torch::zeros({batch_size, seq_len, hidden_dim}, merged_hidden_state_options);
    torch::Tensor merged_embeddings_counts = torch::zeros({batch_size, seq_len}, merged_embeddings_counts_options);
    torch::Tensor merged_attention_mask = torch::zeros({batch_size, seq_len}, mask_options);

    torch::Tensor seq_len_blocks_lengths = torch::zeros({batch_size, seq_len_threads}, merged_embeddings_counts_options);

    // parallelize by 3 dimensions
    prune_tokens_concrete_kernel<<<grid_size, block_size>>>(
        hidden_state.packed_accessor64<float, 3>(),
        concrete_bool.packed_accessor64<bool, 2>(),
        attention_mask.packed_accessor64<bool, 2>(),

        merged_hidden_state.packed_accessor64<float, 3>(),
        merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
        merged_attention_mask.packed_accessor64<bool, 2>(),

        seq_len_blocks_lengths.packed_accessor64<int64_t, 2>(),

        batch_size, seq_len, hidden_dim
    );

    // this kernel could be parallelized only by 2 dimensions

    // todo разделяй и влавствуй
    seq_len_blocks_lengths = seq_len_blocks_lengths.to(torch::kCPU);

    collapse_blocks(
        seq_len_blocks_lengths,
        merged_hidden_state,
        merged_embeddings_counts,
        merged_attention_mask
    );

    // const dim3 merge_block_size(num_threads, 1, 1);  // One thread per hidden_dim_span
    // const dim3 merge_grid_size(batch_size, 1, 1);   // One block per batch element
    // inplace_merge_pruned_tokens<<<merge_grid_size, merge_block_size>>>(
    //     seq_len_blocks_lengths.packed_accessor64<int64_t, 2>(),

    //     merged_hidden_state.packed_accessor64<float, 3>(),
    //     merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
    //     merged_attention_mask.packed_accessor64<bool, 2>(),

    //     batch_size, seq_len, hidden_dim
    // );


    torch::Tensor new_seq_lengths = merged_attention_mask.sum(1);

    int64_t max_seq_len = new_seq_lengths.max().item<int64_t>();

    namespace index = torch::indexing;
    torch::Tensor sliced_merged_hidden_state = merged_hidden_state.index({index::Slice(), index::Slice(0, max_seq_len), index::Slice()});
    torch::Tensor sliced_merged_embeddings_counts = merged_embeddings_counts.index({index::Slice(), index::Slice(0, max_seq_len)});
    torch::Tensor sliced_merged_attention_mask = merged_attention_mask.index({index::Slice(), index::Slice(0, max_seq_len)});

    // Error checking
    // cudaDeviceSynchronize();
    // check_cuda_errors();


    return std::make_tuple(sliced_merged_hidden_state, sliced_merged_embeddings_counts, sliced_merged_attention_mask);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(generate_merges, m) {
    m.def("generate_merges_transform(Tensor a, Tensor b, Tensor c) -> (Tensor, Tensor, Tensor)");
    m.def("batch_repeat_interleave_for_merges_count(Tensor a, Tensor b) -> Tensor");
    m.def("fan_out_restore_residuals(Tensor a, Tensor b, Tensor c) -> Tensor");
    m.def("backward_fan_out_restore_residuals(Tensor a, Tensor b) -> (Tensor, Tensor)");
    m.def("prune_tokens_concrete_cuda(Tensor a, Tensor b, Tensor c) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(generate_merges, CUDA, m) {
    m.impl("generate_merges_transform", &generate_merges_transform_cuda);
    m.impl("batch_repeat_interleave_for_merges_count", &batch_repeat_interleave_for_merges_count);
    m.impl("fan_out_restore_residuals", &fan_out_restore_residuals);
    m.impl("backward_fan_out_restore_residuals", &backward_fan_out_restore_residuals);
    m.impl("prune_tokens_concrete_cuda", &prune_tokens_concrete_cuda);
}
