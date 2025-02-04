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
            new_seq_len_i += 1;
        } else {
            merged_embeddings_counts[batch_i][new_seq_len_i] += 1;
        }
    }

    // Update the merged_attention_mask for the current batch
    for (int i = 0; i < new_seq_len_i; ++i) {
        merged_attention_mask[batch_i][i] = true;
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

        if (current_repeats_num == 1) {
            merging_map_output[batch_i][output_seq_len_i][0] = grad_merging_map[batch_i][seq_len_i][0];
            merging_map_output[batch_i][output_seq_len_i][1] = grad_merging_map[batch_i][seq_len_i][1];
            ++output_seq_len_i;            
        } else {
            // TODO! Test cover
            output_seq_len_i = output_seq_len_i + current_repeats_num - 1;
            merging_map_output[batch_i][output_seq_len_i][1] = grad_merging_map[batch_i][seq_len_i][1];
            ++output_seq_len_i;
            
            for (int repeats_i = 0; repeats_i < current_repeats_num; ++repeats_i) {
                // merging_map_output[batch_i][output_seq_len_i][0] = 0;
                merging_map_output[batch_i][output_seq_len_i][1] = grad_merging_map[batch_i][seq_len_i][1];
                ++output_seq_len_i;
            }
        }

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
    const torch::PackedTensorAccessor64<float, 3> hidden_states,
    torch::PackedTensorAccessor64<float, 3> restored_hidden_states,
    int batch_size, int seq_len, int hidden_dim
) {
    int batch_i = blockIdx.x; // Batch index

    // TODO could be also parallelized by sequence dim!
    if (batch_i >= batch_size) {
        return; // Out of bounds check
    }

    int restored_seq_len = 0;
    for (int seq_len_i = 0; seq_len_i < seq_len; ++seq_len_i) {
        auto num_repeats = merged_embeddings_counts[batch_i][seq_len_i];
        if (num_repeats == 0) {
            break;
        }

        int restored_idx = int(restored_seq_len + num_repeats - 1);
        for (int hi = 0; hi < hidden_dim; ++hi) {
            restored_hidden_states[batch_i][restored_idx][hi] = hidden_states[batch_i][seq_len_i][hi];
        }
        restored_seq_len += num_repeats;
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
    const dim3 block_size(1, 1, 1);  // One thread per sequence element
    const dim3 grid_size(batch_size, 1, 1);   // One block per batch element

    torch::Tensor restored_hidden_states = torch::clone(residual_hidden_states_projection);

    fan_out_restore_residuals_kernel<<<grid_size, block_size>>>(
        merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
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


__global__ void prune_tokens_concrete_kernel(
    const torch::PackedTensorAccessor64<float, 3> hidden_state,
    const torch::PackedTensorAccessor64<bool, 2> concrete_bool,
    const torch::PackedTensorAccessor64<bool, 2> attention_mask,
    torch::PackedTensorAccessor64<float, 3> merged_hidden_state,
    torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_counts,
    torch::PackedTensorAccessor64<bool, 2> merged_attention_mask,
    int batch_size, int seq_len, int hidden_dim
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

        bool is_token_important = concrete_bool[batch_i][seq_len_i];

        if (is_token_important ||  seq_len_i == seq_len - 1) {
            merged_embeddings_counts[batch_i][new_seq_len_i] += 1;
            merged_attention_mask[batch_i][new_seq_len_i] = true;

            for (int i = 0; i < hidden_dim; ++i) {
                merged_hidden_state[batch_i][new_seq_len_i][i] = hidden_state[batch_i][seq_len_i][i];
            }

            new_seq_len_i += 1;
        } else {
            merged_embeddings_counts[batch_i][new_seq_len_i] += 1;
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
    const dim3 block_size(1, 1, 1);  // One thread per sequence element
    const dim3 grid_size(batch_size, 1, 1);   // One block per batch element

    auto options = concrete_bool.options();
    auto device = options.device();

    auto merged_embeddings_counts_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto mask_options = torch::TensorOptions().dtype(torch::kBool).device(device);
    auto merged_hidden_state_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    torch::Tensor merged_hidden_state = torch::zeros({batch_size, seq_len, hidden_dim}, merged_hidden_state_options);
    torch::Tensor merged_embeddings_counts = torch::zeros({batch_size, seq_len}, merged_embeddings_counts_options);
    torch::Tensor merged_attention_mask = torch::zeros({batch_size, seq_len}, mask_options);

    prune_tokens_concrete_kernel<<<grid_size, block_size>>>(
        hidden_state.packed_accessor64<float, 3>(),
        concrete_bool.packed_accessor64<bool, 2>(),
        attention_mask.packed_accessor64<bool, 2>(),

        merged_hidden_state.packed_accessor64<float, 3>(),
        merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
        merged_attention_mask.packed_accessor64<bool, 2>(),
        batch_size, seq_len, hidden_dim
    );

    // Error checking
    // cudaDeviceSynchronize();
    // check_cuda_errors();

    return std::make_tuple(merged_hidden_state, merged_embeddings_counts, merged_attention_mask);
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
