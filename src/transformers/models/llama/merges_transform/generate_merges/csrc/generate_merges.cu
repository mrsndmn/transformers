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

    int buffer_length = 0;
    int start_want_merge = 0;
    int new_seq_len_i = 0;

    for (int seq_len_i = 0; seq_len_i < seq_len; ++seq_len_i) {
        if (!attention_mask[batch_i][seq_len_i]) {
            break;
        }

        bool want_merge = merging_map[batch_i][seq_len_i][1] > 0.0f; // Check if merge is requested
        if (want_merge && seq_len_i < seq_len - 1) {
            if (buffer_length == 0) {
                start_want_merge = seq_len_i;
            }
            buffer_length++;
        } else {
            if (buffer_length > 0) {
                // Handle the merging
                merged_embeddings_counts[batch_i][new_seq_len_i] = seq_len_i - start_want_merge;
                for (int i = start_want_merge; i < seq_len_i; ++i) {
                    aggregated_embeddings_transform[batch_i][new_seq_len_i][i] = 1.0;
                }
                new_seq_len_i++;
                buffer_length = 0;
            }

            // Handle individual token (no merge)
            aggregated_embeddings_transform[batch_i][new_seq_len_i][seq_len_i] = 1.0;
            merged_embeddings_counts[batch_i][new_seq_len_i] = 1;
            new_seq_len_i++;
        }
    }

    // Update the merged_attention_mask for the current batch
    for (int i = 0; i < new_seq_len_i; ++i) {
        merged_attention_mask[batch_i][i] = true;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> generate_merges_transform_cuda(
    const torch::Tensor& merging_map,
    const torch::Tensor& attention_mask
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
    const torch::PackedTensorAccessor64<float, 3> merging_map,
    const torch::PackedTensorAccessor64<int64_t, 2> merged_embeddings_counts,
    torch::PackedTensorAccessor64<float, 3> merging_map_output,
    int batch_size, int seq_len
) {
    int batch_i = blockIdx.x; // Batch index
    // int seq_len_i = threadIdx.x; // Index for the new sequence length

    if (batch_i >= batch_size) {
        return; // Out of bounds check
    }

    merging_map_output[0][0][0] = 1.0f;

    int output_seq_len_i = 0;
    for (int seq_len_i = 0; seq_len_i < seq_len; ++seq_len_i) {
        auto current_repeats_num = merged_embeddings_counts[batch_i][seq_len_i];
        if (current_repeats_num == 0) {
            break;
        }

        for (int repeats_i = 0; repeats_i < current_repeats_num; ++repeats_i) {
            merging_map_output[batch_i][output_seq_len_i] = merging_map[batch_i][seq_len_i];
            ++output_seq_len_i;
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(generate_merges, m) {
    m.def("generate_merges_transform(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
    m.def("batch_repeat_interleave_for_merges_count(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(generate_merges, CUDA, m) {
    m.impl("generate_merges_transform", &generate_merges_transform_cuda);
    m.impl("batch_repeat_interleave_for_merges_count", &batch_repeat_interleave_for_merges_count);
}
