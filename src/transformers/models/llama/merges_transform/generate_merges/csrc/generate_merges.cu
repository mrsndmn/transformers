#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

void check_cuda_errors() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

__global__ void generate_merges_transform_kernel(
    const torch::PackedTensorAccessor64<int64_t, 3> merging_map,
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

        bool want_merge = merging_map[batch_i][seq_len_i][1] > 0L; // Check if merge is requested
        if (want_merge && seq_len_i < seq_len - 1) {
            if (buffer_length == 0) {
                start_want_merge = seq_len_i;
            }
            buffer_length++;
        } else {
            if (buffer_length > 0) {
                // Handle the merging
                merged_embeddings_counts[batch_i][seq_len_i] = seq_len_i - start_want_merge;
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

void generate_merges_transform_cuda(
    const torch::Tensor& merging_map,
    const torch::Tensor& attention_mask,
    torch::Tensor& aggregated_embeddings_transform,
    torch::Tensor& merged_embeddings_counts,
    torch::Tensor& merged_attention_mask
) {
    const int batch_size = merging_map.size(0);
    const int seq_len = merging_map.size(1);

    // Launch the kernel
    const dim3 block_size(1, 1, 1);  // One thread per sequence element
    const dim3 grid_size(batch_size, 1, 1);   // One block per batch element

    generate_merges_transform_kernel<<<grid_size, block_size>>>(
        merging_map.packed_accessor64<int64_t, 3>(),
        attention_mask.packed_accessor64<bool, 2>(),
        merged_embeddings_counts.packed_accessor64<int64_t, 2>(),
        merged_attention_mask.packed_accessor64<bool, 2>(),
        aggregated_embeddings_transform.packed_accessor64<float, 3>(),
        batch_size, seq_len
    );

    // Error checking
    cudaDeviceSynchronize();
    check_cuda_errors();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(generate_merges, m) {
    m.def("generate_merges_transform", &generate_merges_transform_cuda);
}