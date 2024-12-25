import time

import pytest

import torch

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_adaptive_llama import AdaptiveFanInGumbel, AdaptiveFanOut, AdaptiveFanInOutput, AdaptiveFanOutOutput, AdaptiveLlamaModel

def test_adaptive_fan_in_no_merge():
    config = LlamaConfig(hidden_size=256, num_hidden_layers=2)

    adaptive_fan_in = AdaptiveFanInGumbel(config)

    batch_size, seq_len = 3, 6
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ])
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    merging_log_probas = torch.ones([batch_size, seq_len, 2])
    merging_log_probas[:, :, 1] = 0
    merging_log_probas += 1e-4
    merging_log_probas = merging_log_probas.log()

    adaptive_fan_in_output = adaptive_fan_in.forward(hidden_states, attention_mask, special_embeddings_mask, merging_log_probas=merging_log_probas)

    assert adaptive_fan_in_output.attention_mask.shape[1] == seq_len
    assert adaptive_fan_in_output.hidden_state.shape[1] == seq_len
    assert adaptive_fan_in_output.merged_embeddings_counts.shape[1] == seq_len


def test_adaptive_fan_in_all_merge():
    config = LlamaConfig(hidden_size=256, num_hidden_layers=2, generate_merges_transform_impl='python')

    adaptive_fan_in = AdaptiveFanInGumbel(config)

    batch_size, seq_len = 3, 6
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ])
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    merging_log_probas = torch.ones([batch_size, seq_len, 2])
    merging_log_probas[:, :, 0] = 0
    merging_log_probas += 1e-4
    merging_log_probas = merging_log_probas.log()

    adaptive_fan_in_output = adaptive_fan_in.forward(hidden_states, attention_mask, special_embeddings_mask, merging_log_probas=merging_log_probas)

    assert adaptive_fan_in_output.attention_mask.shape[1] == 3 # bos + merged_embedding + eos
    assert adaptive_fan_in_output.hidden_state.shape[1] == 3 # bos + merged_embedding + eos
    assert adaptive_fan_in_output.merged_embeddings_counts.shape[1] == 3 # bos + merged_embedding + eos
    assert adaptive_fan_in_output.special_embeddings_mask.shape[1] == 3 # bos + merged_embedding + eos

    return

def test_adaptive_fan_in_all_but_last_merge():
    config = LlamaConfig(hidden_size=256, num_hidden_layers=2, generate_merges_transform_impl='python')

    adaptive_fan_in = AdaptiveFanInGumbel(config)

    batch_size, seq_len = 3, 6
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ])
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    merging_log_probas = torch.ones([batch_size, seq_len, 2])
    merging_log_probas[:, :, 1] = 0
    merging_log_probas[:, -2] = torch.tensor([0, 1])
    merging_log_probas += 1e-4
    merging_log_probas = merging_log_probas.log()

    adaptive_fan_in_output = adaptive_fan_in.forward(hidden_states, attention_mask, special_embeddings_mask, merging_log_probas=merging_log_probas)

    expected_seq_len = seq_len - 1

    assert adaptive_fan_in_output.attention_mask.shape[1] == expected_seq_len # bos + original_embedding + merged_embedding (with eos)
    assert adaptive_fan_in_output.hidden_state.shape[1] == expected_seq_len # bos + original_embedding + merged_embedding (with eos)
    assert adaptive_fan_in_output.merged_embeddings_counts.shape[1] == expected_seq_len # bos + original_embedding + merged_embedding (with eos)
    assert adaptive_fan_in_output.special_embeddings_mask.shape[1] == expected_seq_len # bos + original_embedding + merged_embedding (with eos)

    assert torch.allclose(adaptive_fan_in_output.hidden_state[:, :-2], hidden_states[:, :-3], atol=1e-4)
    assert torch.allclose(adaptive_fan_in_output.hidden_state[:, 0], hidden_states[:, 0], atol=1e-4)

    return


def test_adaptive_fan_out():
    config = LlamaConfig(hidden_size=256, num_hidden_layers=2)
    adaptive_fan_out = AdaptiveFanOut(config)

    batch_size, seq_len, residual_seq_len = 3, 7, 12

    hidden_states = torch.rand([batch_size, seq_len, config.hidden_size])
    attention_mask = torch.tensor([
        [ 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 0, 0],
        [ 1, 1, 1, 1, 0, 0, 0],
    ], dtype=torch.float32)
    merged_embeddings_counts = torch.tensor([
        [ 1, 3, 1, 2, 1, 3, 1],
        [ 1, 2, 2, 2, 1, 0, 0],
        [ 1, 5, 3, 1, 0, 0, 0],
    ], dtype=torch.long)
    residual_hidden_states = torch.rand([batch_size, residual_seq_len, config.hidden_size])
    residual_attention_mask = torch.tensor([
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    ], dtype=torch.float32)

    restored_hidden_states = adaptive_fan_out.forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        merged_embeddings_counts=merged_embeddings_counts,
        residual_hidden_states=residual_hidden_states,
        residual_attention_mask=residual_attention_mask,
    )

    assert restored_hidden_states.hidden_state.shape == residual_hidden_states.shape

    restored_hidden_states


def test_adaptive_fan_in_fan_out():
    config = LlamaConfig(hidden_size=256, num_hidden_layers=2)

    adaptive_fan_in = AdaptiveFanInGumbel(config)
    adaptive_fan_out = AdaptiveFanOut(config)

    batch_size, seq_len = 3, 6
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ], requires_grad=True)
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    adaptive_fan_in_output = adaptive_fan_in.forward(hidden_states, attention_mask, special_embeddings_mask)
    assert adaptive_fan_in_output.hidden_state.grad_fn is not None

    residual_hidden_states = hidden_states
    residual_attention_mask = attention_mask

    adaptive_fan_out_output = adaptive_fan_out.forward(
        hidden_states=adaptive_fan_in_output.hidden_state,
        attention_mask=adaptive_fan_in_output.attention_mask,
        merged_embeddings_counts=adaptive_fan_in_output.merged_embeddings_counts,
        residual_hidden_states=residual_hidden_states,
        residual_attention_mask=residual_attention_mask,
    )
    restored_hidden_states = adaptive_fan_out_output.hidden_state

    assert restored_hidden_states.shape == residual_hidden_states.shape

    restored_hidden_states.backward(torch.rand_like(restored_hidden_states))

    for name, p in adaptive_fan_in.named_parameters():
        assert p.grad is not None, f"adaptive_fan_in param grad is none: {name}"


def test_adaptive_llama_e2e():
    config = LlamaConfig(hidden_size=256, num_hidden_layers=2, attn_implementation='eager')

    allama_model = AdaptiveLlamaModel(config)
    batch_size, seq_len = 3, 6
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ], requires_grad=True)
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    llama_output = allama_model.forward(inputs_embeds=hidden_states, attention_mask=attention_mask, special_embeddings_mask=special_embeddings_mask, use_cache=False)

    assert llama_output.last_hidden_state.shape == hidden_states.shape

def test_cuda_kernel_merges_transform_generate_merges_transform():
    config_py = LlamaConfig(hidden_size=256, num_hidden_layers=2, attn_implementation='eager', generate_merges_transform_impl="python")
    config_cuda_kernel = LlamaConfig(hidden_size=256, num_hidden_layers=2, attn_implementation='eager', generate_merges_transform_impl="cuda_kernel")
    
    py_adaptive_fan_in_gumbel = AdaptiveFanInGumbel(config_py)
    cuda_adaptive_fan_in_gumbel = AdaptiveFanInGumbel(config_cuda_kernel)

    batch_size = 3
    seq_len = 5
    
    merging_map_1 = torch.zeros([batch_size, seq_len, 2])
    merging_map_1[:, :, 0] = 1.
    
    merging_map_2 = torch.zeros([batch_size, seq_len, 2])
    merging_map_2[:, 2:seq_len-1, 1] = 1
    merging_map_2[:, 0, 0] = 1
    merging_map_2[:, 1, 0] = 1
    merging_map_2[:, -1, 0] = 1
    
    merging_map_3 = torch.zeros([batch_size, seq_len, 2])
    merging_map_3[:, 1:seq_len-2, 1] = 1
    merging_map_3[:, 0, 0] = 1
    merging_map_3[:, -2, 0] = 1
    attention_mask_3 = torch.ones([batch_size, seq_len], dtype=torch.bool)
    attention_mask_3[:, seq_len-1] = False

    merging_map_4 = torch.tensor([
        [
            [ 1., 0. ],
            [ 1., 0. ],
            [ 0., 1. ],
            [ 0., 1. ],
            [ 1., 0. ],
        ],
        [
            [ 1., 0. ],
            [ 1., 0. ],
            [ 0., 1. ],
            [ 1., 0. ],
            [ 0., 0. ],
        ],
        [
            [ 1., 0. ],
            [ 1., 0. ],
            [ 1., 0. ],
            [ 0., 0. ],
            [ 0., 0. ],
        ],
    ])
    attention_mask_4 = torch.ones([batch_size, seq_len], dtype=torch.bool)
    attention_mask_4[1, -1] = False
    attention_mask_4[2, -2:] = False

    test_cases = [
        {
            "name": "dummy no merging",
            "merging_map": merging_map_1,
            "attention_mask": torch.ones([batch_size, seq_len], dtype=torch.bool),
            "expected_seq_len": seq_len,
        },
        {
            "name": "all except bos/eos merged",
            "merging_map": merging_map_2,
            "attention_mask": torch.ones([batch_size, seq_len], dtype=torch.bool),
            "expected_seq_len": 3, # [bos, merged_tokens, eos]
        },
        {
            "name": "all except bos/eos merged with padding",
            "merging_map": merging_map_3,
            "attention_mask": attention_mask_3,
            "expected_seq_len": 3,
        },
        {
            "name": "custom merging with custom padding",
            "merging_map": merging_map_4,
            "attention_mask": attention_mask_4,
            "expected_seq_len": 3,
        },
    ]

    # todo make fixtures not golang-style tests
    for test_case in test_cases:
        test_case_name = test_case['name']
        merging_map = test_case['merging_map']
        attention_mask = test_case['attention_mask']
        
        cuda_merging_map = merging_map.to('cuda').to(dtype=torch.float32)
        cuda_attention_mask = attention_mask.to('cuda').to(dtype=torch.bool)
        
        py_aggregated_embeddings_transform, py_merged_embeddings_counts, py_merged_attention_mask = py_adaptive_fan_in_gumbel.generate_merges_transform(merging_map, attention_mask)
        cuda_aggregated_embeddings_transform, cuda_merged_embeddings_counts, cuda_merged_attention_mask = cuda_adaptive_fan_in_gumbel.generate_merges_transform(cuda_merging_map, cuda_attention_mask)
        
        assert (py_aggregated_embeddings_transform == cuda_aggregated_embeddings_transform.to('cpu')).all(), f"{test_case_name}: aggregated_embeddings_transform mismatch"
        assert (py_merged_embeddings_counts == cuda_merged_embeddings_counts.to('cpu')).all(), f"{test_case_name}: merged_embeddings_counts mismatch"
        assert (py_merged_attention_mask == cuda_merged_attention_mask.to('cpu')).all(), f"{test_case_name}: merged_attention_mask mismatch"

    return


def test_cuda_kernel_merges_transform_benchmark():
    config_py = LlamaConfig(hidden_size=256, num_hidden_layers=2, attn_implementation='eager', generate_merges_transform_impl="python")
    config_cuda_kernel = LlamaConfig(hidden_size=256, num_hidden_layers=2, attn_implementation='eager', generate_merges_transform_impl="cuda_kernel")

    py_adaptive_fan_in_gumbel = AdaptiveFanInGumbel(config_py)
    cuda_adaptive_fan_in_gumbel = AdaptiveFanInGumbel(config_cuda_kernel)
    
    batch_sizes = [ 100 ]
    seq_lens = [ 128 ]
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:

            merging_map_1 = torch.zeros([batch_size, seq_len, 2])
            merging_map_1[:, :, 0] = 1.

            test_cases = [
                {
                    "name": "dummy no merging",
                    "merging_map": merging_map_1,
                    "attention_mask": torch.ones([batch_size, seq_len], dtype=torch.bool),
                },
            ]

            # todo make fixtures not golang-style tests
            for test_case in test_cases:
                test_case_name = test_case['name']
                merging_map = test_case['merging_map']
                attention_mask = test_case['attention_mask']
                
                cuda_merging_map = merging_map.to('cuda').to(dtype=torch.float32)
                cuda_attention_mask = attention_mask.to('cuda').to(dtype=torch.bool)
                
                n_runs = 10
                py_time_start = time.time()
                for _ in range(n_runs):
                    py_aggregated_embeddings_transform, py_merged_embeddings_counts, py_merged_attention_mask = py_adaptive_fan_in_gumbel.generate_merges_transform(merging_map, attention_mask)
                    py_aggregated_embeddings_transform.sum().item()
                py_duration = (time.time() - py_time_start) / n_runs

                cuda_time_start = time.time()
                for _ in range(n_runs):
                    cuda_aggregated_embeddings_transform, cuda_merged_embeddings_counts, cuda_merged_attention_mask = cuda_adaptive_fan_in_gumbel.generate_merges_transform(cuda_merging_map, cuda_attention_mask)
                    cuda_aggregated_embeddings_transform.sum().item()
                cuda_duration = (time.time() - cuda_time_start) / n_runs

                py_cuda_time_start = time.time()
                for _ in range(n_runs):
                    py_aggregated_embeddings_transform, py_merged_embeddings_counts, py_merged_attention_mask = py_adaptive_fan_in_gumbel.generate_merges_transform(cuda_merging_map, cuda_attention_mask)
                    py_aggregated_embeddings_transform.sum().item()
                py_cuda_duration = (time.time() - py_cuda_time_start) / n_runs

                print(f"bs={batch_size} seq_len={seq_len} cuda_duration", cuda_duration)
                print(f"bs={batch_size} seq_len={seq_len} py_cpu_duration", py_duration)
                print(f"bs={batch_size} seq_len={seq_len} py_cuda_duration", py_cuda_duration)
                
                assert (py_aggregated_embeddings_transform == cuda_aggregated_embeddings_transform).all(), f"{test_case_name}: aggregated_embeddings_transform mismatch"
                assert (py_merged_embeddings_counts == cuda_merged_embeddings_counts).all(), f"{test_case_name}: merged_embeddings_counts mismatch"
                assert (py_merged_attention_mask == cuda_merged_attention_mask).all(), f"{test_case_name}: merged_attention_mask mismatch"

    return

def test_cuda_kernel_merges_transform_backward():
    batch_size = 7
    seq_len = 13
    hidden_size = 16

    config_py = LlamaConfig(hidden_size=256, num_hidden_layers=2, attn_implementation='eager', generate_merges_transform_impl="python")
    config_cuda_kernel = LlamaConfig(hidden_size=256, num_hidden_layers=2, attn_implementation='eager', generate_merges_transform_impl="cuda_kernel")

    py_adaptive_fan_in_gumbel = AdaptiveFanInGumbel(config_py)
    cuda_adaptive_fan_in_gumbel = AdaptiveFanInGumbel(config_cuda_kernel)
    
    cuda_adaptive_fan_in_gumbel.fan_in_mlp.weight.data.copy_(py_adaptive_fan_in_gumbel.fan_in_mlp.weight.data)
    
    if py_adaptive_fan_in_gumbel.fan_in_mlp.bias is not None and cuda_adaptive_fan_in_gumbel.fan_in_mlp.bias is not None:
        cuda_adaptive_fan_in_gumbel.fan_in_mlp.bias.data.copy_(py_adaptive_fan_in_gumbel.fan_in_mlp.bias.data)

        assert id(cuda_adaptive_fan_in_gumbel.fan_in_mlp.bias) != id(py_adaptive_fan_in_gumbel.fan_in_mlp.bias)
    
    assert id(cuda_adaptive_fan_in_gumbel.fan_in_mlp.weight) != id(py_adaptive_fan_in_gumbel.fan_in_mlp.weight)
    
    
    # cuda_merging_map = torch.zeros([batch_size, seq_len, 2], device='cuda')
    # cuda_merging_map[:, :, 0] = 1.
    # cuda_merging_map.requires_grad = True
    # cuda_kernel_cuda_merging_map = cuda_merging_map.clone()
    # python_merging_map = cuda_merging_map.clone()

    cuda_attention_mask = torch.ones([batch_size, seq_len], device='cuda', dtype=torch.bool)

    # special_embeddings_mask = torch.zeros_like(cuda_attention_mask)
    # special_embeddings_mask[:, 0] = 1
    # special_embeddings_mask[:, -1] = 1
    
    merging_map = torch.zeros([batch_size, seq_len, 2], device='cuda')
    merging_map[:, :, 0] = 1.
    merging_map[:, 2] = torch.tensor([0., 1.], device='cuda')
    merging_map.requires_grad = True
    
    cuda_merging_map = merging_map.clone()
    py_merging_map = merging_map.clone()
    

    # cuda kernel forward
    cuda_merged_embeddings_transform, cuda_merged_embeddings_counts, cuda_merged_attention_mask = cuda_adaptive_fan_in_gumbel.generate_merges_transform(cuda_merging_map, cuda_attention_mask)

    output_gradients = torch.rand_like(cuda_merged_embeddings_transform)
    output_gradients[~cuda_merged_embeddings_transform.bool()] = 0
    output_gradients_py = output_gradients.clone()
    output_gradients_cuda = output_gradients.clone()
    (input_gradients_cuda_kernel,) = torch.autograd.grad(cuda_merged_embeddings_transform, cuda_merging_map, grad_outputs=output_gradients_cuda)

    # py forward
    py_merged_embeddings_transform, py_merged_embeddings_counts, py_merged_attention_mask = py_adaptive_fan_in_gumbel.generate_merges_transform(py_merging_map, cuda_attention_mask)
    
    assert (py_merged_embeddings_transform == cuda_merged_embeddings_transform).all()
    assert (cuda_merged_embeddings_counts == py_merged_embeddings_counts).all()
    assert (cuda_merged_attention_mask == py_merged_attention_mask).all()

    (input_gradients_python,) = torch.autograd.grad(py_merged_embeddings_transform, py_merging_map, grad_outputs=output_gradients_py)

    assert (input_gradients_python == input_gradients_cuda_kernel).all()

    return


def test_cuda_kernel_fan_out_backward():
    batch_size = 7
    seq_len = 100
    hidden_size = 16
    
    device = 'cuda'

    config_py = LlamaConfig(hidden_size=hidden_size, num_hidden_layers=2, attn_implementation='eager', generate_merges_transform_impl="python")
    config_cuda_kernel = LlamaConfig(hidden_size=hidden_size, num_hidden_layers=2, attn_implementation='eager', generate_merges_transform_impl="cuda_kernel")
    
    adaptive_fan_in = AdaptiveFanInGumbel(config_py).to(device)

    py_adaptive_fan_out = AdaptiveFanOut(config_py).to(device)
    cuda_adaptive_fan_out = AdaptiveFanOut(config_cuda_kernel).to(device)
    cuda_adaptive_fan_out.load_state_dict(py_adaptive_fan_out.state_dict())
    
    hidden_states = torch.rand([ batch_size, seq_len, hidden_size ], requires_grad=True, device=device)
    attention_mask = torch.ones([batch_size, seq_len], device=device)
    special_embeddings_mask = torch.zeros([batch_size, seq_len], device=device)
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    adaptive_fan_in_output = adaptive_fan_in.forward(hidden_states, attention_mask, special_embeddings_mask)
    assert adaptive_fan_in_output.hidden_state.grad_fn is not None
    assert (adaptive_fan_in_output.merged_embeddings_counts > 1).any(), 'at least one token should be merged to be shure gradients calculation is correct'

    py_residual_hidden_states = hidden_states.detach()
    py_residual_hidden_states.requires_grad = True
    cuda_kernel_residual_hidden_states = hidden_states.detach()
    cuda_kernel_residual_hidden_states.requires_grad = True
    residual_attention_mask = attention_mask
    
    py_adaptive_fan_in_output_hidden_state = adaptive_fan_in_output.hidden_state.detach()
    py_adaptive_fan_in_output_hidden_state.requires_grad = True

    cuda_kernel_adaptive_fan_in_output_hidden_state = adaptive_fan_in_output.hidden_state.detach()
    cuda_kernel_adaptive_fan_in_output_hidden_state.requires_grad = True


    py_adaptive_fan_out_output = py_adaptive_fan_out.forward(
        hidden_states=py_adaptive_fan_in_output_hidden_state,
        attention_mask=adaptive_fan_in_output.attention_mask,
        merged_embeddings_counts=adaptive_fan_in_output.merged_embeddings_counts,
        residual_hidden_states=py_residual_hidden_states,
        residual_attention_mask=residual_attention_mask,
    )
    py_restored_hidden_states = py_adaptive_fan_out_output.hidden_state

    cuda_kernel_adaptive_fan_out_output = cuda_adaptive_fan_out.forward(
        hidden_states=cuda_kernel_adaptive_fan_in_output_hidden_state,
        attention_mask=adaptive_fan_in_output.attention_mask,
        merged_embeddings_counts=adaptive_fan_in_output.merged_embeddings_counts,
        residual_hidden_states=cuda_kernel_residual_hidden_states,
        residual_attention_mask=residual_attention_mask,
    )
    cuda_kernel__restored_hidden_states = cuda_kernel_adaptive_fan_out_output.hidden_state
    
    assert (py_restored_hidden_states == cuda_kernel__restored_hidden_states).all()
    
    output_gradients_py = torch.rand_like(cuda_kernel__restored_hidden_states)
    output_gradients_cuda = output_gradients_py.detach()

    (py_adaptive_fan_in_output_hidden_state_gradients, py_residual_hidden_states_gradients) = torch.autograd.grad(py_restored_hidden_states, (py_adaptive_fan_in_output_hidden_state, py_residual_hidden_states), grad_outputs=output_gradients_py)
    (cuda_kernel_adaptive_fan_in_output_hidden_state_gradients, cuda_kernel_residual_hidden_states_gradients) = torch.autograd.grad(cuda_kernel__restored_hidden_states, (cuda_kernel_adaptive_fan_in_output_hidden_state, cuda_kernel_residual_hidden_states), grad_outputs=output_gradients_cuda)

    assert (py_adaptive_fan_in_output_hidden_state_gradients == cuda_kernel_adaptive_fan_in_output_hidden_state_gradients).all()
    assert (py_residual_hidden_states_gradients == cuda_kernel_residual_hidden_states_gradients).all()

    return


