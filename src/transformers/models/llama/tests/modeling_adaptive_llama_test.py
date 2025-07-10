import time

import pytest

import torch

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_adaptive_llama import AdaptiveFanInHCG, AdaptiveFanOutHCG, AdaptiveFanInOutput, AdaptiveFanOutOutput, AdaptiveLlamaModel


def test_adaptive_fan_in_all_merge():
    config = LlamaConfig(hidden_size=256, num_hidden_layers=2, generate_merges_transform_impl='python')

    torch.set_default_device('cuda')

    adaptive_fan_in = AdaptiveFanInHCG(config)
    adaptive_fan_in.eval()
    batch_size, seq_len = 3, 6
    input_ids = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ])
    attention_mask = torch.ones([batch_size, seq_len], dtype=torch.long)
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    adaptive_fan_in_output = adaptive_fan_in.forward(input_ids=input_ids, hidden_state=hidden_states, attention_mask=attention_mask, special_embeddings_mask=special_embeddings_mask)

    assert adaptive_fan_in_output.attention_mask.shape[1] == seq_len # bos + merged_embedding + eos
    assert adaptive_fan_in_output.hidden_state.shape[1] == seq_len # bos + merged_embedding + eos
    assert adaptive_fan_in_output.merged_embeddings_counts.shape[1] == seq_len # bos + merged_embedding + eos
    assert adaptive_fan_in_output.special_embeddings_mask.shape[1] == seq_len # bos + merged_embedding + eos

    return


def test_adaptive_fan_out():
    config = LlamaConfig(hidden_size=256, num_hidden_layers=2)

    torch.set_default_device('cuda')

    adaptive_fan_out = AdaptiveFanOutHCG(config)
    adaptive_fan_out.eval()

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

    torch.set_default_device('cuda')

    adaptive_fan_in = AdaptiveFanInHCG(config)
    adaptive_fan_out = AdaptiveFanOutHCG(config)

    batch_size, seq_len = 3, 6
    input_ids = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ], requires_grad=True)
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    adaptive_fan_in_output = adaptive_fan_in.forward(input_ids=input_ids, hidden_state=hidden_states, attention_mask=attention_mask, special_embeddings_mask=special_embeddings_mask)
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


def test_cuda_kernel_fan_out_backward():

    batch_size = 7
    seq_len = 100
    hidden_size = 16

    device = 'cuda'

    config_py = LlamaConfig(hidden_size=hidden_size, num_hidden_layers=2, attn_implementation='flash_attention_2', generate_merges_transform_impl="python", merging_type='hcg')
    config_cuda_kernel = LlamaConfig(hidden_size=hidden_size, num_hidden_layers=2, attn_implementation='flash_attention_2', generate_merges_transform_impl="cuda_kernel", merging_type='hcg')

    current_dtype = torch.bfloat16
    torch.set_default_dtype(current_dtype)
    adaptive_fan_in = AdaptiveFanInHCG(config_py).to(device)

    py_adaptive_fan_out = AdaptiveFanOutHCG(config_py).to(device)
    cuda_adaptive_fan_out = AdaptiveFanOutHCG(config_cuda_kernel).to(device)
    cuda_adaptive_fan_out.load_state_dict(py_adaptive_fan_out.state_dict())


    hidden_states = torch.rand([ batch_size, seq_len, hidden_size ], requires_grad=True, device=device)
    input_ids = torch.randint(0, 10, (batch_size, seq_len), device=device)
    attention_mask = torch.ones([batch_size, seq_len], device=device, dtype=torch.long)
    special_embeddings_mask = torch.zeros([batch_size, seq_len], device=device)
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    torch.set_default_dtype(torch.float32)

    adaptive_fan_in_output = adaptive_fan_in.forward(input_ids=input_ids, hidden_state=hidden_states, attention_mask=attention_mask, special_embeddings_mask=special_embeddings_mask)
    assert adaptive_fan_in_output.hidden_state.grad_fn is not None
    # assert (adaptive_fan_in_output.merged_embeddings_counts > 1).any(), 'at least one token should be merged to be shure gradients calculation is correct'

    # assert ((adaptive_fan_in_output.merging_map == 0) > 0).any(), 'at least one token should be merged to be shure gradients calculation is correct'

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
        merged_embeddings_counts=adaptive_fan_in_output.merged_embeddings_counts.long(),
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


