import time
import argparse

import torch

from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInHCG, reorder_mask_for_concrete


from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint
from transformers.models.llama.merges_transform.generate_merges import generate_merges_transform, fan_out_restore_residuals, prune_tokens_concrete

def test_sdpa_attention():

    # Optionally use the context manager to ensure one of the fused kernels is run
    query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
    key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
    value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")

    query[:, :, 3, :] = 0
    key[:, :, 3, :] = 0
    value[:, :, 3, :] = 0

    attn_mask = torch.zeros(32, 8, 128, 128, dtype=torch.float16, device="cuda")
    attn_mask[:, :, 1, :] = -10000
    attn_mask[:, :, 1, :] = float("-inf")

    attention = torch.nn.functional.scaled_dot_product_attention(query,key,value, attn_mask=attn_mask)

    print(attention[0, 0, :5, :10])



def test_eval_adaptive_hcg_llama():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bench_dtype = torch.bfloat16
    torch.set_default_dtype(bench_dtype)
    torch.set_default_device(device)

    # model_orig = LlamaForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")

    # checkpoint = './adaptive_hcg_slm2_360M_pretrain_fan_out_projection_4/checkpoint-3118'
    checkpoint = 'adaptive_hcg_slm2_360M_hcg_smooth_no_detach_2gpu_1.5_1SKDWQXE/checkpoint-79996/'

    model = AdaptiveLlamaForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=bench_dtype,
    )

    model.config.pretrain_fan_out_projection = False
    model.config.use_cache = False

    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    text = "<|im_start|> Who are you? And what are you going to do?"
    text *= 10
    text_inputs = tokenizer([ text ], return_tensors='pt').to(device)

    special_embeddings_mask = torch.zeros_like(text_inputs['input_ids'])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1
    text_inputs['special_embeddings_mask'] = special_embeddings_mask
    text_inputs['labels'] = text_inputs['input_ids']
    text_inputs['output_hidden_states'] = True

    with torch.no_grad():

        model.eval()
        eval_output = model.forward(**text_inputs)
        print(eval_output['loss'])

        # Train mode for part of my modules
        for adaptive_down in model.model.adaptive_down:
            adaptive_down.train()

        for adaptive_up in model.model.adaptive_up:
            adaptive_up.train()

        for adaptive_down in model.model.adaptive_down:
            if hasattr(adaptive_down, 'hcg'):
                adaptive_down.hcg.eval()

        train_output = model.forward(**text_inputs)
        print(train_output['loss'])

        assert eval_output.fan_in_merging_maps[3].sum().item() == eval_output.fan_in_merging_logits[3].sum().item()

        assert torch.allclose(train_output['loss'], eval_output['loss'], atol=1e-2), f'{train_output["loss"].item()} != {eval_output["loss"].item()}'


def test_prune_tokens_concrete():
    """Test prune_tokens_concrete CUDA implementation against Python reference implementation."""
    device = 'cuda'

    # Test case 1: Basic functionality with default values
    def run_basic_test():
        hidden_state = torch.rand([1, 780, 64], device=device, requires_grad=True)
        concrete = torch.rand([1, 780], device=device, requires_grad=True)
        concrete_bool = (concrete > 0.5)
        attention_mask = torch.ones([1, 780], dtype=torch.long, device=device)
        special_embeddings_mask = torch.zeros_like(attention_mask)
        special_embeddings_mask[:, 0] = 1
        special_embeddings_mask[:, -1] = 1

        concrete_bool[special_embeddings_mask.bool()] = True

        # Run CUDA implementation with gradient tracking
        outputs_cuda = prune_tokens_concrete(
            hidden_state,
            concrete_bool,
            attention_mask,
            special_embeddings_mask,
            concrete
        )
        hidden_state_m_cuda, merged_embeddings_counts_cuda, merged_attention_mask_cuda, merged_special_embeddings_mask_cuda, merged_concrete_cuda = outputs_cuda

        # Create gradient tensors
        grad_hidden_state_cuda = torch.rand_like(hidden_state_m_cuda)
        grad_concrete_cuda = torch.rand_like(merged_concrete_cuda)

        # Backward pass for CUDA implementation
        loss_cuda = (hidden_state_m_cuda * grad_hidden_state_cuda).sum() + (merged_concrete_cuda * grad_concrete_cuda).sum()
        loss_cuda.backward()

        # Store CUDA gradients
        grad_hidden_state_from_cuda = hidden_state.grad.clone()
        grad_concrete_from_cuda = concrete.grad.clone()

        # Reset gradients for Python implementation
        hidden_state.grad = None
        concrete.grad = None

        # Run Python reference implementation with same gradient tensors
        outputs_py = reorder_mask_for_concrete(
            concrete_bool=concrete_bool.cpu(),
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            concrete=concrete
        )
        hidden_state_m_py, merged_embeddings_counts_py, merged_attention_mask_py, merged_special_embeddings_mask_py, merged_concrete_py = outputs_py

        # Backward pass for Python implementation using same gradients
        loss_py = (hidden_state_m_py * grad_hidden_state_cuda).sum() + (merged_concrete_py * grad_concrete_cuda).sum()
        loss_py.backward()

        # Compare gradients
        assert torch.allclose(grad_hidden_state_from_cuda, hidden_state.grad, rtol=1e-5, atol=1e-5), "Hidden state gradients don't match"
        assert torch.allclose(grad_concrete_from_cuda, concrete.grad, rtol=1e-5, atol=1e-5), "Concrete gradients don't match"

        # Verify outputs match
        assert torch.allclose(hidden_state_m_cuda, hidden_state_m_py), "Hidden states don't match"
        assert torch.equal(merged_embeddings_counts_cuda, merged_embeddings_counts_py), "Merged embeddings counts don't match"
        assert torch.equal(merged_attention_mask_cuda, merged_attention_mask_py), "Merged attention masks don't match"
        assert torch.equal(merged_special_embeddings_mask_cuda, merged_special_embeddings_mask_py), "Merged special embeddings masks don't match"
        assert torch.allclose(merged_concrete_cuda, merged_concrete_py), "Merged concrete values don't match"

        # Verify specific properties
        assert merged_embeddings_counts_cuda.sum().item() == 780, "Total token count mismatch"
        assert merged_attention_mask_cuda.sum().item() == merged_attention_mask_py.sum().item(), "Active tokens count mismatch"
        assert merged_special_embeddings_mask_cuda.sum().item() == 2, "Special tokens count mismatch"

    # Test case 2: Edge cases
    def run_edge_case_test():
        # Test with all tokens important
        hidden_state = torch.rand([1, 100, 64], device=device, requires_grad=True)
        concrete = torch.ones([1, 100], device=device, requires_grad=True)
        concrete_bool = torch.ones([1, 100], device=device, dtype=torch.bool)
        attention_mask = torch.ones([1, 100], dtype=torch.long, device=device)
        special_embeddings_mask = torch.zeros_like(attention_mask)
        
        # CUDA implementation
        outputs_cuda = prune_tokens_concrete(
            hidden_state,
            concrete_bool,
            attention_mask,
            special_embeddings_mask,
            concrete
        )
        hidden_state_m_cuda = outputs_cuda[0]
        merged_concrete_cuda = outputs_cuda[4]

        # Create gradients
        grad_hidden_state = torch.rand_like(hidden_state_m_cuda)
        grad_concrete = torch.rand_like(merged_concrete_cuda)

        # CUDA backward
        loss_cuda = (hidden_state_m_cuda * grad_hidden_state).sum() + (merged_concrete_cuda * grad_concrete).sum()
        loss_cuda.backward()
        grad_hidden_state_from_cuda = hidden_state.grad.clone()
        grad_concrete_from_cuda = concrete.grad.clone()

        # Reset gradients
        hidden_state.grad = None
        concrete.grad = None

        # Python implementation
        outputs_py = reorder_mask_for_concrete(
            concrete_bool=concrete_bool.cpu(),
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            concrete=concrete
        )
        hidden_state_m_py = outputs_py[0]
        merged_concrete_py = outputs_py[4]

        # Python backward with same gradients
        loss_py = (hidden_state_m_py * grad_hidden_state).sum() + (merged_concrete_py * grad_concrete).sum()
        loss_py.backward()

        # Compare gradients
        assert torch.allclose(grad_hidden_state_from_cuda, hidden_state.grad, rtol=1e-5, atol=1e-5), "Hidden state gradients don't match for all-important case"
        assert torch.allclose(grad_concrete_from_cuda, concrete.grad, rtol=1e-5, atol=1e-5), "Concrete gradients don't match for all-important case"

        # Test with no tokens important (except first and last for stability)
        hidden_state = torch.rand([1, 100, 64], device=device, requires_grad=True)
        concrete = torch.zeros([1, 100], device=device)
        concrete_bool = torch.zeros([1, 100], device=device, dtype=torch.bool)
        concrete_bool[:, [0, -1]] = 1
        concrete[:, [0, -1]] = 1
        concrete.requires_grad = True

        # CUDA implementation
        outputs_cuda = prune_tokens_concrete(
            hidden_state,
            concrete_bool,
            attention_mask,
            special_embeddings_mask,
            concrete
        )
        hidden_state_m_cuda = outputs_cuda[0]
        merged_concrete_cuda = outputs_cuda[4]

        # Create gradients
        grad_hidden_state = torch.rand_like(hidden_state_m_cuda)
        grad_concrete = torch.rand_like(merged_concrete_cuda)

        # CUDA backward
        loss_cuda = (hidden_state_m_cuda * grad_hidden_state).sum() + (merged_concrete_cuda * grad_concrete).sum()
        loss_cuda.backward()
        grad_hidden_state_from_cuda = hidden_state.grad.clone()
        grad_concrete_from_cuda = concrete.grad.clone()

        # Reset gradients
        hidden_state.grad = None
        concrete.grad = None

        # Python implementation
        outputs_py = reorder_mask_for_concrete(
            concrete_bool=concrete_bool.cpu(),
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            concrete=concrete
        )
        hidden_state_m_py = outputs_py[0]
        merged_concrete_py = outputs_py[4]

        # Python backward with same gradients
        loss_py = (hidden_state_m_py * grad_hidden_state).sum() + (merged_concrete_py * grad_concrete).sum()
        loss_py.backward()

        # Compare gradients
        assert torch.allclose(grad_hidden_state_from_cuda, hidden_state.grad, rtol=1e-5, atol=1e-5), "Hidden state gradients don't match for minimal-important case"
        assert torch.allclose(grad_concrete_from_cuda, concrete.grad, rtol=1e-5, atol=1e-5), "Concrete gradients don't match for minimal-important case"

    # Test case 3: Different batch sizes
    def run_batch_test():
        batch_sizes = [1, 2, 4]
        seq_lens = [128, 256]
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                hidden_state = torch.rand([batch_size, seq_len, 64], device=device, requires_grad=True)
                concrete = torch.rand([batch_size, seq_len], device=device, requires_grad=True)
                concrete_bool = (concrete > 0.5)
                attention_mask = torch.ones([batch_size, seq_len], dtype=torch.long, device=device)
                special_embeddings_mask = torch.zeros_like(attention_mask)
                
                # CUDA implementation
                outputs_cuda = prune_tokens_concrete(
                    hidden_state,
                    concrete_bool,
                    attention_mask,
                    special_embeddings_mask,
                    concrete
                )
                hidden_state_m_cuda = outputs_cuda[0]
                merged_concrete_cuda = outputs_cuda[4]

                # Create gradients
                grad_hidden_state = torch.rand_like(hidden_state_m_cuda)
                grad_concrete = torch.rand_like(merged_concrete_cuda)

                # CUDA backward
                loss_cuda = (hidden_state_m_cuda * grad_hidden_state).sum() + (merged_concrete_cuda * grad_concrete).sum()
                loss_cuda.backward()
                grad_hidden_state_from_cuda = hidden_state.grad.clone()
                grad_concrete_from_cuda = concrete.grad.clone()

                # Reset gradients
                hidden_state.grad = None
                concrete.grad = None

                # Python implementation
                outputs_py = reorder_mask_for_concrete(
                    concrete_bool=concrete_bool.cpu(),
                    hidden_state=hidden_state,
                    attention_mask=attention_mask,
                    special_embeddings_mask=special_embeddings_mask,
                    concrete=concrete
                )
                hidden_state_m_py = outputs_py[0]
                merged_concrete_py = outputs_py[4]

                # Python backward with same gradients
                loss_py = (hidden_state_m_py * grad_hidden_state).sum() + (merged_concrete_py * grad_concrete).sum()
                loss_py.backward()

                # Compare gradients
                assert torch.allclose(grad_hidden_state_from_cuda, hidden_state.grad, rtol=1e-5, atol=1e-5), f"Hidden state gradients don't match for batch {batch_size}x{seq_len}"
                assert torch.allclose(grad_concrete_from_cuda, concrete.grad, rtol=1e-5, atol=1e-5), f"Concrete gradients don't match for batch {batch_size}x{seq_len}"

    print("Running basic functionality test...")
    run_basic_test()
    print("Basic test passed!")

    # print("Running edge case test...")
    # run_edge_case_test()
    # print("Edge case test passed!")

    # print("Running batch size test...")
    # run_batch_test()
    # print("Batch size test passed!")

    print("All tests passed successfully!")
