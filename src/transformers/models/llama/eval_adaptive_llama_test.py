import time
import argparse

import torch

from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInHCG, reorder_mask_for_concrete


from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint
from transformers.models.llama.merges_transform.generate_merges import generate_merges_transform, fan_out_restore_residuals, prune_tokens_concrete

def test_eval_adaptive_hcg_llama():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bench_dtype = torch.bfloat16
    torch.set_default_dtype(bench_dtype)
    torch.set_default_device(device)

    model_orig = LlamaForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")

    # checkpoint = './adaptive_gumbel_8/checkpoint-996/'
    checkpoint = './adaptive_hcg_8_maintain_loss_nofanoutproj/checkpoint-24996'

    model = AdaptiveLlamaForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=bench_dtype,
    )

    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    text = "<|im_start|> Who are you? And what are you going to do?"
    text *= 60
    text_inputs = tokenizer([ text ], return_tensors='pt').to(device)

    special_embeddings_mask = torch.zeros_like(text_inputs['input_ids'])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1
    text_inputs['special_embeddings_mask'] = special_embeddings_mask
    text_inputs['labels'] = text_inputs['input_ids']

    with torch.no_grad():

        model.eval()
        eval_output = model.forward(**text_inputs)
        print(eval_output['loss'])

        model.train()
        train_output = model.forward(**text_inputs)
        print(train_output['loss'])

        assert train_output['loss'].item() == eval_output['loss'].item()

        del text_inputs['special_embeddings_mask']
        orig_output = model_orig.forward(**text_inputs)
        print("orig_output", orig_output['loss'])

    breakpoint()


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
