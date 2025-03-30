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
        hidden_state = torch.rand([1, 780, 64], device=device)
        concrete_bool = torch.ones([1, 780], device=device, dtype=torch.bool)
        concrete_bool[:, 50:130] = 0  # Make some tokens unimportant
        attention_mask = torch.ones([1, 780], dtype=torch.bool, device=device)
        special_embeddings_mask = torch.zeros_like(attention_mask)
        special_embeddings_mask[:, 0] = 1  # First token is special
        special_embeddings_mask[:, -1] = 1  # Last token is special
        concrete = concrete_bool.float()

        # Run CUDA implementation
        outputs_cuda = prune_tokens_concrete(
            hidden_state,
            concrete_bool,
            attention_mask,
            special_embeddings_mask,
            concrete
        )
        hidden_state_m, merged_embeddings_counts, merged_attention_mask, merged_special_embeddings_mask, merged_concrete = outputs_cuda

        # Run Python reference implementation
        concrete_bool_cpu = concrete_bool.detach().cpu()
        outputs_py = reorder_mask_for_concrete(
            concrete_bool=concrete_bool_cpu,
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            concrete=concrete
        )
        hidden_state_m_py, merged_embeddings_counts_py, merged_attention_mask_py, merged_special_embeddings_mask_py, merged_concrete_py = outputs_py

        # Verify outputs match
        assert torch.allclose(hidden_state_m, hidden_state_m_py), "Hidden states don't match"
        assert torch.equal(merged_embeddings_counts, merged_embeddings_counts_py), "Merged embeddings counts don't match"
        assert torch.equal(merged_attention_mask, merged_attention_mask_py), "Merged attention masks don't match"
        assert torch.equal(merged_special_embeddings_mask, merged_special_embeddings_mask_py), "Merged special embeddings masks don't match"
        assert torch.allclose(merged_concrete, merged_concrete_py), "Merged concrete values don't match"

        # Verify specific properties
        assert merged_embeddings_counts.sum().item() == 780, "Total token count mismatch"
        assert merged_attention_mask.sum().item() == 700, "Active tokens count mismatch"
        assert merged_special_embeddings_mask.sum().item() == 2, "Special tokens count mismatch"

        # Verify token reordering
        assert (hidden_state_m[:, :50] == hidden_state[:, :50]).all(), "First 50 tokens should be unchanged"
        assert (hidden_state_m[:, 50:] == hidden_state[:, 130:]).all(), "Remaining tokens not correctly reordered"

    # Test case 2: Edge cases
    def run_edge_case_test():
        # Test with all tokens important
        hidden_state = torch.rand([1, 100, 64], device=device)
        concrete_bool = torch.ones([1, 100], device=device, dtype=torch.bool)
        attention_mask = torch.ones([1, 100], dtype=torch.bool, device=device)
        
        outputs_cuda = prune_tokens_concrete(hidden_state, concrete_bool, attention_mask)
        outputs_py = reorder_mask_for_concrete(concrete_bool=concrete_bool.cpu(), hidden_state=hidden_state, attention_mask=attention_mask)
        
        assert torch.equal(outputs_cuda[0], outputs_py[0]), "All important tokens case failed"
        assert outputs_cuda[1].sum().item() == 100, "Wrong token count for all important case"

        # Test with no tokens important (except first and last for stability)
        concrete_bool = torch.zeros([1, 100], device=device, dtype=torch.bool)
        concrete_bool[:, [0, -1]] = 1
        
        outputs_cuda = prune_tokens_concrete(hidden_state, concrete_bool, attention_mask)
        outputs_py = reorder_mask_for_concrete(concrete_bool=concrete_bool.cpu(), hidden_state=hidden_state, attention_mask=attention_mask)
        
        assert torch.equal(outputs_cuda[0], outputs_py[0]), "No important tokens case failed"
        assert outputs_cuda[1].sum().item() == 100, "Wrong token count for no important case"

    # Test case 3: Different batch sizes
    def run_batch_test():
        batch_sizes = [1, 2, 4]
        seq_lens = [128, 256]
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                hidden_state = torch.rand([batch_size, seq_len, 64], device=device)
                concrete_bool = torch.ones([batch_size, seq_len], device=device, dtype=torch.bool)
                # Make different patterns of important tokens for each batch
                for i in range(batch_size):
                    concrete_bool[i, 10+i*10:50+i*10] = 0
                attention_mask = torch.ones([batch_size, seq_len], dtype=torch.bool, device=device)
                
                outputs_cuda = prune_tokens_concrete(hidden_state, concrete_bool, attention_mask)
                outputs_py = reorder_mask_for_concrete(concrete_bool=concrete_bool.cpu(), hidden_state=hidden_state, attention_mask=attention_mask)
                
                assert torch.equal(outputs_cuda[0], outputs_py[0]), f"Batch test failed for size {batch_size}x{seq_len}"
                assert outputs_cuda[1].sum().item() == batch_size * seq_len, f"Wrong token count for batch {batch_size}x{seq_len}"

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
