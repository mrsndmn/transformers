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

    device = 'cuda'
    hidden_state = torch.rand([1, 780, 64], device=device)
    concrete_bool = torch.ones([ 1, 780 ], device=device, dtype=torch.bool)
    concrete_bool[:, 50:130] = 0
    attention_mask = torch.ones([ 1, 780 ], dtype=torch.bool, device=device)

    hidden_state_m, merged_embeddings_counts, merged_attention_mask = prune_tokens_concrete(
        hidden_state,
        concrete_bool,
        attention_mask,
    )

    concrete_bool_cpu = concrete_bool.detach().cpu().unsqueeze(-1)
    hidden_state_m_py, merged_embeddings_counts_py, merged_attention_mask_py = reorder_mask_for_concrete(concrete=concrete_bool_cpu, hidden_state=hidden_state, attention_mask=attention_mask)

    assert (hidden_state_m[:, :50] == hidden_state[:, :50]).all()
    assert (hidden_state_m[:, 50:] == hidden_state[:, 130:]).all()

    assert merged_embeddings_counts.sum().item() == 780
    assert merged_attention_mask.sum().item() == 700

    assert (hidden_state_m_py == hidden_state_m).all()
    assert (merged_embeddings_counts_py == merged_embeddings_counts).all()
    assert (merged_attention_mask_py == merged_attention_mask).all()

    breakpoint()
