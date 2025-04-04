import torch

from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint
from transformers import AutoModelForCausalLM

def test_build_adaptive_llama_from_llama_checkpoint_no_pruning():

    torch.set_default_device('cuda')

    llama_checkpoint = 'HuggingFaceTB/SmolLM-135M'
    pretrained_model = AutoModelForCausalLM.from_pretrained(llama_checkpoint)
    pretrained_model.to(torch.bfloat16)

    adaptive_model = build_adaptive_llama_from_llama_checkpoint(
        llama_checkpoint,
        dummy_adaptive_fan_in=[ True ] * pretrained_model.config.num_hidden_layers,
        fan_out_projection=True,
        merging_type=None,
        flash_attention=False,
    )
    adaptive_model.eval()
    pretrained_model.eval()
    
    bs = 3
    seq_len = 7
    
    inputs = torch.randint(0, pretrained_model.config.vocab_size, [ bs, seq_len ])
    
    special_embeddings_mask = torch.zeros_like(inputs)
    
    pretrained_outputs = pretrained_model.forward(
        inputs.clone(),
        output_hidden_states=True,
        use_cache=False
    )
    adaptive_outputs = adaptive_model.forward(
        inputs.clone(),
        special_embeddings_mask=special_embeddings_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    for i, (adaptive_hs, pretrained_hs) in enumerate(zip(adaptive_outputs['hidden_states'], pretrained_outputs['hidden_states'])):
        assert torch.allclose(adaptive_hs, pretrained_hs, atol=1e-4), f"{i} hidden state are not close"

    assert torch.allclose(adaptive_outputs['logits'], pretrained_outputs['logits'], atol=1e-4)
    
    
def test_build_adaptive_llama_from_llama_checkpoint_pruning():
    torch.set_default_device('cuda')

    llama_checkpoint = 'HuggingFaceTB/SmolLM-135M'
    pretrained_model = AutoModelForCausalLM.from_pretrained(llama_checkpoint, torch_dtype=torch.bfloat16)

    dummy_adaptive_fan_in = [ True ] * (pretrained_model.config.num_hidden_layers // 2)
    dummy_adaptive_fan_in[0] = False
    adaptive_model = build_adaptive_llama_from_llama_checkpoint(
        llama_checkpoint,
        dummy_adaptive_fan_in=dummy_adaptive_fan_in,
        generate_merges_transform_impl='cuda_kernel',
        fan_out_projection=True,
        merging_type='no_merging',
        freeze_lm_backbone=True,
        flash_attention=False,
    )
    
    device = 'cuda'
    
    adaptive_model.eval()
    adaptive_model.to(device)
    pretrained_model.eval()
    pretrained_model.to(device)

    bs = 3
    seq_len = 7

    input_ids = torch.randint(0, pretrained_model.config.vocab_size, [ bs, seq_len ], device=device)

    special_embeddings_mask = torch.zeros_like(input_ids, device=device)
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    attention_mask = torch.ones_like(input_ids)

    pretrained_outputs = pretrained_model.forward(
        input_ids.clone(),
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    adaptive_outputs = adaptive_model.forward(
        input_ids.clone(),
        attention_mask=attention_mask,
        special_embeddings_mask=special_embeddings_mask,
        output_hidden_states=True,
    )

    assert (adaptive_outputs['logits'] == pretrained_outputs['logits']).all()
