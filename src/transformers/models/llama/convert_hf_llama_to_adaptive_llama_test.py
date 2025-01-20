import torch

from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint
from transformers import AutoModelForCausalLM

def test_build_adaptive_llama_from_llama_checkpoint_no_pruning():
    
    llama_checkpoint = 'HuggingFaceTB/SmolLM-135M'
    pretrained_model = AutoModelForCausalLM.from_pretrained(llama_checkpoint)
    pretrained_model.to(torch.bfloat16)

    adaptive_model = build_adaptive_llama_from_llama_checkpoint(
        llama_checkpoint,
        dummy_adaptive_fan_in=[ True ] * pretrained_model.config.num_hidden_layers,
        generate_merges_transform_impl='python',
        fan_out_projection=True,
        merging_type=None,
        freeze_lm_backbone=True,
        flash_attention=False,
    )
    adaptive_model.eval()
    pretrained_model.eval()
    
    bs = 3
    seq_len = 7
    
    inputs = torch.randint(0, pretrained_model.config.vocab_size, [ bs, seq_len ])
    
    special_embeddings_mask = torch.zeros_like(inputs)
    
    pretrained_outputs = pretrained_model.forward(inputs.clone(), output_hidden_states=True)
    adaptive_outputs = adaptive_model.forward(
        inputs.clone(),
        special_embeddings_mask=special_embeddings_mask,
        output_hidden_states=True,
    )
    
    assert (adaptive_outputs['logits'] == pretrained_outputs['logits']).all()
    
    
def test_build_adaptive_llama_from_llama_checkpoint_no_pruning():

    llama_checkpoint = 'HuggingFaceTB/SmolLM-135M'
    pretrained_model = AutoModelForCausalLM.from_pretrained(llama_checkpoint, torch_dtype=torch.bfloat16)
    # pretrained_model.to(torch.bfloat16)

    adaptive_model = build_adaptive_llama_from_llama_checkpoint(
        llama_checkpoint,
        dummy_adaptive_fan_in=[ True ] * pretrained_model.config.num_hidden_layers,
        generate_merges_transform_impl='python',
        fan_out_projection=True,
        merging_type=None,
        freeze_lm_backbone=True,
        flash_attention=False,
    )
    adaptive_model.eval()
    pretrained_model.eval()
    
    bs = 3
    seq_len = 7
    
    inputs = torch.randint(0, pretrained_model.config.vocab_size, [ bs, seq_len ])
    
    special_embeddings_mask = torch.zeros_like(inputs)
    
    pretrained_outputs = pretrained_model.forward(inputs.clone(), output_hidden_states=True)
    adaptive_outputs = adaptive_model.forward(
        inputs.clone(),
        special_embeddings_mask=special_embeddings_mask,
        output_hidden_states=True,
    )
    
    assert (adaptive_outputs['logits'] == pretrained_outputs['logits']).all()
    