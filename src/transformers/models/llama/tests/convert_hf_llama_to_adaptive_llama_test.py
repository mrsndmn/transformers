import torch
import pytest
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM


@pytest.mark.parametrize("use_cache", [True, False])
def test_build_adaptive_llama_from_llama_checkpoint_no_pruning(use_cache):

    torch.set_default_device('cuda')

    llama_checkpoint = 'HuggingFaceTB/SmolLM-135M'
    pretrained_model = AutoModelForCausalLM.from_pretrained(llama_checkpoint, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)

    adaptive_model = build_adaptive_llama_from_llama_checkpoint(
        llama_checkpoint,
        dummy_adaptive_fan_in=[ True ] * pretrained_model.config.num_hidden_layers,
        fan_out_projection=True,
        merging_type=None,
        flash_attention=False,
    )

    adaptive_model.eval()
    pretrained_model.eval()

    inputs = torch.randint(0, tokenizer.vocab_size, (4, 256))

    pretrained_outputs = pretrained_model.forward(
        inputs.clone(),
        output_hidden_states=True,
        use_cache=use_cache
    )
    adaptive_outputs = adaptive_model.forward(
        inputs.clone(),
        output_hidden_states=True,
        use_cache=use_cache,
    )

    for i, (adaptive_hs, pretrained_hs) in enumerate(zip(adaptive_outputs['hidden_states'], pretrained_outputs['hidden_states'])):
        assert torch.allclose(adaptive_hs, pretrained_hs, atol=1e-4), f"{i} hidden state are not close"

    assert torch.allclose(adaptive_outputs['logits'], pretrained_outputs['logits'], atol=1e-4)

