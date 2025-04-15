import torch
import pytest
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
from transformers.models.llama.train_adaptive_llama import freeze_lm_backbone

@pytest.mark.parametrize("use_cache", [True, False])
def test_build_adaptive_llama_use_cache(use_cache):

    torch.set_default_device('cuda')

    llama_checkpoint = 'HuggingFaceTB/SmolLM-135M'
    pretrained_model = AutoModelForCausalLM.from_pretrained(llama_checkpoint, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)

    dummy_adaptive_fan_in=[ True ] * (pretrained_model.config.num_hidden_layers // 2)
    dummy_adaptive_fan_in[-1] = False
    adaptive_model = build_adaptive_llama_from_llama_checkpoint(
        llama_checkpoint,
        dummy_adaptive_fan_in=dummy_adaptive_fan_in,
        hcg_log_a=10000.0,
        flash_attention=True,
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


def test_pretrained_checkpoint_perplexity():
    # Broken for new hcg in discrete tokens
    return

    torch.set_default_device('cuda')

    pretrained_checkpoint = "./adaptive_hcg_slm2_360M_pretrain_fan_out_projection_4/_no_fout_proj_checkpoint-3118/"
    adaptive_model = AdaptiveLlamaForCausalLM.from_pretrained(pretrained_checkpoint, torch_dtype=torch.bfloat16)

    adaptive_model.eval()

    inputs = torch.load("prepared_batch.input_ids.pt")

    pretrained_outputs = adaptive_model.forward(
        inputs.clone(),
        labels=inputs,
        output_hidden_states=True,
        use_cache=False
    )
    print("pretrained_outputs.loss", pretrained_outputs.loss)
    assert pretrained_outputs.loss < 3
    assert (pretrained_outputs.fan_in_merging_maps[3] == 1).all()

def test_finetuned_checkpoint_perplexity():
    # TODO
    pass


def test_pretrained_mostly_pruned_backward_grads():

    return

    torch.set_default_device('cuda')

    pretrained_checkpoint = "./adaptive_hcg_slm2_360M_dynamic_test_freeze_lm/checkpoint-40000/"
    adaptive_model = AdaptiveLlamaForCausalLM.from_pretrained(pretrained_checkpoint, torch_dtype=torch.bfloat16)

    freeze_lm_backbone(adaptive_model)

    adaptive_model.train()

    inputs = torch.load("prepared_batch.input_ids.pt")

    pretrained_outputs = adaptive_model.forward(
        inputs.clone(),
        labels=inputs,
        output_hidden_states=True,
        use_cache=False
    )

    pretrained_outputs.loss.backward()

    print("pretrained_outputs.loss", pretrained_outputs.loss)
    assert pretrained_outputs.loss < 3
    assert (pretrained_outputs.fan_in_merging_maps[3] == 1).all()
