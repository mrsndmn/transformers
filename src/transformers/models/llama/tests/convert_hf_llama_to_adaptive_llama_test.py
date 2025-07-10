import torch
import pytest
from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM
# from transformers.models.llama.train_adaptive_llama import freeze_lm_backbone

@pytest.mark.parametrize("use_cache", [True, False])
def test_build_adaptive_llama_use_cache(use_cache):

    torch.set_default_device('cuda')

    llama_checkpoint = 'HuggingFaceTB/SmolLM2-135M'
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

    torch.set_default_device('cuda')

    # pretrained_checkpoint = "adaptive_hcg_slm2_1.7B_w_0.010_l_8_no_self_attn_XFLDZI8W/checkpoint-249974"
    pretrained_checkpoint = "adaptive_hcg_llama31_8B_w_1.000_l_14_IHHIQR0I/checkpoint-90000"
    adaptive_model = AdaptiveLlamaForCausalLM.from_pretrained(pretrained_checkpoint, torch_dtype=torch.float32)
    # adaptive_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B", torch_dtype=torch.float32)

    adaptive_model.model.fan_in_idx = 16
    adaptive_model.model.fan_out_idx = 16

    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)

    adaptive_model.eval()

    tokenizer_output = tokenizer([ 'Question: Which Lloyd Webber musical premiered in the US on 10th December 1993?\nAnswer: Jurassic Earth', ], return_tensors='pt', padding=True)
    input_ids = tokenizer_output['input_ids']

    past_key_values = DynamicCache()

    model_kwargs = {
        'attention_mask': tokenizer_output['attention_mask'],
        'use_cache': True,
        'past_key_values': past_key_values,
    }

    # Prefill
    model_kwargs = adaptive_model._get_initial_cache_position(input_ids, model_kwargs)

    model_inputs = adaptive_model.prepare_inputs_for_generation(input_ids, **model_kwargs)

    pretrained_outputs = adaptive_model(
        **model_inputs,
    )

    model_kwargs = adaptive_model._update_model_kwargs_for_generation(
        pretrained_outputs,
        model_kwargs,
        is_encoder_decoder=adaptive_model.config.is_encoder_decoder,
    )

    next_token_logits = pretrained_outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32)
    next_tokens = torch.argmax(next_token_logits, dim=-1)

    input_ids_new = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

    model_kwargs = adaptive_model.prepare_inputs_for_generation(input_ids_new, **model_kwargs)

    # print("position_ids", model_kwargs['position_ids'])
    # breakpoint()

    # Decode step
    next_token_forward_outputs = adaptive_model(
        **model_kwargs,
        output_hidden_states=True,
    )

    pretrained_outputs_no_cache = adaptive_model(
        input_ids=input_ids_new,
        labels=input_ids_new,
        use_cache=False,
        output_hidden_states=True,
    )

    for i in range(len(next_token_forward_outputs.hidden_states)):
        i_no_cache = i
        # if adaptive_model.model.fan_in_idx <= i_no_cache < adaptive_model.model.fan_out_idx:
        #     i_no_cache += 16-7

        print(f"{i} l1 diff norm", (next_token_forward_outputs.hidden_states[i] - pretrained_outputs_no_cache.hidden_states[i_no_cache][:, -1, :]).norm(1, dim=-1))
        assert torch.allclose(next_token_forward_outputs.hidden_states[i], pretrained_outputs_no_cache.hidden_states[i_no_cache][:, -1, :], atol=1e-5), f'hidden state {i} match'

    assert torch.allclose(next_token_forward_outputs.logits[:, -1], pretrained_outputs_no_cache.logits[:, -1], atol=1e-4), 'pretrained outputs loss match'

    print("pretrained_outputs.loss", pretrained_outputs_no_cache.loss)
    assert pretrained_outputs_no_cache.loss < 4
    # assert (pretrained_outputs_no_cache.fan_in_merging_maps[7] == 1).all()

def test_finetuned_checkpoint_perplexity():
    # TODO
    pass


def test_pretrained_mostly_pruned_backward_grads():

    return

    torch.set_default_device('cuda')

    pretrained_checkpoint = "./adaptive_hcg_slm2_360M_dynamic_test_freeze_lm/checkpoint-40000/"
    adaptive_model = AdaptiveLlamaForCausalLM.from_pretrained(pretrained_checkpoint, torch_dtype=torch.bfloat16)

    # freeze_lm_backbone(adaptive_model)

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
