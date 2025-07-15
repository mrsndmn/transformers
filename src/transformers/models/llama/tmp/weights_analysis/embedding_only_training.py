from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM
import torch

if __name__ == '__main__':

    model_1000 = SentenceLlamaForCausalLM.from_pretrained('./sentence_slm2_1.7B_pretrain_with_end_of_sentence_one_embedding_no_wd_4IQFRDRG/checkpoint-500/')
    model_4000 = SentenceLlamaForCausalLM.from_pretrained('./sentence_slm2_1.7B_pretrain_with_end_of_sentence_one_embedding_no_wd_4IQFRDRG/checkpoint-1000/')

    assert torch.allclose(model_4000.model.layers[0].mlp.gate_proj.weight, model_1000.model.layers[0].mlp.gate_proj.weight)
    assert torch.allclose(model_4000.lm_head.weight[:-1, :], model_1000.lm_head.weight[:-1, :])

    emb_diff = (model_1000.model.embed_tokens.weight - model_4000.model.embed_tokens.weight)

    breakpoint()