from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM

from transformers import LlamaForCausalLM, AutoTokenizer

if __name__ == "__main__":

    init_m1a = AdaptiveLlamaForCausalLM.from_pretrained("./paper_checkpoints/pretrain/adaptive_slm2_135M_random_init")
    init_m1 = LlamaForCausalLM.from_pretrained("./paper_checkpoints/pretrain/slm2_135M_random_init/")

    w1a = init_m1a.get_input_embeddings().weight[:49152]
    w1  = init_m1.get_input_embeddings().weight[:49152]

    w1a1b_equal = (w1a == w1).all(dim=-1)

    print(w1a1b_equal.sum())
    breakpoint()

    m1 = AdaptiveLlamaForCausalLM.from_pretrained("./adaptive_slm2_135M_pretrain_with_end_of_sentence_token_w_0.100_l_10-20_4REEAIIL/checkpoint-12420/")
    m2 = AdaptiveLlamaForCausalLM.from_pretrained("./adaptive_slm2_135M_pretrain_w_0.100_l_10-20_AZJQ5WL0/checkpoint-12420/")
    m3 = LlamaForCausalLM.from_pretrained("./vanilla_slm2_135M_pretrain_w_0.000_l_-_DFAC2NSB/checkpoint-12420")


    w1 = m1.get_input_embeddings().weight[:49152]
    w2 = m2.get_input_embeddings().weight
    w3 = m3.get_input_embeddings().weight

    w12_equal = (w1 == w2).all(dim=-1)
    w13_equal = (w1 == w3).all(dim=-1)

    print(w12_equal)
    print(w13_equal)

    breakpoint()



