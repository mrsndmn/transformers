from transformers.models.llama.modeling_sentence_llama import SentenceLlamaForCausalLM
from transformers import AutoTokenizer
import torch
import os
import shutil

if __name__ == '__main__':

    source_dir = "./sentence_slm2_1.7B_pretrain_with_end_of_sentence_token_w_0.100_l__S9M6BLOE/checkpoint-400_untied"   

    tokenizer = AutoTokenizer.from_pretrained(source_dir)

    model = SentenceLlamaForCausalLM.from_pretrained(
        source_dir,
        torch_dtype=torch.bfloat16,
    )

    with torch.no_grad():
        model.lm_head.weight.copy_(model.model.embed_tokens.weight)
        assert (model.lm_head.weight == model.model.embed_tokens.weight).all()

    output_path = "./sentence_slm2_1.7B_pretrain_with_end_of_sentence_token_w_0.100_l__S9M6BLOE/checkpoint-400_untied_fixed"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    model.save_pretrained(output_path)

    tokenizer.save_pretrained(output_path)

    print(f"saved to {output_path}")
