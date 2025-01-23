
import argparse

import torch

from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInHCG


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    args = parser.parse_args()
    checkpoint: str = args.checkpoint
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if checkpoint.startswith('HuggingFaceTB'):
        model = LlamaForCausalLM.from_pretrained(checkpoint)
    else:
        model = AdaptiveLlamaForCausalLM.from_pretrained(checkpoint)

    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    model_inputs = tokenizer([ '<|im_start|> Who are you?' ], return_tensors='pt')
    model_inputs = model_inputs.to(device)

    if isinstance(model, AdaptiveLlamaForCausalLM):
        special_embeddings_mask = torch.zeros_like(model_inputs['input_ids'])
        special_embeddings_mask[:, 0] = 1
        model_inputs['special_embeddings_mask'] = special_embeddings_mask
    

    max_new_tokens = 10
    gen_params = {
        "do_sample": False,
        "min_new_tokens": 1,
        "max_new_tokens": max_new_tokens,
        "early_stopping": True,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "remove_invalid_values": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "forced_eos_token_id": tokenizer.eos_token_id,
        "stop_strings": [tokenizer.eos_token, '<|im_end|>'],
        "tokenizer": tokenizer,
        "use_cache": False,
        "no_repeat_ngram_size": 4,
        "num_return_sequences": 1,
    }
    
    with torch.no_grad():
        # out = model.generate(
        #     **model_inputs,
        #     **gen_params,
        # )
        # print("generation decode:", tokenizer.batch_decode(out))
        
        if isinstance(model, AdaptiveLlamaForCausalLM):
            text = "<|im_start|> In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models.<|im_end|>"
            text_inputs = tokenizer([ text ], return_tensors='pt').to('cuda')
            
            special_embeddings_mask = torch.zeros_like(text_inputs['input_ids'])
            special_embeddings_mask[:, 0] = 1
            special_embeddings_mask[:, -1] = 1
            text_inputs['special_embeddings_mask'] = special_embeddings_mask

            forward_output = model.forward(**text_inputs)

            token_will_be_passed = forward_output['fan_in_merging_logits'][0].max(dim=-1).indices[0].cpu().numpy().tolist()
            
            logits = forward_output['fan_in_merging_logits'][0][0]
            logits_prune_confidence = logits[:, 0] - logits[:, 1]
            
            print("token_will_be_passed", sum(token_will_be_passed), '/', len(token_will_be_passed))
            input_ids = text_inputs['input_ids'][0].cpu().numpy().tolist()
            for i, (token_will_be_passed, token_id) in enumerate(zip(token_will_be_passed, input_ids)):
                logits_prune_confidence_i = logits_prune_confidence[i]
                
                print(token_will_be_passed, f"\t{logits_prune_confidence_i:.2f}", "\t", token_id, "\t", tokenizer.decode(token_id))

    breakpoint()