import time
import argparse
from tqdm.auto import tqdm
import torch
import numpy as np

from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInHCG


from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_generate(current_model, tokenizer, max_new_tokens=100, use_cache=True):
    model_inputs = tokenizer([ '<|begin_of_text|> Question: Which Lloyd Webber musical premiered in the US on 10th December 1993?', ], return_tensors='pt', padding=True)
    model_inputs = model_inputs.to(device)

    print('model_inputs["input_ids"].shape', model_inputs['input_ids'].shape)

    if isinstance(current_model, AdaptiveLlamaForCausalLM):
        special_embeddings_mask = model_inputs['attention_mask'].cumsum(-1)
        special_embeddings_mask[special_embeddings_mask > 1] = 0
        model_inputs['special_embeddings_mask'] = special_embeddings_mask

    max_new_tokens = 1000
    gen_params = {
        "do_sample": False,
        "min_new_tokens": max_new_tokens,
        "max_new_tokens": max_new_tokens,
        "early_stopping": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "forced_eos_token_id": tokenizer.eos_token_id,
        "stop_strings": [tokenizer.eos_token, '<|im_end|>', '<|end_of_text|>'],
        "tokenizer": tokenizer,
        "use_cache": use_cache,
        "num_return_sequences": 1,
    }

    current_model.config.use_cache = gen_params['use_cache']
    current_model.eval()

    with torch.no_grad():
        gen_params_warmup = {**gen_params, "max_new_tokens": 10}
        out = current_model.generate(
            **model_inputs,
            **gen_params_warmup,
        )
        del out

        start_time = time.time()
        start_sequence = model_inputs['input_ids'].shape[1]
        out_orig = current_model.generate(
            **model_inputs,
            **gen_params,
        )

        out = out_orig[:, start_sequence:]

        generation_decode ="\n\n".join(tokenizer.batch_decode(out, skip_special_tokens=True))

        print("model", type(current_model))
        print("duration:", time.time() - start_time)
        print("generated sequence length:", out.shape[-1])
        print("tokens per second:", out.shape[-1] / (time.time() - start_time))
        print("generation decode:", tokenizer.batch_decode(out))
        print("generation decode:", generation_decode)
        print("generation decode:", "\n\n".join(tokenizer.batch_decode(out_orig, skip_special_tokens=True)))

        if isinstance(current_model, AdaptiveLlamaForCausalLM):
            print("pruned tokens")

    return generation_decode

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--fan_in_index",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--fan_out_index",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--llama_checkpoint",
        default=None,  # 'HuggingFaceTB/SmolLM2-1.7B',
        # required=True,
    )
    parser.add_argument(
        "--bench_iters",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--generate",
        action='store_true',
    )
    parser.add_argument(
        "--profile",
        action='store_true',
    )

    args = parser.parse_args()
    bench_iters = args.bench_iters
    checkpoint: str = args.checkpoint
    llama_checkpoint: str = args.llama_checkpoint

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bench_dtype = torch.float32
    # bench_dtype = torch.bfloat16

    llama_model = None
    if llama_checkpoint is not None:
        llama_model = LlamaForCausalLM.from_pretrained(
            llama_checkpoint,
            torch_dtype=bench_dtype,
            # attn_implementation='flash_attention_2',
        )

    model = AdaptiveLlamaForCausalLM.from_pretrained( checkpoint, torch_dtype=bench_dtype )

    if args.fan_in_index is not None:
        model.model.fan_in_idx = args.fan_in_index
    if args.fan_out_index is not None:
        model.model.fan_out_idx = args.fan_out_index

    model.to(device)
    model.eval()

    # model.config.

    print("model params:", count_params(model))
    print("model.model.fan_in", model.model.fan_in)

    # model.model.adaptive_up[4] = NoopAdaptiveFanOut(model.config)

    if llama_checkpoint is not None:
        llama_model.to(device)
        llama_model.eval()
        print("llama model params:", count_params(llama_model))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')

    with torch.no_grad():
        if args.generate:
            print("go generate")
            # for i in range(2):
            #     current_model = model
            #     if i == 0:
            #         current_model.eval()
            #     else:
            #         current_model.train()
            # for current_model in tqdm([model, llama_model]):
            for current_model in tqdm([model, llama_model]):
                if current_model is None:
                    continue

                generate_cache_true = run_generate(current_model, tokenizer, use_cache=True)
                generate_cache_false = run_generate(current_model, tokenizer, use_cache=False)
                assert generate_cache_true == generate_cache_false

        else:

            import matplotlib.pyplot as plt

            num_tokens = []
            llama_model_points = []
            model_points = []
            pruned_tokens = []
            total_tokens = []

            for i, current_model in enumerate([llama_model, model]):
                if current_model is None:
                    continue

            # for i, current_model in enumerate([model]):
            # for i, current_model in enumerate([llama_model]):
                # for i, current_model in enumerate([model]):
                for scale in tqdm(range(1, 2)):
                    text = "<|im_start|> The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. <|im_end|>"
                    text = text[:20000]
                    text = text * scale
                    text_inputs = tokenizer([ text ] * 16, return_tensors='pt').to(device)

                    if len(num_tokens) < scale:
                        num_tokens.append(text_inputs['input_ids'].shape[-1])

                    print("text_inputs['input_ids'].shape", text_inputs['input_ids'].shape)

                    if isinstance(current_model, AdaptiveLlamaForCausalLM):
                        special_embeddings_mask = text_inputs['attention_mask'].cumsum(-1)
                        special_embeddings_mask[special_embeddings_mask > 1] = 0
                        text_inputs['special_embeddings_mask'] = special_embeddings_mask

                    forward_output = current_model.forward(**text_inputs)
                    forward_output = current_model.forward(**text_inputs)
                    del forward_output

                    if args.profile:
                        from torch.profiler import profile, record_function, ProfilerActivity
                        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
                        # with profile(activities=activities) as prof:
                        with profile(activities=activities, profile_memory=True, record_shapes=True, with_stack=True) as prof:
                            with record_function("model_inference"):
                                model_inputs = tokenizer([ 'Question: Which Lloyd Webber musical premiered in the US on 10th December 1993?\nAnswer:', 'Question: How are you?\nAnswer:', ], return_tensors='pt', padding=True)
                                model_inputs = model_inputs.to(device)

                                print('model_inputs["input_ids"].shape', model_inputs['input_ids'].shape)

                                if isinstance(current_model, AdaptiveLlamaForCausalLM):
                                    special_embeddings_mask = model_inputs['attention_mask'].cumsum(-1)
                                    special_embeddings_mask[special_embeddings_mask > 1] = 0
                                    model_inputs['special_embeddings_mask'] = special_embeddings_mask

                                max_new_tokens = 10
                                gen_params = {
                                    "do_sample": False,
                                    "min_new_tokens": max_new_tokens,
                                    "max_new_tokens": max_new_tokens,
                                    # "early_stopping": True,
                                    # "num_beams": 1,
                                    "eos_token_id": tokenizer.eos_token_id,
                                    "pad_token_id": tokenizer.eos_token_id,
                                    "forced_eos_token_id": tokenizer.eos_token_id,
                                    "stop_strings": [tokenizer.eos_token, '<|im_end|>'],
                                    "tokenizer": tokenizer,
                                    "use_cache": False,
                                    "num_return_sequences": 1,
                                }

                                current_model.config.use_cache = gen_params['use_cache']
                                current_model.eval()

                                out = current_model.generate(
                                    **model_inputs,
                                    **gen_params,
                                )

                                forward_output = current_model.forward(**text_inputs, use_cache=False)
                                torch.cuda.synchronize()

                        prof.export_chrome_trace(f"trace_{i}_{type(current_model)}.json")
                        # prof.export_memory_timeline(f"trace_mem_{i}_{type(current_model)}.html")
                        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

                    print("bench_iters", bench_iters)

                    forward_output = None

                    durations = []
                    for _ in range(bench_iters):
                        del forward_output
                        # torch.cuda.empty_cache()

                        start = time.time()
                        forward_output = current_model.forward(**text_inputs, use_cache=False)
                        torch.cuda.synchronize()
                        durations.append( time.time() - start )

                    if bench_iters > 0:
                        elapced_mean = np.mean(durations)
                        elapced_std = np.std(durations)
                    else:
                        elapced_mean = 0
                        elapced_std = 0

                    print("elapced_mean", elapced_mean, "std", elapced_std, "model", type(current_model))

                    # if isinstance(current_model, AdaptiveLlamaForCausalLM):
                    #     model_points.append(elapced_mean)
                    #     pruned_tokens.append(forward_output.fan_in_merging_logits_attention_mask[-1].shape[-1])
                    #     total_tokens.append(forward_output.fan_in_merging_logits_attention_mask[0].shape[-1])
                    # else:
                    #     llama_model_points.append(elapced_mean)

                    # if isinstance(current_model, AdaptiveLlamaForCausalLM):
                    #     print("attention masks shapes:", [ x.shape for x in forward_output.fan_in_merging_logits_attention_mask ])

            # plt.figure(0)
            # plt.plot(num_tokens, model_points, label=llama_checkpoint)

            # checkpoint_slug = checkpoint.split("/")[1]

            # plt.plot(num_tokens, llama_model_points, label=checkpoint_slug)
            # plt.xlabel("sequence length")
            # plt.ylabel(".forward() wall time")
            # plt.legend()
            # plt.show()
            # plt.savefig("bench_llama_pruned.png")
            # plt.figure(1)
            # plt.plot(num_tokens, pruned_tokens, label="pruned tokens")
            # plt.plot(num_tokens, total_tokens, label="total tokens")
            # plt.xlabel("iteration")
            # plt.ylabel("Tokens")
            # plt.legend()
            # plt.show()
            # plt.savefig("bench_llama_pruned_tokens.png")


    breakpoint()