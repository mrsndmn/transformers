import time
import argparse

import torch

from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInHCG


from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--llama_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--bench_iters",
        default=100,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--generate",
        action='store_true',
    )

    args = parser.parse_args()
    bench_iters = args.bench_iters
    checkpoint: str = args.checkpoint
    llama_checkpoint: str = args.llama_checkpoint

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AdaptiveLlamaForCausalLM.from_pretrained(checkpoint)

    llama_model = LlamaForCausalLM.from_pretrained(llama_checkpoint)
    # llama_model = build_adaptive_llama_from_llama_checkpoint(
    #     llama_checkpoint = 'HuggingFaceTB/SmolLM-360M',
    #     dummy_adaptive_fan_in=[ True ] * model.config.num_hidden_layers,
    #     generate_merges_transform_impl='cuda_kernel',
    #     fan_out_projection=True,
    #     merging_type='hcg',
    #     flash_attention=False,
    # )

    print("model params:", count_params(model))
    print("llama model params:", count_params(llama_model))

    model.to(device)
    model.eval()

    llama_model.to(device)
    llama_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    with torch.no_grad():
        if args.generate:
            for current_model in [llama_model, model]:

                model_inputs = tokenizer([ '<|im_start|> Who are you?' ], return_tensors='pt')
                model_inputs = model_inputs.to(device)

                if isinstance(current_model, AdaptiveLlamaForCausalLM):
                    special_embeddings_mask = torch.zeros_like(model_inputs['input_ids'])
                    special_embeddings_mask[:, 0] = 1
                    model_inputs['special_embeddings_mask'] = special_embeddings_mask


                max_new_tokens = 100
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
                    out = current_model.generate(
                        **model_inputs,
                        **gen_params,
                    )
                    start_time = time.time()
                    out = current_model.generate(
                        **model_inputs,
                        **gen_params,
                    )
                    print("model", type(current_model))
                    print("generation decode:", tokenizer.batch_decode(out))
                    print("duration:", time.time() - start_time)
                    print("tokens per second:", out.shape[-1] / (time.time() - start_time))
        else:
            for i, current_model in enumerate([llama_model, model]):
                text = "<|im_start|> The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. In today's ever-evolving world, technology has become an integral part of our lives, shaping the way we learn, work, and communicate. The COVID-19 pandemic has only accelerated this trend, forcing educational institutions worldwide to adapt quickly to remote learning models. <|im_end|>"
                text_inputs = tokenizer([ text ], return_tensors='pt').to(device)

                if isinstance(current_model, AdaptiveLlamaForCausalLM):
                    special_embeddings_mask = torch.zeros_like(text_inputs['input_ids'])
                    special_embeddings_mask[:, 0] = 1
                    special_embeddings_mask[:, -1] = 1
                    text_inputs['special_embeddings_mask'] = special_embeddings_mask

                forward_output = current_model.forward(**text_inputs)
                forward_output = current_model.forward(**text_inputs)
                del forward_output

                from torch.profiler import profile, record_function, ProfilerActivity

                activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
                # with profile(activities=activities) as prof:
                with profile(activities=activities, profile_memory=True, record_shapes=True, with_stack=True) as prof:
                    with record_function("model_inference"):
                        forward_output = current_model.forward(**text_inputs)

                prof.export_chrome_trace(f"trace_{i}_{type(current_model)}.json")
                prof.export_memory_timeline(f"trace_mem_{i}_{type(current_model)}.html")
                print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

                start = time.time()
                for _ in range(bench_iters):
                    forward_output = current_model.forward(**text_inputs)

                print("model", type(current_model), "elapsed", time.time() - start)

    breakpoint()