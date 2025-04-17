import time
import argparse
from tqdm.auto import tqdm
import torch

from torch.utils.data import DataLoader

import pandas as pd

from transformers import LlamaConfig, AutoTokenizer, DataCollatorForLanguageModeling

from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanInHCG


from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint

import matplotlib.pyplot as plt


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":

    checkpoints_list = [
        {
            "exp_name": "SmolLM2-360M l8 w 0.010",
            "checkpoint": './adaptive_hcg_slm2_360M_w_0.010_l_8_no_self_attn_XKLT4CMQ/checkpoint-230000',
        },
        {
            "exp_name": "SmolLM2-360M l12 w 0.010",
            "checkpoint": './adaptive_hcg_slm2_360M_w_0.010_l_12_no_self_attn_G8Z0KI2B/checkpoint-230000',
        },

        # Original
        {
            "exp_name": "SmolLM2-360M Original",
            "checkpoint": 'HuggingFaceTB/SmolLM2-360M',
            # "seq_lengths": [ 128, 2048, 4096, 4096 + 1024, 8192 ]
            # "seq_lengths": [ 2048 ]
        },
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bench_dtype = torch.bfloat16

    bench_results = []

    data_files = [ f"cosmopedia-v2/train-{i:05}-of-00104.parquet" for i in range(1) ]
    from datasets import load_dataset
    smollm_corpus = load_dataset("HuggingFaceTB/SmolLM2-corpus", split="train", data_files=data_files)

    smollm_corpus = smollm_corpus.map(lambda x: {"text_length": len(x["text"])})
    smollm_corpus = smollm_corpus.sort('text_length')
    smollm_corpus = smollm_corpus.select(range(len(smollm_corpus) - 500, len(smollm_corpus)))
    print("text len", len(smollm_corpus[-1]['text']))

    for checkpoint_desc in tqdm(checkpoints_list, desc="checkpoints"):
        checkpoint_path = checkpoint_desc['checkpoint']
        exp_name = checkpoint_desc['exp_name']

        batch_size = checkpoint_desc.get('batch_size', 16)
        bench_iters = checkpoint_desc.get('bench_iters', 10)
        # seq_lengths = checkpoint_desc.get('seq_lengths', [ 128, 1024, 4096, 8192])
        seq_lengths = checkpoint_desc.get('seq_lengths', [ 1024, 2048, 4096, 8192 ])

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, padding_side='left')
        tokenizer.pad_token_id = 0
        # tokenizer.pad_token = '<|endoftext|>'
        tokenizer.bos_token = '<|im_start|>'
        tokenizer.eos_token = '<|im_end|>'

        nested_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        def crutch_collator(examples):
            collate_dummy = nested_data_collator([ { "input_ids": x['input_ids'], "attention_mask": x['attention_mask'] } for x in examples])

            collate_dummy['special_embeddings_mask'] = collate_dummy['attention_mask'].cumsum(-1)
            collate_dummy['special_embeddings_mask'][ collate_dummy['special_embeddings_mask'] > 1 ] = 0
            collate_dummy['special_embeddings_mask'][:, -1] = 1

            # assert (collate_dummy['special_embeddings_mask'].sum(dim=-1) == 2).all()

            return collate_dummy

        data_collator = crutch_collator

        model_kwargs = {
            'attn_implementation': 'flash_attention_2',
        }

        if 'HuggingFaceTB' in checkpoint_path:
            current_model = LlamaForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=bench_dtype,
                # TODO check with flash attention
                **model_kwargs,
            )
        else:
            current_model = AdaptiveLlamaForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=bench_dtype,
                **model_kwargs,
            )

        current_model.to(device)
        current_model.eval()

        with torch.no_grad():
            # for i, current_model in enumerate([model]):
            # breakpoint()
            def tokenize_function(examples):
                # 2046 = 2048 - 1 - 1 # eos and bos tokens
                text = [ '<|im_start|>' + (x * 10) + '<|im_end|>' for x in examples['text'] ]
                tokenized_inputs = tokenizer(text, truncation=True, max_length=512*16)

                return tokenized_inputs

            smollm_corpus = smollm_corpus.map(tokenize_function, batched=True)
            smollm_corpus = smollm_corpus.map(lambda x: {"input_ids_len": len(x['input_ids'])})
            smollm_corpus = smollm_corpus.sort('input_ids_len')
            smollm_corpus = smollm_corpus.select(range(len(smollm_corpus)-1, -1, -1))

            dataloader = DataLoader(smollm_corpus, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

            text_inputs = next(iter(dataloader))

            print("text_inputs['input_ids'].shape", text_inputs['input_ids'].shape)

            orig_input_ids = text_inputs['input_ids'].to(device)
            orig_attention_mask = text_inputs['attention_mask'].to(device)
            orig_special_embeddings_mask = text_inputs['special_embeddings_mask'].to(device)


            for sequence_length in tqdm(seq_lengths):

                torch.cuda.reset_peak_memory_stats()

                input_ids = orig_input_ids[:, :sequence_length]
                attention_mask = orig_attention_mask[:, :sequence_length]
                special_embeddings_mask = orig_special_embeddings_mask[:, :sequence_length]

                # warmup
                if isinstance(current_model, AdaptiveFanInHCG):
                    forward_output = current_model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        special_embeddings_mask=special_embeddings_mask,
                        use_cache=True,
                    )
                else:
                    forward_output = current_model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )

                forward_output = None
                logits_cpu = None

                start = time.time()
                print("bench_iters", bench_iters)

                for _ in range(bench_iters):
                    del forward_output
                    del logits_cpu
                    torch.cuda.empty_cache()

                    if isinstance(current_model, AdaptiveFanInHCG):
                        forward_output = current_model.forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            special_embeddings_mask=special_embeddings_mask,
                        )
                    else:
                        forward_output = current_model.forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )

                    logits_cpu = forward_output['logits'][0,0,0].item()

                elapced_mean = (time.time() - start) / bench_iters
                print("elapced_mean", elapced_mean, "model", type(current_model))

                max_mem_alloc = torch.cuda.max_memory_allocated()

                run_info = {
                    "experiment": exp_name,
                    "seq_len": input_ids.shape[1],
                    "elapced_mean":  elapced_mean,
                    "max_mem_alloc": max_mem_alloc,
                }

                if isinstance(current_model, AdaptiveLlamaForCausalLM):
                    fan_in_mask = None
                    for x in forward_output.fan_in_merging_logits_attention_mask:
                        if x is not None:
                            fan_in_mask = x
                            break

                    run_info["pruned_tokens"] = input_ids.shape[1] - fan_in_mask.shape[-1]
                else:
                    run_info["pruned_tokens"] = 0

                del forward_output
                del logits_cpu

                bench_results.append(run_info)

                df = pd.DataFrame(bench_results)
                df.to_csv('exps_evaluation/benchmarks_by_sequence_length.csv')

                plt.clf()

                for exp_name in df['experiment'].unique():
                    df_exp = df[df['experiment'] == exp_name]
                    plt.plot(df_exp['seq_len'], df_exp['elapced_mean'], label=exp_name)

                plt.legend()
                plt.show()
                figure_path = f'exps_evaluation/benchmarks_by_sequence_length.png'
                plt.savefig(figure_path)
                print(f"Saved figure to {figure_path}")

