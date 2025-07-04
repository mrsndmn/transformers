import os
import json

from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM

from transformers.models.llama.interpretation.explore_eval_hard_concrete_percent import evaluate_acc_hellaswag, evaluate_acc_winogrande, evaluate_acc_piqa, evaluate_acc_siqa, evaluate_acc_openbookqa

if __name__ == "__main__":

    vanilla_checkpoints_dir = "./vanilla_slm2_1.7B_pretrain_w_0.000_l_-_BM2DG7DH"
    vanilla_16L_checkpoints_dir = "./vanilla_slm2_1.7B_pretrain_16L_w_0.000_l_-_X3NTEOHS"
    adaptive_checkpoints_dir = "./adaptive_slm2_1.7B_pretrain_with_end_of_sentence_token_ftte_w_0.100_l_8-16_K6WC5X8F"

    model_to_checkpoints = [
        ("vanilla", LlamaForCausalLM, vanilla_checkpoints_dir),
        ("vanilla_16L", LlamaForCausalLM, vanilla_16L_checkpoints_dir),
        ("adaptive", AdaptiveLlamaForCausalLM, adaptive_checkpoints_dir),
    ]

    benchmarks = [
        ("hellaswag", evaluate_acc_hellaswag),
        ("winogrande", evaluate_acc_winogrande),
        ("piqa", evaluate_acc_piqa),
        ("siqa", evaluate_acc_siqa),
        ("openbookqa", evaluate_acc_openbookqa),
    ]

    total_results = []

    for model_name, model_class, checkpoints_dir in model_to_checkpoints:
        for dir_name in sorted(os.listdir(checkpoints_dir)):
            if dir_name.startswith("checkpoint-"):
                checkpoint_dir = os.path.join(checkpoints_dir, dir_name)
                model = model_class.from_pretrained(checkpoint_dir)
                model.eval()
                model.to("cuda")

                for benchmark_name, benchmark_fn in benchmarks:
                    bench_results = benchmark_fn(model)
                    acc_norm =  bench_results['acc_norm']

                    current_result = [model_name, dir_name, benchmark_name, acc_norm]
                    total_results.append(current_result)
                    print(" ".join(map(str, current_result)))

    for res_tuple in total_results:
        print(" ".join(map(str, res_tuple)))

    with open("slm_checkpoints_benchmarks_results.json", "w") as f:
        json.dump(total_results, f, indent=4)
