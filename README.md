
# Embedding-Wise Layer Hopping in Transformers

This repository is a fork of huggingface/transformers framework.

Here is a list of the most important files:

| File | Description |
| -- | --- |
| `src/transformers/models/llama/modeling_adaptive_llama.py` | Source code of adaptive fan in and fan out modules |
| `src/transformers/models/llama/train_adaptive_llama.py` | Training script for all possible experiments |
| `src/transformers/models/llama/merges_transform/generate_merges/csrc/generate_merges.cu` | Source code for cuda kernels |
| `src/transformers/models/llama/report_benchmark_adaptive_llama.py` | Memory and latency benchmark |


# Building Cuda kernel

```
cd src/transformers/models/llama/merges_transform/
python setup.py build_ext --inplace
```

# Pretraining

```
python ./src/transformers/models/llama/train_adaptive_llama.py --save_strategy steps --save_steps 1000 --generate_merges_transform_impl cuda_kernel --per_device_train_batch_size 16 --learning_rate 0.0001 --num_train_epochs 1 --seed 1008 --training_dataset smollm-corpus --model_type pretrained --llama_checkpoint HuggingFaceTB/SmolLM2-1.7B --dummy_adaptive_fan_in_layers_str 1,1,1,1,1,1,1,0,1,1,1,1 --full_unmerge_str 0,0,0,0,0,0,0,0,0,0,0,0 --full_unmerge_loss_weight 0.0 --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type cosine --merging_type hcg --fan_out_type hcg --temperature_schedule 0 --freeze_lm_backbone 1 --fan_out_projection 1 --logging_steps 50 --warmup_steps 100 --output_dir ./adaptive_hcg_slm2_1.7B_pretrain_8 --max_steps_pretrain_fan_modules 0  --learnt_temperature 0 --select_train_dataset_items 80000 --eval_steps 250 --gumbel_tau 1.0 --weight_decay 0.1 --scale_not_pruned_gradients 0.0 --hcg_loss_weight 10 --ce_merging_loss_weight 0.0 --gumbel_loss_weight_dynamic 0 --hcg_loss_weight_dynamic 1 --dataloader_num_workers 0 --bf16 1 --torch_compile 1 --sparsity_level 0.0 --concrete_random_mask_proba 0 --lm_loss_max_value 1.0 --hcg_loss_max_value 0.0
```


# Training

```
python ./src/transformers/models/llama/train_adaptive_llama.py --save_strategy steps --save_steps 1000 --generate_merges_transform_impl cuda_kernel --per_device_train_batch_size 16 --learning_rate 5e-05 --num_train_epochs 1 --seed 1008 --training_dataset smollm-corpus --model_type pretrained_checkpoint --llama_checkpoint ./adaptive_hcg_slm2_1.7B_pretrain_8/checkpoint-4993/ --dummy_adaptive_fan_in_layers_str 1,1,1,1,1,1,1,0,1,1,1,1 --full_unmerge_str 0,0,0,0,0,0,0,0,0,0,0,0 --full_unmerge_loss_weight 0.0 --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type cosine --merging_type hcg --fan_out_type hcg --temperature_schedule 0 --freeze_lm_backbone 0 --fan_out_projection 1 --logging_steps 50 --warmup_steps 2000 --output_dir ./adaptive_hcg_slm2_1.7B_layersi_max_loss_1.25_fix_fanoutproj_8 --max_steps_pretrain_fan_modules 0  --learnt_temperature 0 --select_train_dataset_items 300000 --eval_steps 250 --gumbel_tau 1.0 --weight_decay 0.1 --scale_not_pruned_gradients 0.0 --hcg_loss_weight 10 --ce_merging_loss_weight 0.0 --gumbel_loss_weight_dynamic 0 --hcg_loss_weight_dynamic 1 --dataloader_num_workers 0 --bf16 1 --torch_compile 1 --sparsity_level 0.0 --concrete_random_mask_proba 0 --lm_loss_max_value 1.25 --hcg_loss_max_value 0.0
```


# Evaluation

# TODO commit lighteval

For evaluations lighteval should be installed from local `./lighteval` distribution.

```
lighteval accelerate --override-batch-size 32 --output-dir ./exps_evaluation --custom-tasks ./cosmopedia/evaluation/lighteval_tasks.py "pretrained=./adaptive_hcg_slm2_1.7B_nofoutproj_8/checkpoint-18743,dtype=bfloat16,device=cuda" "custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|arc|0|1,custom|piqa|0|1"
```

