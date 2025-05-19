
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

# Statistival Pruning Vocabulary

1. Compute pairwise hidden states similarities
```
python src/transformers/models/llama/paper/calculate_hopping_potential.py --llama_checkpoint Qwen/Qwen2.5-7B --batch_size 4 --num_samples 1024
python src/transformers/models/llama/paper/calculate_hopping_potential.py --llama_checkpoint unsloth/Meta-Llama-3.1-8B --batch_size 4 --num_samples 1024
```

2. Compute plots and mintoken_potential_min_df_csv_path
```
# Llama3.1 8B
python src/transformers/models/llama/paper/analyze_hopping_potential.py --per_token_hopping_potentials_path results/token_embeddings_hopping_potential/per_token_hopping_potentials_Meta-Llama-3.1-8B.pkl --token_occurences_path results/token_embeddings_hopping_potential/token_occurencies_Meta-Llama-3.1-8B.pkl --output_dir results/token_embeddings_hopping_potential --llama_checkpoint unsloth/Meta-Llama-3.1-8B

# Qwen2.5-7B
python src/transformers/models/llama/paper/analyze_hopping_potential.py --per_token_hopping_potentials_path results/token_embeddings_hopping_potential/per_token_hopping_potentials_Qwen2.5-7B.pkl --token_occurences_path results/token_embeddings_hopping_potential/token_occurencies_Qwen2.5-7B.pkl --output_dir results/token_embeddings_hopping_potential --llama_checkpoint Qwen/Qwen2.5-7B
```

4. Build Vocab And Recommended first layer for hopping and number of hopping layers

```
# Llama3.1 8B
python src/transformers/models/llama/paper/build_statistical_pruning_vocabulary.py --llama_checkpoint unsloth/Meta-Llama-3.1-8B --output_dir results/token_embeddings_hopping_potential --token_potential_min_df_csv_path results/token_embeddings_hopping_potential/token_potential_min_Meta-Llama-3.1-8B.csv

# Qwen2.5-7B
python src/transformers/models/llama/paper/build_statistical_pruning_vocabulary.py --llama_checkpoint Qwen/Qwen2.5-7B --output_dir results/token_embeddings_hopping_potential --token_potential_min_df_csv_path results/token_embeddings_hopping_potential/token_potential_min_Qwen2.5-7B.csv
```

5. Build Model Checkpoint From pretrained vocabulary

Hyperparameters should be set based on 4th step script output.

```
# Llama3.1-8B
LLAMA31_8B_FAN_IN_INDEX=22
LLAMA31_8B_FAN_OUT_INDEX=26
python src/transformers/models/llama/paper/adaptive_llama_from_vocab.py --checkpoint_base_path unsloth/Meta-Llama-3.1-8B --fan_in_idx $LLAMA31_8B_FAN_IN_INDEX --fan_out_idx $LLAMA31_8B_FAN_OUT_INDEX --vocab_csv_path results/token_embeddings_hopping_potential/pruning_vocab_q1_Meta-Llama-3.1-8B.csv --output_dir ./adaptive_hcg_llama31_8B_l${LLAMA31_8B_FAN_IN_INDEX}-${LLAMA31_8B_FAN_OUT_INDEX}_analytical_pruning

# Qwen2.5-7B
QWEN25_7B_FAN_IN_INDEX=12
QWEN25_7B_FAN_OUT_INDEX=5
python src/transformers/models/llama/paper/adaptive_llama_from_vocab.py --checkpoint_base_path Qwen/Qwen2.5-7B --fan_in_idx $QWEN25_7B_FAN_IN_INDEX --fan_out_idx $QWEN25_7B_FAN_OUT_INDEX --vocab_csv_path results/token_embeddings_hopping_potential/pruning_vocab_q25.csv --output_dir ./adaptive_hcg_qwen25_7B_l${QWEN25_7B_FAN_IN_INDEX}-${QWEN25_7B_FAN_OUT_INDEX}_analytical_pruning
```

# Training

llama31 hcg loss weight 0.1
```
python src/transformers/models/llama/train_adaptive_llama.py --save_strategy steps --save_steps 10000 --generate_merges_transform_impl cuda_kernel --per_device_train_batch_size 4 --learning_rate 0.08 --hcg_learning_rate 0.08 --init_hcg_a 1.0 --hard_hcg_log_a 0 --max_grad_norm 0 --num_train_epochs 1 --seed 1008 --training_dataset smollm-corpus --model_type pretrained --llama_checkpoint unsloth/Meta-Llama-3.1-8B --dummy_adaptive_fan_in_layers_str 1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1 --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type constant_with_warmup --merging_type hcg --temperature_schedule 0 --freeze_lm_backbone 1 --fan_out_projection 0 --warmup_steps 5000 --output_dir adaptive_hcg_llama31_8B_w_0.100_l_22-28_E4OZENLB --learnt_temperature 0 --select_train_dataset_items 500000 --weight_decay 0.1 --scale_not_pruned_gradients 0.0 --hcg_loss_weight 0.1 --hcg_loss_weight_dynamic 0 --bf16 1 --torch_compile 1 --sparsity_level 0.0 --concrete_random_mask_proba 0 --lm_loss_max_value 1.5 --hcg_loss_max_value 0.0 --prohibit_end_of_sentence_pruning 0 --early_stopping_for_pretraining 0 --gradient_accumulation_steps 1 --pretrain_fan_out_projection 0 --eval_strategy steps --concrete_uniform_pruning 0 --concrete_stop_word_pruning 0 --each_layer_pruning 0 --fan_in_idx 22 --fan_out_idx 28 --forward_residuals 0 --train_after_fan_out_llm_layer 0 --freeze_hcg 0 --eval_steps 10000 --init_fan_out_mlp 0

llama31 hcg loss weight 1.0
```
python /workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/accelerate launch --config_file accelerate_config_1gpu.yaml src/transformers/models/llama/train_adaptive_llama.py --save_strategy steps --save_steps 10000 --generate_merges_transform_impl cuda_kernel --per_device_train_batch_size 4 --learning_rate 0.08 --hcg_learning_rate 0.08 --init_hcg_a 1.0 --hard_hcg_log_a 0 --max_grad_norm 0 --num_train_epochs 1 --seed 1008 --training_dataset smollm-corpus --model_type pretrained --llama_checkpoint unsloth/Meta-Llama-3.1-8B --dummy_adaptive_fan_in_layers_str 1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1 --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type constant_with_warmup --merging_type hcg --temperature_schedule 0 --freeze_lm_backbone 1 --fan_out_projection 0 --warmup_steps 5000 --output_dir adaptive_hcg_llama31_8B_w_1.000_l_22-28_SW605VCT --learnt_temperature 0 --select_train_dataset_items 500000 --weight_decay 0.1 --scale_not_pruned_gradients 0.0 --hcg_loss_weight 1.0 --hcg_loss_weight_dynamic 0 --bf16 1 --torch_compile 1 --sparsity_level 0.0 --concrete_random_mask_proba 0 --lm_loss_max_value 1.5 --hcg_loss_max_value 0.0 --prohibit_end_of_sentence_pruning 0 --early_stopping_for_pretraining 0 --gradient_accumulation_steps 1 --pretrain_fan_out_projection 0 --eval_strategy steps --concrete_uniform_pruning 0 --concrete_stop_word_pruning 0 --each_layer_pruning 0 --fan_in_idx 22 --fan_out_idx 28 --forward_residuals 0 --train_after_fan_out_llm_layer 0 --freeze_hcg 0 --eval_steps 10000 --init_fan_out_mlp 0
```


qwen25 hcg loss weight 0.1
```
python src/transformers/models/llama/train_adaptive_llama.py --save_strategy steps --save_steps 10000 --generate_merges_transform_impl cuda_kernel --per_device_train_batch_size 4 --learning_rate 0.08 --hcg_learning_rate 0.08 --init_hcg_a 1.0 --hard_hcg_log_a 0 --max_grad_norm 0 --num_train_epochs 1 --seed 1008 --training_dataset smollm-corpus --model_type pretrained --llama_checkpoint Qwen/Qwen2.5-7B --dummy_adaptive_fan_in_layers_str 1,1,1,1,1,1,1,1,1,1,1,0,1,1 --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type constant_with_warmup --merging_type hcg --temperature_schedule 0 --freeze_lm_backbone 1 --fan_out_projection 0 --warmup_steps 5000 --output_dir adaptive_hcg_qwen25_7B_w_0.100_l_5-19_7A0P103X --learnt_temperature 0 --select_train_dataset_items 500000 --weight_decay 0.1 --scale_not_pruned_gradients 0.0 --hcg_loss_weight 0.1 --hcg_loss_weight_dynamic 0 --bf16 1 --torch_compile 1 --sparsity_level 0.0 --concrete_random_mask_proba 0 --lm_loss_max_value 1.5 --hcg_loss_max_value 0.0 --prohibit_end_of_sentence_pruning 0 --early_stopping_for_pretraining 0 --gradient_accumulation_steps 1 --pretrain_fan_out_projection 0 --eval_strategy steps --concrete_uniform_pruning 0 --concrete_stop_word_pruning 0 --each_layer_pruning 0 --fan_in_idx 5 --fan_out_idx 19 --forward_residuals 0 --train_after_fan_out_llm_layer 0 --freeze_hcg 0 --eval_steps 10000 --init_fan_out_mlp 0
```

qwen25 hcg loss weight 1.0
```
python /workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin/accelerate launch --config_file accelerate_config_1gpu.yaml src/transformers/models/llama/train_adaptive_llama.py --save_strategy steps --save_steps 10000 --generate_merges_transform_impl cuda_kernel --per_device_train_batch_size 4 --learning_rate 0.08 --hcg_learning_rate 0.08 --init_hcg_a 1.0 --hard_hcg_log_a 0 --max_grad_norm 0 --num_train_epochs 1 --seed 1008 --training_dataset smollm-corpus --model_type pretrained --llama_checkpoint Qwen/Qwen2.5-7B --dummy_adaptive_fan_in_layers_str 1,1,1,1,1,1,1,1,1,1,1,0,1,1 --adam_beta1 0.9 --adam_beta2 0.95 --lr_scheduler_type constant_with_warmup --merging_type hcg --temperature_schedule 0 --freeze_lm_backbone 1 --fan_out_projection 0 --warmup_steps 5000 --output_dir adaptive_hcg_qwen25_7B_w_1.000_l_5-19_NYNAYG86 --learnt_temperature 0 --select_train_dataset_items 500000 --weight_decay 0.1 --scale_not_pruned_gradients 0.0 --hcg_loss_weight 1.0 --hcg_loss_weight_dynamic 0 --bf16 1 --torch_compile 1 --sparsity_level 0.0 --concrete_random_mask_proba 0 --lm_loss_max_value 1.5 --hcg_loss_max_value 0.0 --prohibit_end_of_sentence_pruning 0 --early_stopping_for_pretraining 0 --gradient_accumulation_steps 1 --pretrain_fan_out_projection 0 --eval_strategy steps --concrete_uniform_pruning 0 --concrete_stop_word_pruning 0 --each_layer_pruning 0 --fan_in_idx 5 --fan_out_idx 19 --forward_residuals 0 --train_after_fan_out_llm_layer 0 --freeze_hcg 0 --eval_steps 10000 --init_fan_out_mlp 0
```


# Evaluation

# TODO commit lighteval

For evaluations lighteval should be installed from local `./lighteval` distribution.

```
lighteval accelerate --override-batch-size 32 --output-dir ./exps_evaluation --custom-tasks ./cosmopedia/evaluation/lighteval_tasks.py "pretrained=./adaptive_hcg_slm2_1.7B_nofoutproj_8/checkpoint-18743,dtype=bfloat16,device=cuda" "custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|arc|0|1,custom|piqa|0|1"
```

