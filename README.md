
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

TODO

```
```


# Evaluation

# TODO commit lighteval

For evaluations lighteval should be installed from local `./lighteval` distribution.

```
lighteval accelerate --override-batch-size 32 --output-dir ./exps_evaluation --custom-tasks ./cosmopedia/evaluation/lighteval_tasks.py "pretrained=./adaptive_hcg_slm2_1.7B_nofoutproj_8/checkpoint-18743,dtype=bfloat16,device=cuda" "custom|mmlu_cloze|0|1,custom|mmlu_pro_cloze|0|1,custom|arc|0|1,custom|piqa|0|1"
```

