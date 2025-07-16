
export PATH=/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin:$PATH

cd /workspace-SR004.nfs2/d.tarasov/transformers_adaptive_fan_in_fan_out

export PYTHONPATH="./src"

python src/transformers/models/llama/tmp/process_special_embedding_mask_with_slow_eos_close_token.py $@
