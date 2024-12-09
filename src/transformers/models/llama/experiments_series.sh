
set -o xtrace

# reverse experiments with 1, 4, 8 fan in

WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_01-01" WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_01-01' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 2 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 14 --reverse_dummy_adaptive_fan_in_layers 1 --select_train_dataset_items 20000  --generate_merges_transform_impl cuda_kernel
WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_01-04" WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_01-04' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 2 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 11 --reverse_dummy_adaptive_fan_in_layers 1 --select_train_dataset_items 20000  --generate_merges_transform_impl cuda_kernel
WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_01-08" WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_01-08' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 2 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 7  --reverse_dummy_adaptive_fan_in_layers 1 --select_train_dataset_items 20000  --generate_merges_transform_impl cuda_kernel

# straight experiments with 1, 4, 8 fan in

WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_15-15" WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_15-15' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 2 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 14 --select_train_dataset_items 20000  --generate_merges_transform_impl cuda_kernel
WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_11-15" WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_11-15' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 2 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 11 --select_train_dataset_items 20000  --generate_merges_transform_impl cuda_kernel
WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_07-15" WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_07-15' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 2 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 7  --select_train_dataset_items 20000  --generate_merges_transform_impl cuda_kernel

# full adaptive
WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_01-15" WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_01-15' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 2 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 0  --select_train_dataset_items 20000  --generate_merges_transform_impl cuda_kernel


# Исследовательские вопросы:
# * Как влияет расположение модулей мерджинга - в начале они или в конце?
# * Можно ли сжимать контекст за счет этого механизма?
#       Чтобы на средних слоях в аттеншн влияли сжатые токены из прошлого.
#       При этом на первом и последних слоях эти токены, скорее всего, не нужны.

# Что дальше?
# run generation for adaptive model
# * Модифицировать мерджинг токенов через атеншн?
# * Мб не резидуалы, а mlp? Или декодер на фиксированное кол-во токенов?


