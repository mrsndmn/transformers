
set -o xtrace

# reverse experiments with 1, 4, 8 fan in

TRAIN_DATASET_ITEMS=40000


WANDB_PROJECT=adaptive_attention WANDB_NAME="vanilla-llama" WANDB_MODE=online PYTHONPATH=./src python src/transformers/models/llama/train_adaptive_llama.py --output_dir 'vanilla-llama' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 15  --select_train_dataset_items "$TRAIN_DATASET_ITEMS"


# WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_01-01" WANDB_MODE=online PYTHONPATH=./src python src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_01-01' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 14 --reverse_dummy_adaptive_fan_in_layers 1 --select_train_dataset_items "$TRAIN_DATASET_ITEMS"  --generate_merges_transform_impl cuda_kernel
WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_01-04" WANDB_MODE=online PYTHONPATH=./src python src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_01-04' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 11 --reverse_dummy_adaptive_fan_in_layers 1 --select_train_dataset_items "$TRAIN_DATASET_ITEMS"  --generate_merges_transform_impl cuda_kernel
WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_01-08" WANDB_MODE=online PYTHONPATH=./src python src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_01-08' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 7  --reverse_dummy_adaptive_fan_in_layers 1 --select_train_dataset_items "$TRAIN_DATASET_ITEMS"  --generate_merges_transform_impl cuda_kernel

# straight experiments with 1, 4, 8 fan in

WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_15-15" WANDB_MODE=online PYTHONPATH=./src python src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_15-15' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 14 --select_train_dataset_items "$TRAIN_DATASET_ITEMS"  --generate_merges_transform_impl cuda_kernel
WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_11-15" WANDB_MODE=online PYTHONPATH=./src python src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_11-15' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 11 --select_train_dataset_items "$TRAIN_DATASET_ITEMS"  --generate_merges_transform_impl cuda_kernel
WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_07-15" WANDB_MODE=online PYTHONPATH=./src python src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_07-15' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 7  --select_train_dataset_items "$TRAIN_DATASET_ITEMS"  --generate_merges_transform_impl cuda_kernel

# full adaptive
WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_01-15" WANDB_MODE=online PYTHONPATH=./src python src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_01-15' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 0  --select_train_dataset_items "$TRAIN_DATASET_ITEMS"  --generate_merges_transform_impl cuda_kernel


# adaptive with residual projection
WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_11-15_fan_out_with_residual_projection" WANDB_MODE=online PYTHONPATH=./src python src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_15-15_fan_out_with_residual_projection' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 11 --select_train_dataset_items "$TRAIN_DATASET_ITEMS"  --generate_merges_transform_impl cuda_kernel

# TODO Is projection required at all?

# Next experiments
# --fan_out_projection 0
# 
# WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_11-15_fan_out_with_residual_no_projection" WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_11-15_fan_out_with_residual_no_projection' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 11 --select_train_dataset_items 20000  --generate_merges_transform_impl cuda_kernel --fan_out_projection 0

# --ce_merging_loss_weight 0.0001
# 
# WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_11-15_fan_out_with_residual_projection_ce_loss" WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_11-15_fan_out_with_residual_projection_ce_loss' --per_device_train_batch_size 5 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 11 --select_train_dataset_items 20000  --generate_merges_transform_impl cuda_kernel --fan_out_projection 1 --ce_merging_loss_weight 0.0001

# Tune hyper parameters, increase batch size
# warmup_steps=1000
# batch_size = 20
# learning_rate = 1e-4
# adam_beta1=0.9
# adam_beta2=0.95
# WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_14-15_residual_projection_bs20" WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_13-15_residual_projection_bs20' --per_device_train_batch_size 20 --learning_rate 0.0001 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 13 --select_train_dataset_items 80000 --generate_merges_transform_impl cuda_kernel --adam_beta1 0.9 --adam_beta2 0.95

# warmup_steps=1000
# batch_size = 40
# learning_rate = 2e-4
# adam_beta1=0.9
# adam_beta2=0.95
# WANDB_PROJECT=adaptive_attention WANDB_NAME="adaptive_14-15_residual_projection_bs40" WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --output_dir 'adaptive_13-15_residual_projection_bs40' --per_device_train_batch_size 40 --learning_rate 0.0002 --num_train_epochs 1 --seed 1003 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --dummy_adaptive_fan_in_layers 13 --select_train_dataset_items 80000  --generate_merges_transform_impl cuda_kernel --adam_beta1 0.9 --adam_beta2 0.95

# TODO
# bs 40 + 2 GPU +? freeze all prev layers
# bs 40 + 2 GPU --ce_merging_loss_weight 0.001

# Выводы:
# FanOutProjection:
#   - На скорость не влияет
#   - В сочетании с распространением градиентов только на непоследние смердженые токены дает eval лосс даже чуть лучше, чем Vanilla Llama


# Исследовательские вопросы:
# * Почему процент смердженых токенов не растет?
# * Как процент смердженых токенов растет на разных слоях?
# * Перезапустить эксп с лоссом на распределение мерджига?
# * Как влияет расположение модулей мерджинга - в начале они или в конце?
# * Исследовать другие способы определения, должны ли токены быть смерджены.
#       - Cosine Similarity
#       - Attention Head Similarities
#       - MHAttention Head Similarities Voting
# * Какие именно токены мерджатся? Попробовать проинтерпретировать это
# * Почему для гумбеля градиенты текут только на единички - ок ли это?
# * Нужен ли на последнем промежуточном слое fanin / fan out?
# * Как адаптивность влияет на бенчмарки?

# * Корится, если запустить DataParallel обучение

# * Как именно мерджить токены?
#       - Сумма
#       - Конкатенация + MLP
#       - Поэлементное умножение + LN
#       - TODO

# * Можно ли сжимать контекст за счет этого механизма?
#       - Необходимо: Как управлять степенью сжатия контекста?
#       - Необходимо: Чтобы на текстовых данных сжатие было хотя бы ~10%, в идеале больше
#       Чтобы на средних слоях в аттеншн влияли сжатые токены из прошлого.
#       При этом на первом и последних слоях эти токены, скорее всего, не нужны.

# * Как можно использовать это для токенизации аудио?
#       - Для сжатия контекста должно хорошо подойти

# Что дальше?
# run generation for adaptive model
# * Модифицировать мерджинг токенов через атеншн?
# * Мб не резидуалы, а mlp? Или декодер на фиксированное кол-во токенов?


