
# reverse experiments with 1, 2, 4 fan in

WANDB_PROJECT=adaptive_attention WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --per_device_train_batch_size 5 --num_train_epochs 1 --seed 1002 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --ce_merging_loss_weight 0.1 --dummy_adaptive_fan_in_layers 14 --reverse_dummy_adaptive_fan_in_layers 1
WANDB_PROJECT=adaptive_attention WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --per_device_train_batch_size 5 --num_train_epochs 1 --seed 1002 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --ce_merging_loss_weight 0.1 --dummy_adaptive_fan_in_layers 13 --reverse_dummy_adaptive_fan_in_layers 1
WANDB_PROJECT=adaptive_attention WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --per_device_train_batch_size 5 --num_train_epochs 1 --seed 1002 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --ce_merging_loss_weight 0.1 --dummy_adaptive_fan_in_layers 11 --reverse_dummy_adaptive_fan_in_layers 1

# straight experiments with 1, 2, 4 fan in

WANDB_PROJECT=adaptive_attention WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --per_device_train_batch_size 5 --num_train_epochs 1 --seed 1002 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --ce_merging_loss_weight 0.1 --dummy_adaptive_fan_in_layers 13 --reverse_dummy_adaptive_fan_in_layers 0
WANDB_PROJECT=adaptive_attention WANDB_MODE=online PYTHONPATH=./src python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --per_device_train_batch_size 5 --num_train_epochs 1 --seed 1002 --training_dataset smollm-corpus --model_type pretrained --gradient_checkpointing 1 --ce_merging_loss_weight 0.1 --dummy_adaptive_fan_in_layers 11 --reverse_dummy_adaptive_fan_in_layers 0


# Текущие проблемы:
# * Долгое вычисление модуля - логика мерджа сложная и не векторизуется - мб можно напилить свое куда ядро
# * Без явного лосса на стремление к мерджингу ничего не схлопывается (возможно, это следствие первой проблемы, тк для гумбеля нужны больше батчи и больше данных)
# * Как влияет расположение модулей мерджинга - в начале они или в конце?

