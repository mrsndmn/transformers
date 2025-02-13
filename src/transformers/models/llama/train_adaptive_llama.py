import pytest
from dataclasses import dataclass, field
import math

import wandb

from torch.nn.utils.rnn import pad_sequence
import torch


from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_adaptive_llama import AdaptiveFanInGumbel, AdaptiveFanInGumbel, AdaptiveLlamaForCausalLM, AdaptiveFanOut, AdaptiveFanInOutput, AdaptiveFanOutOutput, AdaptiveLlamaModel, AdaptiveCausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from datasets import load_dataset
import datasets

from transformers import GenerationConfig

VOCAB_SIZE = 200
MAX_SEQ_LEN = 20

import random
import torch
import transformers
from transformers import LlamaConfig, AutoTokenizer, DataCollatorForLanguageModeling

from transformers import Trainer
from transformers import TrainingArguments

import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import Trainer
from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional, Union, Any, Union, Dict, Tuple

import numpy as np

from dataclasses import dataclass, field
import transformers
from transformers import GenerationConfig
from transformers.trainer import nested_detach
from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size, IterableDatasetShard
from transformers.trainer_utils import has_length, denumpify_detensorize, EvalLoopOutput

from torch.utils.data import DataLoader

import time

import torch
from typing import Any, Dict
from typing import List, Optional

@dataclass
class AdaptiveTrainingArguments(TrainingArguments):
    output_dir: str = field(default="llama_for_sequential_numbers",)
    learning_rate: float = field(default=2e-4)
    warmup_steps: int = field(default=500)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=16)
    num_train_epochs: int = field(default=1)
    max_steps_pretrain_fan_modules: int = field(default=2000)
    hcg_temperature: float = field(default=1.0)
    learnt_temperature: bool = field(default=False)
    lr_scheduler_type: str = field(default='constant_with_warmup')

    llama_checkpoint: str = field(default='')

    weight_decay: float = field(default=0.01)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=500)
    save_strategy: str = field(default="no")
    save_steps: int = 10000
    save_total_limit: Optional[int] = field(default=1)
    
    push_to_hub: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    report_to: str = field(default="wandb")
    logging_steps: int = field(default=50)
    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)
    merging_type: str = field(default="next_token_merge_mlp")
    freeze_lm_backbone: bool = field(default=False)

    training_dataset: str = "sequential-numbers" # sequential-numbers | smollm-corpus
    model_type: str = "dummy" # dummy | pretrained | SmolLM-1.7B
    
    ce_merging_loss_weight: float = 0.0
    full_unmerge_loss_weight: float = 1.0
    hcg_loss_weight: float = 0.0
    hcg_loss_weight_dynamic: bool = False

    hcg_loss_max_value: float = 0.0

    lm_loss_max_value: float = 1.5
    sparsity_level: float = 1.0
    gumbel_loss_weight_dynamic: bool = False
    dummy_adaptive_fan_in_layers: Optional[int] = None
    dummy_adaptive_fan_in_layers_str: Optional[str] = None
    concrete_random_mask_proba: Optional[float] = None
    
    gumbel_tau: float = 2.0
    scale_not_pruned_gradients: float = 0.0
    
    full_unmerge_str: Optional[str] = None
    fan_out_type: Optional[str] = None
    
    generate_merges_transform_impl: str = 'cuda_kernel'

    reverse_dummy_adaptive_fan_in_layers: bool = False
    temperature_schedule: bool = False
    temperature_schedule_max_value: int = field(default=10)
    
    select_train_dataset_items: int = 20000
    fan_out_projection: bool = True

    with_special_embeddings_mask: bool = True

class ComputeMetrics():

    def __call__(self, predictions=None, label_ids=None, losses=None, inputs=None, prefix_ids=None, generated_ids=None, **kwargs) -> Dict:
        accuracy = (generated_ids == kwargs['input_ids'][:, :generated_ids.shape[1]]).sum() / generated_ids.size
        # print("generated_ids: ", generated_ids)
        # print("input_ids    : ", kwargs['input_ids'])

        return {
            "accuracy": accuracy
        }


class SequentialNumbersDataset():
    def __init__(self, length=10000, num_numbers=100, max_sequence_length=20):
        self.length = length
        self.num_numbers = num_numbers
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return self.length

    def __getitem__(self, i):

        current_length = self.max_sequence_length
        start_from = random.randint(3, self.num_numbers - current_length - 3) # pad + bos + eos tokens

        max_padding = self.max_sequence_length - current_length

        inputs_ids = [1] + list(range(start_from, start_from + current_length)) + [2] + ([0] * max_padding)
        labels = [1] + list(range(start_from, start_from + current_length)) + [2] + ([-100] * max_padding)
        attention_mask_length = current_length + 2
        attention_mask = ([1] * attention_mask_length) + ([0] * max_padding)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        special_embeddings_mask = torch.zeros_like(attention_mask)
        special_embeddings_mask[0] = 1
        special_embeddings_mask[attention_mask_length - 1] = 1

        assert len(attention_mask) == len(inputs_ids)

        return {
            "input_ids": torch.tensor(inputs_ids),
            "labels": torch.tensor(labels),
            "special_embeddings_mask": special_embeddings_mask,
            "attention_mask": attention_mask,
        }



class AdaptiveLlamaTrainer(Trainer):
    def compute_loss(self, model: AdaptiveLlamaForCausalLM, inputs, return_outputs=False, log_metrics=True, log_prefix='debug', force_log=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        labels = inputs.get('labels', None)
        if labels is None:
            labels = inputs['input_ids'].clone()
            inputs['input_ids'][inputs['input_ids'] == self.tokenizer.pad_token_id] = -100

        special_embeddings_mask = inputs.get('special_embeddings_mask')
        if special_embeddings_mask is None:
            special_embeddings_mask = inputs.get('special_tokens_mask') > 0

        attention_mask = inputs['attention_mask']
        model_kwargs = {
            "input_ids": inputs['input_ids'],
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if self.args.with_special_embeddings_mask:
            assert special_embeddings_mask is not None
            # assert special_embeddings_mask.sum() > 1

            model_kwargs["special_embeddings_mask"] = special_embeddings_mask

            assert special_embeddings_mask.shape == attention_mask.shape

        outputs = model.forward(**model_kwargs)
        # [ bs, seq_len, 2 ]

        # fan_in_merging_logits_sum = sum(x.sum(dim=[0, 1]) for x in fan_in_merging_logits)
        outputs_no_pruning = None
        if model.config.full_unmerge is not None and sum(model.config.full_unmerge) > 0:
            model_kwargs['full_unmerge'] = model.config.full_unmerge
            outputs_no_pruning = model.forward(**model_kwargs)
        

        ce_merging_loss_sum = torch.tensor(0.0, device=outputs.loss.device)
        count_merging_losses = 0
        sum_pruned_tokens = 0

        if self.args.ce_merging_loss_weight > 0.0:
            for i, (fan_in_merging_logits, fan_in_merging_logits_attention_mask) in enumerate(zip(outputs.fan_in_merging_logits, outputs.fan_in_merging_logits_attention_mask)):
                # ce_targets = outputs.fan_in_merging_maps[i][:, :, 1].flatten()
                if fan_in_merging_logits is None:
                    continue

                fan_in_merging_logits = fan_in_merging_logits.flatten(0, 1)
                ce_targets = torch.zeros([ fan_in_merging_logits.shape[0] ], device=fan_in_merging_logits.device, dtype=torch.long)
                ce_targets[fan_in_merging_logits_attention_mask.flatten().bool() == False] = -100
                ce_merging_loss_sum += torch.nn.functional.cross_entropy(fan_in_merging_logits, ce_targets, label_smoothing=0.1)

                # def debug_grad(grad):
                #     print(grad)
                #     breakpoint()
                #     return grad
                # ce_merging_loss_sum.register_hook(debug_grad)

                count_merging_losses+=1
                # breakpoint()
                # print("fan_in_merging_logits", fan_in_merging_logits[:2])
                # print("ce_merging_loss_sum", i, ce_merging_loss_sum)
                    # print(fan_in_merging_logits[:10])

            if count_merging_losses > 0:
                ce_merging_loss_sum /= count_merging_losses

        ce_merging_loss_sum *= self.args.ce_merging_loss_weight

        if self.args.gumbel_loss_weight_dynamic:
            exp_scale = 30 * max(outputs.loss.detach().item() - 1.8, 0)
            ce_merging_loss_sum /= torch.exp(torch.tensor(exp_scale, device=outputs.loss.device))


        count_hcg_layers = 0
        hcg_loss = 0
        sum_pruned_tokens = 0
        if self.args.hcg_loss_weight > 0.0 and  model.config.merging_type == 'hcg':
            for i, (hcg_p_open, hcg_p_open_attention_mask) in enumerate(zip(outputs.fan_in_merging_logits, outputs.fan_in_merging_logits_attention_mask)):
                if hcg_p_open is None:
                    continue

                count_hcg_layers += 1
                # [ bs * seq_len ]
                hcg_p_open = hcg_p_open.squeeze(2).flatten()
                concrete_non_masked = hcg_p_open[hcg_p_open_attention_mask.flatten().bool()]

                hcg_loss += concrete_non_masked.mean()

            sum_pruned_tokens = sum([ x[:, :, 0].sum().item() for x in outputs.fan_in_merging_logits if x is not None])


            if count_hcg_layers > 0:
                hcg_loss /= count_hcg_layers

        total_tokens = attention_mask.sum().item()

        if self.args.hcg_loss_weight_dynamic and self.args.hcg_loss_max_value > 0:
            raise ValueError("hcg_loss_max_value cant be used with hcg_loss_max_value")

        if self.args.hcg_loss_weight_dynamic:
            # print("sum_pruned_tokens / total_tokens", sum_pruned_tokens / total_tokens)
            if outputs.loss < self.args.lm_loss_max_value:
                hcg_loss *= self.args.hcg_loss_weight
            else:
                hcg_loss = 0
        elif self.args.hcg_loss_max_value > 0:
            if hcg_loss.item() < self.args.hcg_loss_max_value:
                hcg_loss = 0
            hcg_loss *= self.args.hcg_loss_weight
        else:
            hcg_loss *= self.args.hcg_loss_weight

        # loss = outputs.loss
        pruning_loss = outputs.loss
        loss = pruning_loss + ce_merging_loss_sum + hcg_loss

        # print("pruning_loss", pruning_loss)
        # print("ce_merging_loss", ce_merging_loss_sum)
        # print("hcg_loss", hcg_loss)
        
        if outputs_no_pruning is not None:
            # print("outputs_no_pruning_loss", outputs_no_pruning.loss.item())
            loss += outputs_no_pruning.loss * self.args.full_unmerge_loss_weight

        outputs.loss = loss

        # assert ~ loss.isnan().any(), 'loss cant be none'

        if force_log or log_metrics and self.state.global_step % self.args.logging_steps == 0:
            outputs_loss = outputs.loss
            if len(outputs_loss.shape) > 0:
                outputs_loss = outputs.loss.mean()
                
            hcg_loss_to_log = hcg_loss
            if isinstance(hcg_loss_to_log, torch.Tensor):
                hcg_loss_to_log = hcg_loss_to_log.item()

            ce_merging_loss_sum_float = ce_merging_loss_sum
            if isinstance(ce_merging_loss_sum_float, torch.Tensor):
                ce_merging_loss_sum_float = ce_merging_loss_sum_float.item()

            log_info = {
                f"{log_prefix}/pruning_loss": pruning_loss.detach().item(),
                f"{log_prefix}/hcg_loss": hcg_loss_to_log,
                f"{log_prefix}/ce_merging_loss": ce_merging_loss_sum_float,
                f"{log_prefix}/sum_pruned_tokens": (total_tokens - sum_pruned_tokens),
                f"{log_prefix}/not_pruned_tokens": sum_pruned_tokens,
                f"{log_prefix}/total_tokens": total_tokens,
                f"{log_prefix}/not_pruned_tokens_percent": (sum_pruned_tokens / (total_tokens + 1e-4)),
            }

            if model.config.merging_type == 'hcg':
                for i, (hcg_p_open, fan_in_merging_logits_attention_mask) in enumerate(zip(outputs.fan_in_merging_logits, outputs.fan_in_merging_logits_attention_mask)):
                    if hcg_p_open is None:
                        continue

                    # [ bs * seq_len ]
                    hcg_p_open = hcg_p_open.squeeze(2).flatten()
                    concrete_non_masked = hcg_p_open[fan_in_merging_logits_attention_mask.flatten().bool()]
                    log_info[f'{log_prefix}/concrete_mean_{i}'] = concrete_non_masked.mean().item()
                    log_info[f'{log_prefix}/concrete_lt_0.01'] = (concrete_non_masked < 0.01).sum().item()
                    log_info[f'{log_prefix}/concrete_lt_0.1'] = (concrete_non_masked < 0.1).sum().item()
                    log_info[f'{log_prefix}/concrete_lt_0.5'] = (concrete_non_masked < 0.5).sum().item()
                    q = torch.tensor([0.1, 0.5, 0.9], device=concrete_non_masked.device)

                    # [ 3 ]
                    concrete_quantiles = torch.quantile(concrete_non_masked.float(), q, dim=0, keepdim=False)
                    # [ 3 ]
                    log_info[f'{log_prefix}/concrete_q10_mean_{i}'] = concrete_quantiles[0].item()
                    log_info[f'{log_prefix}/concrete_q50_mean_{i}'] = concrete_quantiles[1].item()
                    log_info[f'{log_prefix}/concrete_q90_mean_{i}'] = concrete_quantiles[2].item()

                    if self.args.learnt_temperature:
                        log_info[f'{log_prefix}/concrete_{i}_temperature'] = model.model.adaptive_down[i].hcg.temperature.item()

                for i, adaptive_down in enumerate(model.model.adaptive_down):
                    if isinstance(adaptive_down, AdaptiveFanInGumbel):
                        log_info[f'{log_prefix}/gumbel_tau_{i}'] = adaptive_down.gumbel_tau

            if outputs_no_pruning:
                log_info[f"{log_prefix}/no_pruning_loss"] = outputs_no_pruning.loss.detach().item()

            self.log(log_info)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: AdaptiveLlamaForCausalLM, *args, **kwargs):
        result = super().training_step(model, *args, **kwargs)
        
        base_temperature_value = 1.0
        current_tau = base_temperature_value + abs(math.sin(math.pi * self.state.global_step / 2000)) * (self.args.temperature_schedule_max_value - base_temperature_value)
        
        if self.state.global_step % self.args.logging_steps == 0:
            extra_log = dict()
            if self.args.temperature_schedule:
                extra_log['tau'] = current_tau

            if False and hasattr(model.model, "adaptive_down"):
                for i, adown in enumerate(model.model.adaptive_down):
                    if isinstance(adown, (AdaptiveFanInGumbel)):
                        merger_mpl_grad = adown.fan_in_mlp.weight.grad.norm(2).item()
                        fan_in_mlp_weight_grad_sum = adown.fan_in_mlp.weight.grad.sum(1)
                        fan_in_mlp_weight_sum = adown.fan_in_mlp.weight.sum(1)
                        fan_in_mlp_bias = adown.fan_in_mlp.bias
                        assert merger_mpl_grad is not None, "merger_mpl_grad is expected to be not none"
                        extra_log[f"merger_mpl_grad_norm_{i}"] = merger_mpl_grad
                        extra_log[f"merger_mpl_weight_grad_sum_0_{i}"] = fan_in_mlp_weight_grad_sum[0].item()
                        extra_log[f"merger_mpl_weight_grad_sum_1_{i}"] = fan_in_mlp_weight_grad_sum[1].item()

                        extra_log[f"merger_mpl_weight_sum_0_{i}"] = fan_in_mlp_weight_sum[0].item()
                        extra_log[f"merger_mpl_weight_sum_1_{i}"] = fan_in_mlp_weight_sum[1].item()
                        
                        if fan_in_mlp_bias is not None:
                            extra_log[f"merger_mpl_bias_0_{i}"] = fan_in_mlp_bias[0].item()
                            extra_log[f"merger_mpl_bias_1_{i}"] = fan_in_mlp_bias[1].item()

            self.log(extra_log)

        if self.args.temperature_schedule:
            if hasattr(model.model, "adaptive_down"):
                # if self.state.global_step % 50 == 0:
                #     print("self.state.global_step, tau=", current_tau, "global_step", self.state.global_step)
                for i, adown in enumerate(model.model.adaptive_down):
                    adown.set_gumbel_tau(current_tau)
        
        return result


    def update_eval_set_kwargs_containers(self, model, inputs):

        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        forced_eos_token_id = eos_token_id

        if self.processing_class is not None:
            bos_token_id = self.processing_class.bos_token_id
            eos_token_id = self.processing_class.eos_token_id
            pad_token_id = self.processing_class.pad_token_id
            forced_eos_token_id = eos_token_id

        gen_params = {
            "do_sample": False,
            "early_stopping": False,
            "num_beams": 1,
            "repetition_penalty": 2.5,
            "remove_invalid_values": True,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "forced_eos_token_id": forced_eos_token_id,
            "use_cache": False,
            "no_repeat_ngram_size": 4,
            "num_return_sequences": 1,
        }
        genconfig = GenerationConfig()

        caption_legth = inputs['input_ids'].shape[1] - 2
        genconfig.max_length = caption_legth

        batch_size, seq_len = inputs['input_ids'].shape[0], 2
        special_embeddings_mask = torch.ones([batch_size, seq_len], device=inputs['input_ids'].device)
        special_embeddings_mask[:, 1] = 0
        attention_mask = torch.ones([batch_size, seq_len], device=inputs['input_ids'].device)

        prefix_ids = inputs['input_ids'][:, :2]
        all_generation_params = {
            'generation_config': genconfig,
            'max_new_tokens': caption_legth,
            'inputs': prefix_ids,
            'special_embeddings_mask': special_embeddings_mask,
            'attention_mask': attention_mask,
            **gen_params,
        }

        result = {
            "prefix_ids": prefix_ids,
            "input_ids": inputs['input_ids'],
        }

        if self.args.training_dataset == 'sequential-numbers':
            result["generated_ids"] = model.generate(**all_generation_params)

        return result

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        
        # print("inputs", inputs.keys())
        # breakpoint()
        
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []


        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                    log_metrics=False,
                    log_prefix='eval_debug',
                    force_log=True,
                )
            loss = loss.mean().detach()

            logits = outputs.logits

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        labels = None

        return (loss, logits, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=0)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            extra_eval_set_kwargs = self.update_eval_set_kwargs_containers(model, inputs)
            for key, value in extra_eval_set_kwargs.items():
                if key not in eval_set_kwargs:
                    eval_set_kwargs[key] = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=0)

                eval_set_kwargs[key].add(value)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        predictions=all_preds,
                        label_ids=all_labels,
                        **eval_set_kwargs
                    )

                del losses, logits, labels, inputs, eval_set_kwargs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                for key in eval_set_kwargs.keys():
                    eval_set_kwargs[key].to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()
        eval_set_kwargs_arrays = dict()
        for key, value in eval_set_kwargs.items():
            eval_set_kwargs_arrays[key] = eval_set_kwargs[key].get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            # and all_preds is not None
            # and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs_arrays["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs_arrays["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                predictions=all_preds,
                label_ids=all_labels,
                **eval_set_kwargs_arrays
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


def build_model(training_args: AdaptiveTrainingArguments):
    tokenizer = None

    if training_args.model_type == 'dummy':
        num_layers = 2
        num_layers_half = num_layers // 2

        # Маска, с помощью которой можно управлять,
        # для каких слоев нужно использовать обучаемый FanIn,
        # а для каких слоев будет использоваться просто Identity (DummyFanIn)
        dummy_adaptive_fan_in = [ False ] * num_layers_half
        # dummy_adaptive_fan_in = [ False, False, False, False ]
        # dummy_adaptive_fan_in = [ True, True, True, False ]
        # dummy_adaptive_fan_in = [ False, True, True, True ]
        assert len(dummy_adaptive_fan_in) == num_layers_half
        llama_config = LlamaConfig(
            hidden_size=128,
            vocab_size=VOCAB_SIZE,
            intermediate_size=256,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            max_position_embeddings=MAX_SEQ_LEN,
            use_cache=False,
            attn_implementation = 'eager',
            dummy_adaptive_fan_in = dummy_adaptive_fan_in,
            merging_type=training_args.merging_type,
        )

        model = AdaptiveLlamaForCausalLM(llama_config)
    elif training_args.model_type == 'pretrained_checkpoint':
        llama_checkpoint = training_args.llama_checkpoint
        print("Load model from", llama_checkpoint)
        model = AdaptiveLlamaForCausalLM.from_pretrained(llama_checkpoint, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)
        
    elif training_args.model_type == 'pretrained':
        from transformers.models.llama.convert_hf_llama_to_adaptive_llama import build_adaptive_llama_from_llama_checkpoint

        # llama_checkpoint = "HuggingFaceTB/SmolLM-1.7B"
        # llama_checkpoint = "HuggingFaceTB/SmolLM-135M"
        llama_checkpoint = training_args.llama_checkpoint
        if llama_checkpoint is None or llama_checkpoint == "":
            llama_checkpoint = "HuggingFaceTB/SmolLM-360M"
        llama_config = LlamaConfig.from_pretrained(llama_checkpoint)
        num_layers = llama_config.num_hidden_layers
        num_layers_half = num_layers // 2

        if training_args.dummy_adaptive_fan_in_layers is not None:
            smart_layers_count = num_layers_half - training_args.dummy_adaptive_fan_in_layers
            dummy_adaptive_fan_in = [ True ] * training_args.dummy_adaptive_fan_in_layers + [ False ] * smart_layers_count
        elif training_args.dummy_adaptive_fan_in_layers_str is not None:
            assert not training_args.reverse_dummy_adaptive_fan_in_layers, 'reverse_dummy_adaptive_fan_in_layers is prohibited with dummy_adaptive_fan_in_layers_str'
            dummy_adaptive_fan_in = list(map(lambda x: bool(int(x)), training_args.dummy_adaptive_fan_in_layers_str.split(',')))
        else:
            raise ValueError("either dummy_adaptive_fan_in_layers or dummy_adaptive_fan_in_layers_str must be defined")
        if training_args.reverse_dummy_adaptive_fan_in_layers:
            dummy_adaptive_fan_in = list(reversed(dummy_adaptive_fan_in))
        
        full_unmerge = None
        if training_args.full_unmerge_str is not None:
            full_unmerge = list(map(lambda x: bool(int(x)), training_args.full_unmerge_str.split(',')))
        
        print("dummy_adaptive_fan_in", dummy_adaptive_fan_in)
        
        assert len(dummy_adaptive_fan_in) == num_layers_half
        model = build_adaptive_llama_from_llama_checkpoint(
            llama_checkpoint,
            dummy_adaptive_fan_in=dummy_adaptive_fan_in,
            generate_merges_transform_impl=training_args.generate_merges_transform_impl,
            fan_out_projection=training_args.fan_out_projection,
            merging_type=training_args.merging_type,
            freeze_lm_backbone=training_args.freeze_lm_backbone,
            full_unmerge=full_unmerge,
            fan_out_type=training_args.fan_out_type,
            hcg_temperature=training_args.hcg_temperature,
            learnt_temperature=training_args.learnt_temperature,
            gumbel_tau=training_args.gumbel_tau,
            scale_not_pruned_gradients=training_args.scale_not_pruned_gradients,
            concrete_random_mask_proba=training_args.concrete_random_mask_proba,
        )

        tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)
    elif training_args.model_type == 'SmolLM-1.7B':
        llama_checkpoint = "HuggingFaceTB/SmolLM-1.7B"
        model = LlamaForCausalLM.from_pretrained(llama_checkpoint, )
        tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)
    else:
        raise ValueError(f"{training_args.model_type} is not supported")

    print("num trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    tokenizer.padding_side = 'left'

    return model, tokenizer


# pretrained
# WANDB_MODE=online PYTHONPATH=/Users/d.tarasov/workspace/transformers/src:./src ~/miniconda3/envs/audio/bin/python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --per_device_train_batch_size 32 --num_train_epochs 10 --seed 1001 --training_dataset smollm-corpus --model_type pretrained

# dummy
# WANDB_MODE=online PYTHONPATH=/Users/d.tarasov/workspace/transformers/src:./src ~/miniconda3/envs/audio/bin/python -m pdb -c continue src/transformers/models/llama/train_adaptive_llama.py --per_device_train_batch_size 32 --num_train_epochs 10 --seed 1001
if __name__ == "__main__":

    import subprocess
    subprocess.check_output(['nvidia-smi'])


    hf_parser = transformers.HfArgumentParser(AdaptiveTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()
    
    model, tokenizer = build_model(training_args)
    
    compute_metrics = None
    data_collator = None

    if training_args.training_dataset == "sequential-numbers":
        compute_metrics = ComputeMetrics()
        train_dataset = SequentialNumbersDataset(length=2000, num_numbers=VOCAB_SIZE, max_sequence_length=MAX_SEQ_LEN)
        eval_dataset = SequentialNumbersDataset(length=64, num_numbers=VOCAB_SIZE, max_sequence_length=MAX_SEQ_LEN)

        def collate_sequential_numbers(elements):
            
            input_ids               = pad_sequence([ el['input_ids'] for el in elements ], batch_first=True)
            labels                  = pad_sequence([ el['labels'] for el in elements ], batch_first=True)
            attention_mask          = pad_sequence([ el['attention_mask'] for el in elements ], batch_first=True)
            special_embeddings_mask = pad_sequence([ el['special_embeddings_mask'] for el in elements ], batch_first=True)

            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "special_embeddings_mask": special_embeddings_mask,
            }

        data_collator = collate_sequential_numbers

    elif training_args.training_dataset == "smollm-corpus":

        tokenizer.pad_token = tokenizer.eos_token
        # from tokenizers.processors import TemplateProcessing
        # tokenizer.post_processor = TemplateProcessing(
        #     single=f"{tokenizer.bos_token} $A {tokenizer.eos_token}",
        #     special_tokens=[(tokenizer.bos_token, tokenizer.bos_token_id), (tokenizer.eos_token, tokenizer.eos_token_id)],
        # )

        im_start_token_id = 1
        im_end_token_id = 2

        disk_dataset_path = "data/tokenized-smollm-corpus-1-shard.dataset"

        # if os.path.exists(disk_dataset_path):
        if False:
            smollm_corpus = datasets.Dataset.load_from_disk(disk_dataset_path)
        else:
            # load and tokenize
            data_files = [ f"cosmopedia-v2/train-{i:05}-of-00104.parquet" for i in range(10) ]
            smollm_corpus = load_dataset("HuggingFaceTB/smollm-corpus", split="train", data_files=data_files)

            def tokenize_function(examples):
                # 2046 = 2048 - 1 - 1 # eos and bos tokens
                tokenized_inputs = tokenizer(examples['text'], return_special_tokens_mask=True, truncation=True, max_length=1022)
                for x in tokenized_inputs['input_ids']:
                    x.insert(0, im_start_token_id)
                    x.append(im_end_token_id)

                for x in tokenized_inputs['attention_mask']:
                    x.insert(0, 1)
                    x.append(1)

                for x in tokenized_inputs['special_tokens_mask']:
                    x.insert(0, 1)
                    x.append(1)
                
                return tokenized_inputs

            print("training_args.select_train_dataset_items", training_args.select_train_dataset_items)
            if training_args.select_train_dataset_items > 0:
                smollm_corpus = smollm_corpus.select(range(training_args.select_train_dataset_items))

            smollm_corpus = smollm_corpus.map(tokenize_function, batched=True)

            # smollm_corpus = smollm_corpus.rename_column('special_tokens_mask', 'special_embeddings_mask')
            # print(smollm_corpus[0]['input_ids'])
            # breakpoint()
            # smollm_corpus.save_to_disk("data/tokenized-smollm-corpus-1-shard.dataset")

        assert sum(smollm_corpus[0]['special_tokens_mask']) > 0
        
        if len(smollm_corpus) <= 100:
            train_dataset = smollm_corpus
            eval_dataset = smollm_corpus
        else:
            smollm_corpus = smollm_corpus.train_test_split(test_size=100, seed=1)
            train_dataset = smollm_corpus['train']
            eval_dataset = smollm_corpus['test']
        
        nested_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        def crutch_collator(examples):
            collate_dummy = nested_data_collator(examples)
            
            collate_dummy['special_tokens_mask'] = torch.zeros_like(collate_dummy['attention_mask'])

            for i, ex in enumerate(examples):
                currrent_special_tokens_mask = ex['special_tokens_mask']
                collate_dummy['special_tokens_mask'][i, :len(currrent_special_tokens_mask)] = torch.tensor(currrent_special_tokens_mask, dtype=torch.long)

            # assert (collate_dummy['special_tokens_mask'].sum(dim=-1) == 2).all()

            return collate_dummy

        data_collator = crutch_collator
    else:
        raise ValueError(f"{training_args.training_dataset} is not supported")

    # training_args.max_steps = training_args.max_steps_pretrain_fan_modules
    # trainer.args.max_steps = -1
    # trainer.args.warmup_steps = 0

    trackers_project_name = os.path.basename(training_args.output_dir)
    training_args.run_name = trackers_project_name

    trainer = AdaptiveLlamaTrainer(
        model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.accelerator.init_trackers(
        project_name=trackers_project_name,
    )

    # with torch.autograd.set_detect_anomaly(True):
    trainer.train(
        # resume_from_checkpoint="adaptive_gumbel_2-2_1.7B_model_my_checkpoint-4995",
        # resume_from_checkpoint='adaptive_13-13_hcg_temp_5.0/checkpoint-4995/',
    )
