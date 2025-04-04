import pytest
from dataclasses import dataclass, field
import math

import wandb

from torch.nn.utils.rnn import pad_sequence
import torch

from transformers import TrainerCallback
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM, AdaptiveFanOut, AdaptiveFanInOutput, AdaptiveFanOutOutput, AdaptiveLlamaModel, AdaptiveCausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from transformers.utils import is_sagemaker_mp_enabled

from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

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

import torch.profiler

@dataclass
class AdaptiveTrainingArguments(TrainingArguments):
    output_dir: str = field(default="llama_for_sequential_numbers",)
    learning_rate: float = field(default=2e-4)
    hcg_learning_rate: float = field(default=1e-3)

    warmup_steps: int = field(default=500)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=4)
    num_train_epochs: int = field(default=1)

    hcg_temperature: float = field(default=1.0)
    learnt_temperature: bool = field(default=False)
    lr_scheduler_type: str = field(default='constant_with_warmup')

    average_tokens_across_devices: bool = field(default=True)

    llama_checkpoint: str = field(default='')

    weight_decay: float = field(default=0.01)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=1000)
    save_strategy: str = field(default="no")
    save_steps: int = 10000
    save_total_limit: Optional[int] = field(default=15)

    prohibit_end_of_sentence_pruning: bool = field(default=False)
    # scale_token_frequency: bool = field(default=False)

    push_to_hub: bool = field(default=False)
    optim: str = field(default="adamw_torch_fused")
    report_to: str = field(default="wandb")
    logging_steps: int = field(default=100)
    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    merging_type: str = field(default="next_token_merge_mlp")
    freeze_lm_backbone: bool = field(default=False)
    bf16: bool = field(default=True)

    early_stopping_for_pretraining: bool = field(default=False)
    pretrain_fan_out_projection: bool = field(default=False)

    training_dataset: str = "sequential-numbers" # sequential-numbers | smollm-corpus
    model_type: str = "dummy" # dummy | pretrained | SmolLM-1.7B
    
    hcg_loss_weight: float = 0.0
    hcg_loss_weight_dynamic: bool = False

    hcg_loss_max_value: float = 0.0

    lm_loss_max_value: float = 1.5
    sparsity_level: float = 1.0
    dummy_adaptive_fan_in_layers: Optional[int] = None
    dummy_adaptive_fan_in_layers_str: Optional[str] = None
    concrete_random_mask_proba: Optional[float] = None
    
    scale_not_pruned_gradients: float = 0.0
    
    fan_out_type: Optional[str] = None
    
    generate_merges_transform_impl: str = 'cuda_kernel'

    reverse_dummy_adaptive_fan_in_layers: bool = False
    temperature_schedule: bool = False
    temperature_schedule_max_value: int = field(default=10)
    
    select_train_dataset_items: int = 20000
    fan_out_projection: bool = True

class ComputeMetrics():

    def __call__(self, predictions=None, label_ids=None, losses=None, inputs=None, prefix_ids=None, generated_ids=None, **kwargs) -> Dict:
        accuracy = (generated_ids == kwargs['input_ids'][:, :generated_ids.shape[1]]).sum() / generated_ids.size
        # print("generated_ids: ", generated_ids)
        # print("input_ids    : ", kwargs['input_ids'])

        return {
            "accuracy": accuracy
        }


class AdaptiveLlamaTrainer(Trainer):

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """

        if is_sagemaker_mp_enabled():
            raise ValueError("SMP is not supported")

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            decay_parameters = set(decay_parameters)

            # hcg_lr = self.args.hcg_learning_rate

            hcg_params = []
            # hcg_params = set([ p for n, p in opt_model.named_parameters() if "fan_in_mlp" in n ])
            # hcg_params_no_decay = set([ p for n, p in opt_model.named_parameters() if "bias" in n ])
            # decay_parameters = decay_parameters - hcg_params
            # TODO separate group for HCG linear?

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in hcg_params and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                # {
                #     "params": [
                #         p for n, p in opt_model.named_parameters() if (n in hcg_params_no_decay and p.requires_grad)
                #     ],
                #     "weight_decay": 0.0,
                #     "learning_rate": hcg_lr,
                # },
                # {
                #     "params": [
                #         p for n, p in opt_model.named_parameters() if (n in hcg_params and p.requires_grad)
                #     ],
                #     "weight_decay": self.args.weight_decay,
                #     "learning_rate": hcg_lr,
                # },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                raise ValueError("params in optimizer_kwargs is not supported")
                # optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                raise ValueError("model in optimizer_kwargs is not supported")
                # optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                raise ValueError("optimizer_dict in optimizer_kwargs is not supported")
                # optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            print("optimizer_cls, optimizer_kwargs", optimizer_cls, optimizer_kwargs)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                raise ValueError("Adam8bit optimizer is not supported")

        return self.optimizer



    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, log_metrics=True, log_prefix='debug', force_log=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")

        labels = inputs.get('labels', None)
        if labels is None:
            labels = inputs['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

        special_embeddings_mask = inputs.get('special_embeddings_mask')

        attention_mask = inputs['attention_mask']
        token_frequency = inputs.get('token_frequency', None)
        model_kwargs = {
            "input_ids": inputs['input_ids'],
            "labels": labels,
            "attention_mask": attention_mask,
            # "token_frequency": token_frequency,
            "use_cache": None,
            "output_attentions": False,
        }

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            model_kwargs = {**model_kwargs, **loss_kwargs}


        assert special_embeddings_mask is not None
        model_kwargs["special_embeddings_mask"] = special_embeddings_mask

        assert special_embeddings_mask.shape == attention_mask.shape

        outputs = model.forward(**model_kwargs)
        # [ bs, seq_len, 2 ]

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None and self.label_smoother is not None or self.compute_loss_func is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        causal_lm_loss = loss


        # fan_in_merging_logits_sum = sum(x.sum(dim=[0, 1]) for x in fan_in_merging_logits)
        outputs_no_pruning = None
        model_unwrapped = model
        if type(model_unwrapped) != AdaptiveLlamaForCausalLM and hasattr(model_unwrapped, "module"):
            model_unwrapped = model_unwrapped.module

        model_config = model_unwrapped.config

        count_merging_losses = 0

        count_hcg_layers = 0
        hcg_loss = 0
        if self.args.hcg_loss_weight != 0.0 and  model_config.merging_type == 'hcg':
            for i, (hcg_p_open, hcg_p_open_attention_mask) in enumerate(zip(outputs.fan_in_merging_logits, outputs.fan_in_merging_logits_attention_mask)):
                if hcg_p_open is None:
                    continue

                count_hcg_layers += 1
                # [ bs * seq_len ]
                hcg_p_open = hcg_p_open.squeeze(2).flatten()
                p_open_non_masked = hcg_p_open[hcg_p_open_attention_mask.flatten().bool()]

                hcg_loss += p_open_non_masked.mean()

            if count_hcg_layers > 0:
                hcg_loss /= count_hcg_layers


        if self.args.hcg_loss_weight_dynamic and self.args.hcg_loss_max_value > 0:
            raise ValueError("hcg_loss_max_value cant be used with hcg_loss_max_value")

        if self.args.hcg_loss_weight_dynamic:
            # print("sum_pruned_tokens / total_tokens", sum_pruned_tokens / total_tokens)
            outputs_loss = causal_lm_loss
            if len(outputs_loss.shape) > 0:
                outputs_loss = outputs_loss.mean()

            if outputs_loss < self.args.lm_loss_max_value:
                hcg_loss *= self.args.hcg_loss_weight
            else:
                hcg_loss = 0
        elif self.args.hcg_loss_max_value > 0:
            if hcg_loss.item() < self.args.hcg_loss_max_value:
                hcg_loss = 0
            hcg_loss *= self.args.hcg_loss_weight
        else:
            hcg_loss *= self.args.hcg_loss_weight

        # loss = causal_lm_loss
        lm_loss = causal_lm_loss.mean()
        loss = lm_loss + hcg_loss

        # print("pruning_loss", pruning_loss)
        # print("hcg_loss", hcg_loss)

        outputs.loss = loss

        # assert ~ loss.isnan().any(), 'loss cant be none'
        total_tokens = attention_mask.sum().item()
        sum_pruned_tokens = 0

        if force_log or log_metrics and self.state.global_step % self.args.logging_steps == 0:
            outputs_loss = causal_lm_loss
            if len(outputs_loss.shape) > 0:
                outputs_loss = causal_lm_loss.mean()

            hcg_loss_to_log = hcg_loss
            if isinstance(hcg_loss_to_log, torch.Tensor):
                hcg_loss_to_log = hcg_loss_to_log.item()

            log_info = {
                f"{log_prefix}/lm_loss": lm_loss.detach().item(),
                f"{log_prefix}/hcg_loss": hcg_loss_to_log,
                f"{log_prefix}/total_tokens": total_tokens,
            }

            assert model_config.merging_type == 'hcg'
            for i, (concrete, hcg_p_open, fan_in_merging_logits_attention_mask) in enumerate(zip(outputs.fan_in_merging_maps, outputs.fan_in_merging_logits, outputs.fan_in_merging_logits_attention_mask)):
                if hcg_p_open is None:
                    continue

                # [ bs * seq_len ]
                hcg_p_open = hcg_p_open.squeeze(2).flatten()
                p_open_non_masked = hcg_p_open
                concrete_non_masked = concrete.flatten()
                if model.training:
                    p_open_non_masked = p_open_non_masked[fan_in_merging_logits_attention_mask.flatten().bool()]
                    concrete_non_masked = concrete_non_masked[fan_in_merging_logits_attention_mask.flatten().bool()]

                log_info[f'{log_prefix}/concrete_mean_{i}'] = p_open_non_masked.mean().item()
                log_info[f'{log_prefix}/concrete_lt_0.01'] = (p_open_non_masked < 0.01).sum().item()
                log_info[f'{log_prefix}/concrete_lt_0.1'] = (p_open_non_masked < 0.1).sum().item()
                log_info[f'{log_prefix}/concrete_lt_0.5'] = (p_open_non_masked < 0.5).sum().item()

                pruned_tokens_p_open = (p_open_non_masked == 0).sum().item()
                not_pruned_tokens_p_open = total_tokens - pruned_tokens_p_open

                pruned_tokens_concrete = (concrete_non_masked == 0).sum().item()
                not_pruned_tokens_concrete = total_tokens - pruned_tokens_concrete

                log_info[f'{log_prefix}/p_open_pruned_tokens'] = pruned_tokens_p_open
                log_info[f'{log_prefix}/p_open_not_pruned_tokens'] = not_pruned_tokens_p_open
                log_info[f'{log_prefix}/p_open_pruned_tokens_percent'] = pruned_tokens_p_open / total_tokens

                log_info[f'{log_prefix}/concrete_pruned_tokens'] = pruned_tokens_concrete
                log_info[f'{log_prefix}/concrete_not_pruned_tokens'] = not_pruned_tokens_concrete
                log_info[f'{log_prefix}/concrete_pruned_tokens_percent'] = pruned_tokens_concrete / total_tokens

                q = torch.tensor([0.1, 0.5, 0.9], device=p_open_non_masked.device)

                # [ 3 ]
                concrete_quantiles = torch.quantile(p_open_non_masked.float(), q, dim=0, keepdim=False)
                # [ 3 ]
                log_info[f'{log_prefix}/concrete_q10_mean_{i}'] = concrete_quantiles[0].item()
                log_info[f'{log_prefix}/concrete_q50_mean_{i}'] = concrete_quantiles[1].item()
                log_info[f'{log_prefix}/concrete_q90_mean_{i}'] = concrete_quantiles[2].item()

                if self.args.learnt_temperature:
                    log_info[f'{log_prefix}/concrete_{i}_temperature'] = model.model.adaptive_down[i].hcg.temperature.item()

            if outputs_no_pruning:
                log_info[f"{log_prefix}/no_pruning_loss"] = outputs_no_pruning.loss.detach().item()

            self.log(log_info)

        return (loss, outputs) if return_outputs else loss

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
        special_embeddings_mask = inputs.get('special_embeddings_mask', None)
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
            # print('inputs shape', inputs['input_ids'].shape)

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

        assert not llama_checkpoint.startswith("./")

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

        print("dummy_adaptive_fan_in", dummy_adaptive_fan_in)

        assert len(dummy_adaptive_fan_in) == num_layers_half
        model = build_adaptive_llama_from_llama_checkpoint(
            llama_checkpoint,
            dummy_adaptive_fan_in=dummy_adaptive_fan_in,
            generate_merges_transform_impl=training_args.generate_merges_transform_impl,
            fan_out_projection=training_args.fan_out_projection,
            merging_type=training_args.merging_type,
            hcg_temperature=training_args.hcg_temperature,
            learnt_temperature=training_args.learnt_temperature,
            scale_not_pruned_gradients=training_args.scale_not_pruned_gradients,
            concrete_random_mask_proba=training_args.concrete_random_mask_proba,
            pretrain_fan_out_projection=training_args.pretrain_fan_out_projection,
        )

        tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)
    elif training_args.model_type == 'SmolLM-1.7B':
        llama_checkpoint = "HuggingFaceTB/SmolLM-1.7B"
        model = LlamaForCausalLM.from_pretrained(llama_checkpoint, )
        tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint)
    else:
        raise ValueError(f"{training_args.model_type} is not supported")

    tokenizer.padding_side = 'left'

    # model.config.scale_token_frequency = training_args.scale_token_frequency

    if torch.cuda.device_count() > 1:
        model.config.distributed = True

    print("model.config.distributed", model.config.distributed)

    model.config.pretrain_fan_out_projection = training_args.pretrain_fan_out_projection
    if training_args.freeze_lm_backbone:
        for p in model.parameters():
            p.requires_grad = False

        for p in model.model.adaptive_down.parameters():
            p.requires_grad = True

        for p in model.model.adaptive_up.parameters():
            p.requires_grad = True

    if training_args.pretrain_fan_out_projection:
        print("Pretrain fan out projection. Freeze Fan In parameters")
        for adaptive_down in model.model.adaptive_down:
            for p in adaptive_down.parameters():
                p.requires_grad = False

    print("num trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model, tokenizer


class EarlyStoppingCallbacForPretraining(TrainerCallback):

    def __init__(self, min_steps=10):
        self.current_step = 0
        self.min_steps = min_steps

        self.subsequent_steps_metric_ok = 0

    def on_step_end(self, args, state, control, **kwargs):

        self.current_step += 1

        if self.current_step < self.min_steps:
            return control

        if len(state.log_history) == 0:
            return control

        metric_name = 'debug/not_pruned_tokens_percent'
        metric_values = [ x[metric_name] for x in state.log_history if metric_name in x ]

        if metric_values[-1] > 0.99:
            self.subsequent_steps_metric_ok += 1
            if self.subsequent_steps_metric_ok > 500:
                print("Early stopping because of low not pruned tokens percent")
                control.should_training_stop = True
                control.should_save = True
        else:
            self.subsequent_steps_metric_ok = 0

        return control


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

    if training_args.training_dataset == "smollm-corpus":

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
            data_files = [ f"cosmopedia-v2/train-{i:05}-of-00104.parquet" for i in range(20) ]
            smollm_corpus = load_dataset("HuggingFaceTB/smollm-corpus", split="train", data_files=data_files, num_proc=16)

            def tokenize_function(examples):
                # 2046 = 2048 - 1 - 1 # eos and bos tokens
                text = [ '<|im_start|>' + x + '<|im_end|>' for x in examples['text'] ]

                tokenized_inputs = tokenizer(text, truncation=True, padding='max_length', max_length=2046, return_tensors='pt')

                return tokenized_inputs

            print("training_args.select_train_dataset_items", training_args.select_train_dataset_items)
            if training_args.select_train_dataset_items > 0:
                smollm_corpus = smollm_corpus.select(range(training_args.select_train_dataset_items))

            smollm_corpus = smollm_corpus.map(tokenize_function, batched=True, num_proc=32)


        if len(smollm_corpus) <= 100:
            train_dataset = smollm_corpus
            eval_dataset = smollm_corpus
        else:
            smollm_corpus = smollm_corpus.train_test_split(test_size=100, seed=1)
            train_dataset = smollm_corpus['train']
            eval_dataset = smollm_corpus['test']

        nested_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        special_tokens = None
        if training_args.prohibit_end_of_sentence_pruning:
            special_tokens = [ x[0] for x in tokenizer([ '.', '..', '...', '?', '!', ':', ';' ])['input_ids'] ]

        def crutch_collator(examples):
            collate_dummy = nested_data_collator(examples)

            collate_dummy['special_embeddings_mask'] = collate_dummy['attention_mask'].cumsum(-1)
            collate_dummy['special_embeddings_mask'][ collate_dummy['special_embeddings_mask'] > 1 ] = 0
            collate_dummy['special_embeddings_mask'][:, -1] = 1

            if special_tokens is not None:
                for special_token in special_tokens:
                    collate_dummy['special_embeddings_mask'][ collate_dummy['input_ids'] == special_token ] = 1

            return collate_dummy

        data_collator = crutch_collator
    else:
        raise ValueError(f"{training_args.training_dataset} is not supported")

    trackers_project_name = os.path.basename(training_args.output_dir)
    training_args.run_name = trackers_project_name

    # breakpoint()

    callbacks = []
    if training_args.early_stopping_for_pretraining:
        callbacks.append(EarlyStoppingCallbacForPretraining())

    trainer = AdaptiveLlamaTrainer(
        model,
        callbacks=callbacks,

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

    trainer.train()

    # Profile training
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     # on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    # ) as prof:
    #     trainer.train()

    # prof.export_chrome_trace("profile.prof")
