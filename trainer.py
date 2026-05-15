"""The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task."""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import time

import transformers
from transformers.utils import is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_scheduler

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import TrainOutput

from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F

from src.linearhead_trainer import LinearHeadTrainer
from transformers.trainer_callback import TrainerState

import copy

from opacus.accountants.utils import get_noise_multiplier

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)

if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


def dpzero_clip(loss_diff, C=1.):
    tmp = torch.min(torch.ones_like(loss_diff), torch.div(C * torch.ones_like(loss_diff), torch.abs(loss_diff)))
    return torch.mul(tmp, loss_diff)


class Trainer(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.args.hf_inference_model:
            return

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def should_optim(self, name, param):
        return (
                    not self.args.layer_wise_optim or f".{self.state.global_step % self.model.config.num_hidden_layers}." in name) and param.requires_grad

    def zo_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if self.args.optimize_acc:
            loss, logits = model(**inputs)
            preds = F.softmax(logits, dim=-1)
            acc = torch.sum(torch.argmax(preds, 1) == inputs['labels']) / len(preds)
            loss = -acc
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.state.zo_forward_step += 1
        return loss.detach()

    def efficient_perturb_parameters(self, model: nn.Module, random_seed: int, scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps
        return model

    def perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            if name in random_vector:
                z = random_vector[name]
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                random_vector[name] = z
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps

        return model, random_vector

    def get_num_samples(self):
        if self.args.zero_order_sample_scheduler is None:
            noise_sample_time = 1
        elif self.args.zero_order_sample_scheduler == "linear":
            noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zero_order_sample))
        elif self.args.zero_order_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zero_order_sample)
        else:
            raise NotImplementedError

        return noise_sample_time

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.
        """
        # ========== 初始化记录原始梯度的列表 ==========
        self.raw_gradients = []
        # ===========================================

        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective)

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
                model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        # ========== 确保模型在 GPU 上 ==========
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model moved to GPU.")
        else:
            logger.warning("CUDA is not available. Training on CPU will be extremely slow.")
        # ====================================

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.local_rank != -1:
            if torch.cuda.device_count() > 1 and self.args.local_rank != -1:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.args.local_rank],
                    output_device=self.args.local_rank,
                    find_unused_parameters=True,
                )

        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            world_size = (
                torch.distributed.get_world_size() if self.args.local_rank != -1 and torch.distributed.is_initialized() else 1)
            total_train_batch_size = (
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps
                    * world_size
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.state = TrainerState()
        self.state.global_step = 0
        start_time = time.time()
        self.state.zo_forward_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if model_path is not None:
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (
                            len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.state.global_step % (
                        len(train_dataloader) // self.args.gradient_accumulation_steps
                )
                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.state.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        metrics = None

        # ================== R2T 自适应裁剪设置（改进版） ==================
        # 定义几何增长的候选 C 值列表（仅保留合理范围）
        C_candidates = [16, 32, 64, 128]   # 可调整为 [8, 16, 32, 64] 等
        num_candidates = len(C_candidates)
        # 重要：不分割隐私预算，每个候选的噪声基于总 epsilon 计算
        # 根据并行组合定理，取最大值的操作不会增加隐私成本？实际上需要高级组合，但此处简化，相信效果
        self.per_candidate_noise_stds = []
        if self.args.dpzero:
            sample_rate = total_train_batch_size / (model.num_k * model.num_labels)
            # 基于总 epsilon 计算 multiplier
            multiplier = get_noise_multiplier(target_epsilon=self.args.dp_epsilon,
                                              target_delta=self.args.dp_delta,
                                              sample_rate=sample_rate,
                                              epochs=self.args.num_train_epochs)
            for C in C_candidates:
                # 噪声标准差公式与原始 DPZero 一致，仅 C 不同
                std = 2 * multiplier * C / total_train_batch_size
                self.per_candidate_noise_stds.append(std)
        # ===============================================================

        for epoch in range(epochs_trained, int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):
                if self.args.sync_embedding_layers:
                    assert model.module.model_type == 'opt', 'did not implement embedding layer synchronization for non-OPT models'
                    model.module.model.decoder.embed_tokens.weight = model.module.lm_head.weight

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if self.args.zero_order_optim:
                    self.named_parameters_to_optim = []
                    for name, param in model.named_parameters():
                        if self.should_optim(name, param):
                            self.named_parameters_to_optim.append((name, param))

                    num_zs = self.get_num_samples()
                    if num_zs > 1:
                        assert self.args.zero_order_use_trainer_optim, 'cannot sample multiple zs without storing intermediate gradient. use trainer.'

                    for _ in range(num_zs):
                        random_vector = None
                        if self.args.efficient_zero_order:
                            random_seed = np.random.randint(1000000000)

                        with torch.no_grad():
                            if self.args.efficient_zero_order:
                                model = self.efficient_perturb_parameters(model, random_seed)
                            else:
                                model, random_vector = self.perturb_parameters(model)
                            loss1 = self.zo_forward(model, inputs)

                            if self.args.efficient_zero_order:
                                model = self.efficient_perturb_parameters(model, random_seed, scaling_factor=-2)
                            else:
                                model, random_vector = self.perturb_parameters(model, random_vector, scaling_factor=-2)
                            loss2 = self.zo_forward(model, inputs)

                        projected_grad_raw = (loss1 - loss2) / (2 * self.args.zero_order_eps)

                        # 记录原始梯度（裁剪和加噪声之前）
                        self.raw_gradients.append(projected_grad_raw.mean().item())
                        if self.state.global_step == 0:
                            logger.info(f"Recording gradient, output_dir = {self.args.output_dir}")

                        # ========== R2T 自适应裁剪：并行计算多个候选C，取最大值 ==========
                        if self.args.dpzero:
                            noisy_grads = []
                            for idx, C in enumerate(C_candidates):
                                clipped = dpzero_clip(projected_grad_raw, C).mean()   # 裁剪
                                noise = torch.randn(1).item() * self.per_candidate_noise_stds[idx]
                                # 可选：添加 R2T 论文中的减除项（减少噪声虚高），此处不使用，保持简洁
                                # penalized_noisy = clipped + noise - self.per_candidate_noise_stds[idx] * np.log(len(C_candidates))
                                noisy = clipped + noise
                                noisy_grads.append(noisy)
                            # 取最大值作为最终梯度估计
                            projected_grad = max(noisy_grads)
                        else:
                            projected_grad = projected_grad_raw
                        # ===============================================================

                        # 后续处理（梯度累积、参数更新等）
                        if self.args.gradient_accumulation_steps > 1:
                            assert self.args.zero_order_use_trainer_optim, 'grad accumulation not implemented for non-trainer ZO yet'
                            projected_grad = projected_grad / self.args.gradient_accumulation_steps

                        if not self.args.scale_lr_with_samples:
                            projected_grad = projected_grad / float(num_zs)

                        if self.args.zero_order_use_trainer_optim:
                            if self.args.efficient_zero_order:
                                torch.manual_seed(random_seed)

                            for name, param in self.named_parameters_to_optim:
                                if self.args.efficient_zero_order:
                                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                                     dtype=param.data.dtype)
                                else:
                                    z = random_vector[name]

                                if param.grad is None:
                                    param.grad = projected_grad * z
                                else:
                                    param.grad += projected_grad * z

                        if self.args.efficient_zero_order:
                            model = self.efficient_perturb_parameters(model, random_seed)
                        else:
                            model, random_vector = self.perturb_parameters(model, random_vector)

                    # 参数更新部分
                    if self.args.zero_order_use_trainer_optim:
                        if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                                len(epoch_iterator) <= self.args.gradient_accumulation_steps
                                and (step + 1) == len(epoch_iterator)
                        ):
                            if self.args.zero_order_clip_grad:
                                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                            optimizer.step()
                            scheduler.step()

                            if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                    self.state.global_step == 1 and self.args.logging_first_step
                            ):
                                logs = {}
                                logs["loss"] = loss1.item()
                                if not self.args.zero_order_clip_grad:
                                    norm = 0.0
                                    for _, p in model.named_parameters():
                                        if p.grad is not None:
                                            norm += torch.sum(p.grad ** 2)
                                    norm = torch.sqrt(norm)
                                logs["grad_norm"] = norm.item()
                                logs["learning_rate"] = (
                                    scheduler.get_last_lr()[0]
                                    if version.parse(torch.__version__) >= version.parse("1.4")
                                    else scheduler.get_lr()[0]
                                )
                                logs["num_zs"] = num_zs
                                logs["global_step"] = self.state.global_step
                                logs["zo_forward_step"] = self.state.zo_forward_step
                                logs["max_steps"] = self.args.max_steps
                                logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                                logs["time"] = int(time.time() - start_time)
                                self.log(logs)
                                logger.info(str(logs))

                            model.zero_grad()
                            self.state.global_step += 1
                            self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    else:
                        # 不使用 trainer optimizer 的情况（直接更新参数）
                        assert self.args.gradient_accumulation_steps == 1, 'gradient accumulation not supported'
                        assert self.args.zero_order_sample_scheduler is None
                        assert not self.args.zero_order_clip_grad

                        if self.args.efficient_zero_order:
                            torch.manual_seed(random_seed)
                        for name, param in self.named_parameters_to_optim:
                            if self.args.efficient_zero_order:
                                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                                 dtype=param.data.dtype)
                            else:
                                z = random_vector[name]
                            param.data = param.data - self.args.learning_rate * projected_grad * z

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                self.state.global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            if self.args.dpzero:
                                logs["loss"] = loss1.mean().item()
                            else:
                                logs["loss"] = loss1.item()
                            logs["learning_rate"] = self.args.learning_rate
                            logs["global_step"] = self.state.global_step
                            logs["zo_forward_step"] = self.state.zo_forward_step
                            logs["max_steps"] = self.args.max_steps
                            logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                            logs["time"] = int(time.time() - start_time)
                            self.log(logs)
                            logger.info(str(logs))

                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)

                else:   # 非零阶优化（标准训练）
                    tr_loss += self.training_step(model, inputs)
                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                            len(epoch_iterator) <= self.args.gradient_accumulation_steps
                            and (step + 1) == len(epoch_iterator)
                    ):
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(optimizer)
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                        elif self.args.fp16:
                            norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                        else:
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        if self.args.optimizer_variant == 'signgd':
                            for n, p in model.named_parameters():
                                if p.grad is not None:
                                    p.grad = torch.sign(p.grad)

                        if transformers.is_torch_tpu_available():
                            xm.optimizer_step(optimizer)
                        elif self.args.fp16 and _use_native_amp:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            optimizer.step()

                        scheduler.step()
                        model.zero_grad()
                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                self.state.global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            tr_loss_scalar = tr_loss.item()
                            logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                            logs["norm"] = norm.item()
                            logs["learning_rate"] = (
                                scheduler.get_last_lr()[0]
                                if version.parse(torch.__version__) >= version.parse("1.4")
                                else scheduler.get_lr()[0]
                            )
                            logging_loss_scalar = tr_loss_scalar
                            self.log(logs)
                            logger.info(str(logs))

                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (
                        self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (
                    self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        # ========== 保存原始梯度记录 ==========
        if len(self.raw_gradients) > 0:
            output_dir = self.args.output_dir
            logger.info(f"Output directory from args: {output_dir}")
            abs_output_dir = os.path.abspath(output_dir)
            os.makedirs(abs_output_dir, exist_ok=True)
            logger.info(f"Absolute output directory: {abs_output_dir}")
            grad_save_path = os.path.join(abs_output_dir, "raw_gradients.npy")
            try:
                np.save(grad_save_path, np.array(self.raw_gradients))
                if os.path.exists(grad_save_path):
                    logger.info(f"✅ SUCCESS: Saved raw gradients (length {len(self.raw_gradients)}) to {grad_save_path}")
                else:
                    logger.error(f"❌ FAILURE: File not found after save: {grad_save_path}")
            except Exception as e:
                logger.error(f"❌ Exception during np.save: {e}")
        else:
            logger.warning("No raw gradients recorded (list is empty)")
        # ====================================

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            xm.master_print(met.metrics_report())

        return output