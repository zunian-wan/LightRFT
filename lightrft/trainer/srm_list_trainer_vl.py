"""
Trainer utilities for scalar reward models (vision-language capable).

This module provides a trainer that supports pairwise preference training with
different loss functions, including Bradley-Terry (BT) loss and Human Preference
Score (HPS) Loss. It integrates with the project's Strategy abstraction for distributed
training, gradient accumulation, and checkpointing, while optionally logging to
Weights & Biases or TensorBoard.
"""

import os
import json
from tqdm import tqdm
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.distributed as dist

from lightrft.models import ListMLELoss, RankNetLoss, pad_to_length
from lightrft.utils import DistributedSampler, all_gather_and_flatten, all_reduce_dict


class SRMListTrainerVL:
    """
    Trainer for listwise scalar vision-language reward models (ListMLE/listwise).

    :param model: The model to be trained.
    :type model: torch.nn.Module
    :param strategy: Training strategy.
    :type strategy: Strategy
    :param optim: Optimizer.
    :type optim: torch.optim.Optimizer
    :param train_dataloader: Dataloader providing listwise data.
    :type train_dataloader: torch.utils.data.DataLoader
    :param scheduler: Learning rate scheduler.
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param tokenizer: Tokenizer.
    :type tokenizer: Callable
    :param eval_dataloader: Optional dataloader used for evaluation.
    :type eval_dataloader: Optional[torch.utils.data.DataLoader]
    :param max_epochs: Maximum number of training epochs.
    :type max_epochs: int
    :param loss: Loss function to use. Default 'listwise'.
    :type loss: str
    """
    def __init__(
        self,
        model: nn.Module,
        strategy,
        optim: Optimizer,
        train_dataloader,
        scheduler,
        tokenizer,
        eval_dataloader=None,
        max_epochs: int = 2,
        loss: str = "listwise",
        margin: float = 0.0,
    ) -> None:
        self.strategy = strategy
        self.epochs = max_epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args

        if loss == "listmle":
            self.loss = "ListMLE"
            self.loss_fn = ListMLELoss()
            self.strategy.print("ListMLE Loss")
        elif loss == "ranknet":
            self.loss = "RankNet"
            self.loss_fn = RankNetLoss(margin=margin)
            self.strategy.print(f"RankNet Loss with margin {margin}")
        else:
            raise ValueError(f"invalid loss type: {loss}")

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)


    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None) -> None:
        """
        Train the model for ``max_epochs`` (Listwise).
        """
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())

        head_types = self.model.head_types
        # Stats tracking
        loss_mean = {ht: 0.0 for ht in head_types}
        total_loss_mean = 0.0

        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            
            for batch in self.train_dataloader:
                device = torch.cuda.current_device()
                
                # Unpack listwise batch
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]
                
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device)
                    image_grid_thw = image_grid_thw.to(device)
                
                # Ranks: [B, K]
                ranks = batch["ranks"].to(device)
                candidate_masks = batch.get("candidate_masks")
                if candidate_masks is not None:
                    candidate_masks = candidate_masks.to(device)
                
                B, K = ranks.shape
                
                # Forward pass - Model sees B*K items as individual samples
                scores_dict = self.model(
                    sequences=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )

                total_loss = 0.0
                head_loss = {}

                for head_type in head_types:
                    scores = scores_dict.get(head_type)
                    if scores is None:
                        continue
                        
                    # Flattened scores [Total_Valid, 1] -> [Total_Valid]
                    scores_flat = scores.view(-1)
                    
                    # Recover structure [B, max_K]
                    full_scores = torch.full((B, K), float('-inf'), device=device, dtype=scores.dtype)
                    
                    # Ensure mask is bool and used for assignment
                    mask_bool = candidate_masks.bool()
                    full_scores[mask_bool] = scores_flat
                    
                    # Compute Listwise Loss
                    loss = self.loss_fn(full_scores, ranks, mask=candidate_masks)
                    
                    head_loss[head_type] = loss
                    total_loss += loss
                    
                    # Update stats
                    loss_mean[head_type] = loss_mean[head_type] * 0.9 + 0.1 * loss.item()

                total_loss_mean = total_loss_mean * 0.9 + 0.1 * total_loss.item()
                
                self.strategy.backward(total_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                # Logging
                logs_dict = {
                    "loss": total_loss.item(),
                    "loss_mean": total_loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                
                for head_type in head_types:
                    logs_dict[f"{head_type}_loss"] = head_loss.get(head_type, torch.tensor(0.0)).item()
                    logs_dict[f"{head_type}_loss_mean"] = loss_mean[head_type]

                # Reduce logs
                for k in logs_dict.keys():
                    logs_dict[k] = self.strategy.all_reduce(logs_dict[k])
                
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # Checkpointing
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}) -> None:
        """
        Log metrics and optionally run evaluation and checkpointing.

        :param args: Training arguments providing logging/eval/save intervals
            and checkpoint configurations.
        :type args: Any
        :param global_step: Current global optimization step.
        :type global_step: int
        :param step_bar: Progress bar for step-level updates.
        :type step_bar: tqdm.tqdm
        :param logs_dict: Dictionary of metrics to log (will be reduced across
            ranks via ``strategy.all_reduce`` before display/logging).
        :type logs_dict: Dict[str, float]
        :param client_states: Extra state to persist with checkpoints (e.g.,
            consumed samples).
        :type client_states: Dict[str, Any]

        :returns: None
        :rtype: NoneType
        """
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if self.eval_dataloader and len(self.eval_dataloader) > 0:
                # Pass args first to match evaluate signature (args, dataloader, steps)
                self.evaluate(args, self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )

    def evaluate(self, args, eval_dataloader, steps=0) -> None:
        """
        Evaluate (Listwise).
        Calculates loss on evaluation set.
        """
        step_bar = tqdm(
            range(len(eval_dataloader)),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        
        loss_sum = 0
        total_batches = 0
        
        try:
            with torch.no_grad():
                for batch in eval_dataloader:
                    device = torch.cuda.current_device()
                    
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    pixel_values = batch.get("pixel_values")
                    image_grid_thw = batch.get("image_grid_thw")
                    ranks = batch["ranks"].to(device)
                    candidate_masks = batch.get("candidate_masks")
                    if candidate_masks is not None:
                        candidate_masks = candidate_masks.to(device)
                        
                    if pixel_values is not None:
                        pixel_values = pixel_values.to(device)
                        image_grid_thw = image_grid_thw.to(device)
                        
                    B, K = ranks.shape
                    
                    scores_dict = self.model(
                        sequences=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                    )
                    
                    batch_loss = 0
                    valid_head = False
                    for head_type in self.model.head_types:
                        scores = scores_dict.get(head_type)
                        if scores is None: continue
                        
                        # Recover scores structure: [Total_Valid] -> [B, max_K]
                        scores_flat = scores.view(-1)
                        full_scores = torch.full((B, K), float('-inf'), device=device, dtype=scores.dtype)
                        
                        # Use candidate_masks (boolean) to scatter valid scores back
                        mask_bool = candidate_masks.bool()
                        full_scores[mask_bool] = scores_flat
                        
                        batch_loss += self.loss_fn(full_scores, ranks, mask=candidate_masks)
                        valid_head = True
                    
                    if valid_head:
                        loss_sum += batch_loss.item()
                        total_batches += 1
                    step_bar.update()

            mean_loss = loss_sum / total_batches if total_batches > 0 else 0.0
            
            # Log evaluation metrics
            logs_dict = {"eval_loss": mean_loss}
            # Simple all_reduce mean 
            # (assuming balanced batches, otherwise we need weighted average, 
            # but for logging this is usually fine)
            for k in logs_dict.keys():
                logs_dict[k] = self.strategy.all_reduce(logs_dict[k]) / self.strategy.get_world_size() # Average across ranks?
            
            if self.strategy.is_rank_0():
                self.strategy.print(f"Eval steps {steps}: Loss={logs_dict['eval_loss']}")
                
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs_dict, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs_dict.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        finally:
            self.model.train()


