import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, NamedTuple

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm, trange

from autoencoders.utils import (
    MetricsTracker,
    setup_wandb,
    log_to_wandb,
    setup_model,
    setup_dataloader,
    save_checkpoint,
    evaluate_and_visualize,
)

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    metrics_tracker: MetricsTracker,
    epoch: int,
    total_epochs: int,
    args: Any
) -> None:
    """Train model for one epoch."""
    model.train()
    metrics_tracker.reset()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs}')
    total_steps = len(dataloader)
    
    for batch_idx, (images, _) in enumerate(pbar, 1):
        loss_output = model.training_step(images)
        batch_size = len(images)
        
        metrics = {
            'loss': loss_output.total_loss,
            **loss_output.components
        }
        metrics_tracker.update(batch_size, **metrics)
        
        current_averages = metrics_tracker.get_averages()
        pbar.set_postfix({'loss': f'{current_averages["loss"]:.4f}'})
        
        if args.track and batch_idx < total_steps:
            log_to_wandb('train', {k: v/batch_size for k, v in metrics.items()})

def train(args: Any) -> None:
    """Main training loop."""
    # Setup run name and logging
    args.exp_name = f"{args.data_name}_{args.model_name}"
    args.run_name = f"{args.exp_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    setup_wandb(args)
    model = setup_model(args)
    dataloader = setup_dataloader(args)
    metrics_tracker = MetricsTracker()
    
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):.2e}")
    
    for epoch in trange(args.epochs):
        train_epoch(model, dataloader, metrics_tracker, epoch, args.epochs, args)
        
        # Update learning rate
        if model.scheduler:
            if isinstance(model.scheduler, ReduceLROnPlateau):
                model.scheduler.step(metrics_tracker.metrics['loss'])
            else:
                model.scheduler.step()
        
        # Log epoch metrics
        epoch_metrics = metrics_tracker.get_averages()
        epoch_metrics['learning_rate'] = model.get_current_lr()
        
        if args.track:
            log_to_wandb('epoch', epoch_metrics, epoch + 1)
        
        # Periodic evaluation and checkpointing
        if (epoch + 1) % 2 == 0:
            evaluate_and_visualize(model, dataloader, epoch, args)
            save_checkpoint(model, epoch, args)