from datetime import datetime
from typing import Any
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from autoencoders.utils import (
    MetricsTracker,
    setup_wandb,
    log_to_wandb,
    setup_model,
    setup_dataloader,
    save_checkpoint,
    evaluate_and_visualize,
    calculate_data_variance,
)


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    metrics_tracker: MetricsTracker,
    epoch: int,
    total_epochs: int,
    args: Any,
) -> None:
    """Train model for one epoch."""
    model.train()
    metrics_tracker.reset()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
    total_steps = len(dataloader)

    for batch_idx, (images, _) in enumerate(pbar, 1):
        loss_output = model.training_step(images)
        batch_size = len(images)

        metrics = {"loss": loss_output.total_loss, **loss_output.components}
        metrics_tracker.update(batch_size, **metrics)

        current_averages = metrics_tracker.get_averages()

        # Update progress bar
        postfix = {"loss": f'{current_averages["loss"]:.4f}'}
        if "codebook_usage" in current_averages:
            postfix["codebook_usage"] = f'{current_averages["codebook_usage"]:.4f}'
        pbar.set_postfix(postfix)

        if args.track and batch_idx < total_steps:
            log_to_wandb("train", {k: v for k, v in metrics.items()})


def train(args: Any) -> None:
    """Main training loop."""
    # Setup run name and logging
    args.exp_name = f"{args.data_name}_{args.model_name}"
    args.run_name = f"{args.exp_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    setup_wandb(args)
    model = setup_model(args)
    dataloader = setup_dataloader(args)
    metrics_tracker = MetricsTracker()

    if args.data_variance is None:
        print("Calculating data variance...")
        batch_size = 1000 if args.limit else 10_000
        data_variance = calculate_data_variance(args, batch_size)
        model.set_data_variance(data_variance)
        print(f"Data variance: {data_variance:.4f}")
    else:
        model.set_data_variance(args.data_variance)

    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in trange(args.epochs):
        train_epoch(model, dataloader, metrics_tracker, epoch, args.epochs, args)

        # Update learning rate
        if model.scheduler:
            if isinstance(model.scheduler, ReduceLROnPlateau):
                model.scheduler.step(metrics_tracker.metrics["loss"])
            else:
                model.scheduler.step()

        # Log epoch metrics
        epoch_metrics = metrics_tracker.get_averages()
        epoch_metrics["learning_rate"] = model.get_current_lr()

        if args.track:
            log_to_wandb("epoch", epoch_metrics, epoch + 1)

        # Periodic evaluation and checkpointing
        if (epoch + 1) % 2 == 0:
            evaluate_and_visualize(model, dataloader, epoch, args)
            save_checkpoint(model, epoch, args)
