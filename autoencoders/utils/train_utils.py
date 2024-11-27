import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

from autoencoders.utils.datasets import CustomImageDataset
from autoencoders.models import VAE, VanillaAutoencoder, VQVAE
from autoencoders.utils.visualizations import (
    plot_reconstructions,
    visualize_latent_space,
    visualize_similar_images,
)


@dataclass
class MetricsTracker:
    """Tracks and accumulates metrics during training."""

    total_samples: int = 0
    metrics: Dict[str, float] = None

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset metrics for new epoch."""
        self.total_samples = 0
        self.metrics = {}

    def update(self, batch_size: int, **metrics: float) -> None:
        """
        Update metrics with batch results.

        Args:
            batch_size: Number of samples in the batch
            **metrics: Metric name and value pairs to update
        """
        self.total_samples += batch_size
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0
            self.metrics[key] += value * batch_size

    def get_averages(self) -> Dict[str, float]:
        """Calculate averaged metrics over all samples."""
        return {key: value / self.total_samples for key, value in self.metrics.items()}


def setup_wandb(args: Any) -> None:
    """Initialize Weights & Biases logging."""
    if not args.track:
        return

    wandb.init(
        project=args.wandb_project,
        config=args,
        name=args.run_name,
    )
    wandb.run.log_code("../")


def log_to_wandb(
    prefix: str, metrics: Dict[str, float], epoch: Optional[int] = None
) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        prefix: Metric prefix for grouping
        metrics: Dictionary of metrics to log
        epoch: Optional epoch number
    """
    if not wandb.run:
        return

    log_data = {f"{prefix}/{k}": v for k, v in metrics.items()}
    if epoch is not None:
        log_data["epoch"] = epoch
    wandb.log(log_data)


def setup_model(args: Any) -> torch.nn.Module:
    """Initialize model based on arguments."""
    model_map = {
        "vae": VAE,
        "vanilla": VanillaAutoencoder,
        "vqvae": VQVAE,
    }

    if args.model_name == "vqvae":
        args.latent_dim = args.embedding_dim

    model_class = model_map[args.model_name]
    new_size = (args.W // args.downsize, args.H // args.downsize)

    model = model_class(
        input_size=new_size,
        latent_dim=args.latent_dim,
        args=args,
    ).to(args.device)

    return model


def setup_dataloader(args: Any) -> DataLoader:
    """Initialize data loader with preprocessing."""
    new_size = (args.W // args.downsize, args.H // args.downsize)
    transform = transforms.Compose([transforms.Resize(new_size), transforms.ToTensor()])

    dataset = CustomImageDataset(
        root_dir=f"{args.data_folder}", transform=transform, limit=args.limit
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )


def calculate_data_variance(args: Any, batch_size: int) -> float:
    """Calculate data variance."""
    args.batch_size = batch_size
    data_loader = setup_dataloader(args)
    data = next(iter(data_loader))[0]

    return torch.var(data)


def save_checkpoint(model: torch.nn.Module, epoch: int, args: Any) -> None:
    """Save model checkpoint."""
    if not args.checkpoint:
        return

    checkpoint_dir = f"artifacts/{args.run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/vae_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), checkpoint_path)

    if args.track:
        wandb.save(checkpoint_path)


def evaluate_and_visualize(
    model: torch.nn.Module, dataloader: DataLoader, epoch: int, args: Any
) -> None:
    """Generate evaluation metrics and visualizations."""
    # if args.model_name == "vqvae" and args.use_ema:
    #     model.vector_quantizer.toggle_ema_update(enabled=False)

    model.eval()
    plot_reconstructions(model, dataloader, args.device, epoch, args.track)

    if args.visualize_latent:
        visualize_latent_space(model, dataloader, args.device, epoch, args.track)

    # if args.visualize_similar:
    #     visualize_similar_images(
    #         model,
    #         dataloader,
    #         args.device,
    #         epoch,
    #         args.track,
    #         sample_size=args.visualize_similar_sample,
    #     )

    # if args.model_name == "vqvae" and args.use_ema:
    #     model.vector_quantizer.toggle_ema_update(enabled=True)
