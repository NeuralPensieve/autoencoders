import torch
import torch.nn as nn
from typing import Optional, NamedTuple


class AutoencoderOutput(NamedTuple):
    reconstruction: torch.Tensor
    latent: torch.Tensor
    parameters: Optional[dict] = None


class LossOutput(NamedTuple):
    total_loss: float
    components: Optional[dict] = None


class BaseAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dim, args):
        super().__init__()
        self.device = args.device
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.initial_lr = args.lr
        self.args = args

    def _init_optimizer(self):
        """Initialize optimizer after model is fully constructed"""
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.initial_lr, weight_decay=1e-5
        )
        self._init_scheduler(self.args)

    def _init_scheduler(self, args):
        if not hasattr(args, "lr_schedule"):
            self.scheduler = None
            return

        scheduler_map = {
            "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=args.min_lr
            ),
            "step": lambda: torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=args.epochs // 3,
                gamma=0.1,
            ),
            "exponential": lambda: torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.85
            ),
            "plateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=args.min_lr,
                threshold=0.05,
                threshold_mode="rel",
            ),
        }

        self.scheduler = scheduler_map.get(args.lr_schedule, lambda: None)()

    def training_step(self, images: torch.Tensor) -> LossOutput:
        self.train()
        images = images.to(self.device)

        # Forward pass
        output = self(images)

        # Compute loss
        loss_output = self._compute_loss(output, images)

        # Backprop
        self.optimizer.zero_grad()
        loss_output.total_loss.backward()
        self.optimizer.step()

        return loss_output

    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def set_data_variance(self, data_variance):
        self.data_variance = data_variance

    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        """
        Forward pass of the model. Should be implemented by subclasses.
        Must return an AutoencoderOutput object.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings of the model. Should be implemented by subclasses.
        Must return an tensor.
        """
        raise NotImplementedError

    def get_reconstructions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the reconstructions of the model. Should be implemented by subclasses.
        Must return an tensor.
        """
        raise NotImplementedError

    def get_output_size(self):
        """Returns the expected output size of the autoencoder"""
        return (self.input_width, self.input_height)
