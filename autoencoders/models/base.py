import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, NamedTuple


class AutoencoderOutput(NamedTuple):
    reconstruction: torch.Tensor
    latent: torch.Tensor
    parameters: Optional[dict] = None


class LossOutput(NamedTuple):
    total_loss: float
    components: Optional[dict] = None


class BaseAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dim, args):
        super(BaseAutoencoder, self).__init__()

        self.device = args.device
        self.input_size = input_size
        self.latent_dim = latent_dim

    def _init_scheduler(self, args):
        """Initialize the learning rate scheduler based on arguments"""
        if not hasattr(args, "lr_schedule"):
            self.scheduler = None
            return

        if args.lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=args.min_lr
            )
        elif args.lr_schedule == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=args.epochs // 3,  # decrease LR every 1/3 of total epochs
                gamma=0.1,
            )
        elif args.lr_schedule == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.85
            )
        elif args.lr_schedule == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",  # reduce LR when metric stops decreasing
                factor=0.5,  # multiply LR by this factor
                patience=2,  # number of epochs with no improvement after which LR will be reduced
                min_lr=args.min_lr,  # lower bound on the learning rate
                threshold=0.05,  # minimum change in loss to qualify as an improvement (1%)
                threshold_mode="rel",  # interpret threshold as a relative value
            )
        else:
            self.scheduler = None

    def get_output_size(self):
        """Returns the expected output size of the autoencoder"""
        return (self.input_width, self.input_height)

    def _compute_loss(
        self, output: AutoencoderOutput, target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the loss for the autoencoder. Should be implemented by subclasses.
        Returns:
            total_loss: The total loss to be backpropagated
            loss_components: A dictionary containing the individual loss components
        """
        raise NotImplementedError("Subclasses must implement _compute_loss method")

    def training_step(self, images: torch.Tensor) -> LossOutput:
        """
        Perform a single training step.
        """
        self.train()
        images = images.to(self.device)

        # Forward pass
        output = self(images)

        # Compute loss
        loss_output = self._compute_loss(output, images)

        # Backprop
        self.optimizer.zero_grad()
        loss_output.total_loss.backward()

        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        return loss_output

    def get_current_lr(self):
        """
        Get the current learning rate
        """
        return self.optimizer.param_groups[0]["lr"]

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
