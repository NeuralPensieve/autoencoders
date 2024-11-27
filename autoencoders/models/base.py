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

        # Store input parameters
        self.input_width, self.input_height = input_size
        self.latent_dim = latent_dim

        # Calculate the number of downsampling steps (each step reduces dimension by 2)
        self.n_downsample = 4
        self.max_filter_size = 512

        # Calculate final encoder output dimensions
        self.final_encoder_height = self.input_height // (2**self.n_downsample)
        self.final_encoder_width = self.input_width // (2**self.n_downsample)
        print(
            f"Final encoder output dimensions: {self.final_encoder_width}x{self.final_encoder_height}"
        )

        # Ensure dimensions are valid
        if self.final_encoder_height == 0 or self.final_encoder_width == 0:
            raise ValueError(
                f"Input size {input_size} is too small for {self.n_downsample} downsampling steps"
            )

        # Calculate flattened dimension for fully connected layers
        self.flat_dim = (
            self.max_filter_size * self.final_encoder_height * self.final_encoder_width
        )
        print(f"Final encoder output dimensions: {self.flat_dim} (flattened).")

        # Common encoder and decoder components
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # Create scheduler parameters
        self.initial_lr = args.lr

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.initial_lr, weight_decay=1e-5
        )

        # Initialize scheduler based on args
        self._init_scheduler(args)

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

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size,
                kernel_num,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num,
                kernel_num,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _build_encoder(self):
        return nn.Sequential(
            self._conv(3, 64),
            self._conv(64, 128),
            self._conv(128, 256),
            self._conv(256, self.max_filter_size),
        )

    def _build_decoder(self):
        return nn.Sequential(
            self._deconv(self.max_filter_size, 256),
            self._deconv(256, 128),
            self._deconv(128, 64),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Upsample(
                size=(self.input_width, self.input_height),
                mode="bilinear",
                align_corners=True,
            ),
            nn.Sigmoid(),
        )

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


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        W=89,
        H=109,
    ):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.upsample = nn.Upsample(
            size=(int(W), int(H)),
            mode="bilinear",
            align_corners=True,
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        x = self._conv_trans_2(x)
        return self.upsample(x)
