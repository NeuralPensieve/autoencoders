import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoders.models import (
    BaseAutoencoder,
    AutoencoderOutput,
    LossOutput,
)


class VAEEncoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        latent_dim=128,
        hidden_dims=[64, 128, 256, 512],
    ):
        super().__init__()

        # Build encoder convolutional layers
        modules = []
        in_channels = input_channels

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                    # Add residual connections
                    nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # 1x1 convolutions to project to latent space
        self.conv_mu = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=1)
        self.conv_log_var = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)
        return mu, log_var


class VAEDecoder(nn.Module):
    def __init__(
        self,
        output_size,
        latent_dim=128,
        output_channels=3,
        hidden_dims=[64, 128, 256, 512],
    ):
        super().__init__()

        # Reverse hidden_dims
        hidden_dims = hidden_dims[::-1]

        # Initial 1x1 convolution to project from latent space
        self.initial_conv = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(),
        )

        # Build decoder layers
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    # Add additional convolution for better feature learning
                    nn.Conv2d(
                        hidden_dims[i + 1], hidden_dims[i + 1], kernel_size=3, padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(output_channels),
                nn.Sigmoid(),
            )
        )

        self.decoder = nn.Sequential(*modules)

        self.upsample = nn.Upsample(
            size=(output_size[0], output_size[1]),
            mode="bilinear",
            align_corners=True,
        )

    def forward(self, z):
        x = self.initial_conv(z)
        return self.upsample(self.decoder(x))


class VAE(BaseAutoencoder):
    def __init__(self, input_size, latent_dim, args):
        super().__init__(input_size, latent_dim, args)

        self.encoder = VAEEncoder(hidden_dims=args.hidden_dims)
        self.decoder = VAEDecoder(hidden_dims=args.hidden_dims, output_size=input_size)

        # Create scheduler parameters
        self.initial_lr = args.lr

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.initial_lr, weight_decay=1e-5
        )

        # Initialize scheduler based on args
        self._init_scheduler(args)

        self.free_bits = torch.tensor(args.free_bits).to(args.device)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        # Encode
        mu, log_var = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        reconstruction = self.decoder(z)

        return AutoencoderOutput(
            reconstruction=reconstruction,
            latent=z,
            parameters={"mu": mu, "log_var": log_var},
        )

    def _compute_loss(
        self, output: AutoencoderOutput, target: torch.Tensor
    ) -> LossOutput:
        mu = output.parameters["mu"]
        log_var = output.parameters["log_var"]

        recon_loss = F.mse_loss(output.reconstruction, target) / self.data_variance

        # Compute KL per dimension
        kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

        # Apply free bits constraint per dimension
        kl_per_dim = torch.max(kl_per_dim, self.free_bits)
        kl_div = torch.mean(kl_per_dim)

        total_loss = recon_loss + kl_div

        return LossOutput(
            total_loss=total_loss,
            components={
                "recon_loss": recon_loss,
                "kl_loss": kl_div,
            },
        )

    def set_data_variance(self, data_variance):
        self.data_variance = data_variance
