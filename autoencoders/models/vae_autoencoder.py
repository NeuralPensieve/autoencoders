import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoders.models import (
    BaseAutoencoder,
    AutoencoderOutput,
    LossOutput,
    BaseEncoder,
    BaseDecoder,
)


# vae.py
class VAEEncoder(BaseEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dims = kwargs.get("hidden_dims", [32, 64, 128, 256])
        latent_dim = kwargs.get("latent_dim", 128)

        # Additional VAE-specific layers
        self.conv_mu = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=1)
        self.conv_log_var = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=1)

    def forward(self, x):
        x = self._encode(x)
        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)
        return mu, log_var


class VAEDecoder(BaseDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dims = kwargs.get("hidden_dims", [32, 64, 128, 256])[::-1]
        latent_dim = kwargs.get("latent_dim", 128)

        # Additional VAE-specific layers
        self.initial_conv = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(),
        )

    def forward(self, z):
        x = self.initial_conv(z)
        return self._decode(x)


class VAE(BaseAutoencoder):
    def __init__(self, input_size, latent_dim, args):
        super().__init__(input_size, latent_dim, args)
        self.encoder = VAEEncoder(hidden_dims=args.hidden_dims)
        self.decoder = VAEDecoder(hidden_dims=args.hidden_dims, output_size=input_size)

        self.free_bits = torch.tensor(args.free_bits, requires_grad=False).to(
            args.device
        )
        self.free_bits_gamma = args.free_bits_gamma

        # Initialize optimizer after model is constructed
        self._init_optimizer()

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return AutoencoderOutput(
            reconstruction=reconstruction,
            latent=z,
            parameters={"mu": mu, "log_var": log_var},
        )

    def get_embeddings(self, x):
        mu, log_var = self.encoder(x)
        return self.reparameterize(mu, log_var)

    def get_reconstructions(self, z):
        return self.decoder(z)

    def _compute_loss(
        self, output: AutoencoderOutput, target: torch.Tensor
    ) -> LossOutput:
        mu = output.parameters["mu"]
        log_var = output.parameters["log_var"]

        recon_loss = F.mse_loss(output.reconstruction, target)
        kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_per_dim_max = torch.max(kl_per_dim, self.free_bits)
        kl_div = torch.mean(kl_per_dim_max)

        total_loss = recon_loss / self.data_variance + kl_div
        self.free_bits = self.free_bits * self.free_bits_gamma

        return LossOutput(
            total_loss=total_loss,
            components={
                "recon_loss": recon_loss.item(),
                "kl_loss": kl_div.item(),
                "std_mean_ratio": torch.mean(
                    torch.exp(0.5 * log_var) / mu.abs()
                ).item(),
                "mean_mu": torch.mean(mu.abs()).item(),
                "mean_std": torch.mean(torch.exp(0.5 * log_var)).item(),
                "mean_kl_per_dim": torch.mean(kl_per_dim).item(),
                "min_kl_per_dim": torch.mean(
                    torch.amin(kl_per_dim, dim=(1, 2, 3))
                ).item(),
                "max_kl_per_dim": torch.mean(
                    torch.amax(kl_per_dim, dim=(1, 2, 3))
                ).item(),
                "free_bits": self.free_bits.item(),
            },
        )
