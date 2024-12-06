import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoders.models import BaseAutoencoder, AutoencoderOutput, LossOutput


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        latent_dim=128,
        hidden_dims=[32, 64, 128, 256],
    ):
        super().__init__()

        # Build encoder convolutional layers
        modules = []
        in_channels = input_channels

        for h_dim in hidden_dims[:-1]:
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

        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, hidden_dims[-1], kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.ReLU(),
                # Add residual connections
                nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
            )
        )

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(
        self,
        output_size,
        latent_dim=128,
        output_channels=3,
        hidden_dims=[32, 64, 128, 256],
    ):
        super().__init__()

        # Reverse hidden_dims
        hidden_dims = hidden_dims[::-1]

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
        return self.upsample(self.decoder(z))


class VanillaAutoencoder(BaseAutoencoder):
    def __init__(self, input_size, latent_dim, args):
        super().__init__(input_size, latent_dim, args)

        self.encoder = Encoder(hidden_dims=args.hidden_dims)
        self.decoder = Decoder(hidden_dims=args.hidden_dims, output_size=input_size)

        # Create scheduler parameters
        self.initial_lr = args.lr

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.initial_lr, weight_decay=1e-5
        )

        # Initialize scheduler based on args
        self._init_scheduler(args)

    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return AutoencoderOutput(reconstruction=reconstruction, latent=z)

    def get_embeddings(self, x):
        # Encode
        return self.encoder(x)

    def get_reconstructions(self, z):
        # Decode
        return self.decoder(z)

    def _compute_loss(
        self, output: AutoencoderOutput, target: torch.Tensor
    ) -> LossOutput:
        recon_loss = F.mse_loss(output.reconstruction, target)

        total_loss = recon_loss / self.data_variance
        return LossOutput(
            total_loss=total_loss,
            components={"recon_loss": recon_loss},
        )

    def set_data_variance(self, data_variance):
        self.data_variance = data_variance
