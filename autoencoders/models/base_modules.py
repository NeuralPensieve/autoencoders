import torch.nn as nn


class BaseEncoder(nn.Module):
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

    def _encode(self, x):
        return self.encoder(x)


class BaseDecoder(nn.Module):
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

    def _decode(self, z):
        return self.upsample(self.decoder(z))
