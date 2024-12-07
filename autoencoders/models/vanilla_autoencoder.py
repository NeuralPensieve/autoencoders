import torch
import torch.nn.functional as F

from autoencoders.models import (
    BaseAutoencoder,
    AutoencoderOutput,
    LossOutput,
    BaseEncoder,
    BaseDecoder,
)


class VanillaEncoder(BaseEncoder):
    def forward(self, x):
        return self._encode(x)


class VanillaDecoder(BaseDecoder):
    def forward(self, z):
        return self._decode(z)


class VanillaAutoencoder(BaseAutoencoder):
    def __init__(self, input_size, latent_dim, args):
        super().__init__(input_size, latent_dim, args)
        self.encoder = VanillaEncoder(hidden_dims=args.hidden_dims)
        self.decoder = VanillaDecoder(
            hidden_dims=args.hidden_dims, output_size=input_size
        )

        # Initialize optimizer after model is constructed
        self._init_optimizer()

    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return AutoencoderOutput(reconstruction=reconstruction, latent=z)

    def get_embeddings(self, x):
        return self.encoder(x)

    def get_reconstructions(self, z):
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
