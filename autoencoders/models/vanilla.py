import torch
import torch.nn.functional as F
from typing import Tuple

from autoencoders.models import BaseAutoencoder, AutoencoderOutput, LossOutput

class VanillaAutoencoder(BaseAutoencoder):
    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        z = self.encoder(x)
        z_flat = z.view(z.size(0), -1)
        reconstruction = self.decoder(z)
        return AutoencoderOutput(
            reconstruction=reconstruction,
            latent=z_flat,
            parameters=None
        )

    def _compute_loss(self, output: AutoencoderOutput, target: torch.Tensor) -> LossOutput:
        batch_size = target.size(0)

        recon_loss = F.mse_loss(output.reconstruction, target, reduction='sum') / batch_size
        return LossOutput(
            total_loss=recon_loss,
            components={'recon_loss': recon_loss},
        )