import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoders.models import BaseAutoencoder, AutoencoderOutput, LossOutput


class VAE(BaseAutoencoder):
    def __init__(self, input_size, latent_dim, args):
        super().__init__(input_size, latent_dim, args)
        
        # Latent space layers
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_var = nn.Linear(self.flat_dim, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flat_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x = self.decoder_input(z)
        x = x.view(x.size(0), self.max_filter_size, self.final_height, self.final_width)
        reconstruction = self.decoder(x)
        
        return AutoencoderOutput(
            reconstruction=reconstruction,
            latent=z,
            parameters={'mu': mu, 'log_var': log_var}
        )

    def _compute_loss(self, output: AutoencoderOutput, target: torch.Tensor) -> LossOutput:
        batch_size = target.size(0)
        mu = output.parameters['mu']
        log_var = output.parameters['log_var']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(output.reconstruction, target, reduction='sum') / batch_size
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        
        # Dynamic KL weight adjustment based on target KL
        kl_weight = 1.0  # base weight
        target_kl = 0.5  # target KL value
        kl_weight = kl_weight * (1.0 + torch.abs(kl_div - target_kl))
        
        # Total loss
        total_loss = recon_loss + kl_weight * kl_div
        
        return LossOutput(
            total_loss=total_loss,
            components={
                'recon_loss': recon_loss, 
                'kl_loss': kl_div, 
                'kl_weight': kl_weight
                }
        )