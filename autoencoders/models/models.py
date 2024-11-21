import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
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
        self.input_height, self.input_width = input_size
        self.latent_dim = latent_dim
        
        # Calculate the number of downsampling steps (each step reduces dimension by 2)
        self.n_downsample = 4
        self.max_filter_size = 512
        
        # Calculate final encoder output dimensions
        self.final_height = self.input_height // (2 ** self.n_downsample)
        self.final_width = self.input_width // (2 ** self.n_downsample)
        print(f"Final encoder output dimensions: {self.final_height}x{self.final_width}")
        
        # Ensure dimensions are valid
        if self.final_height == 0 or self.final_width == 0:
            raise ValueError(f"Input size {input_size} is too small for {self.n_downsample} downsampling steps")
        
        # Calculate flattened dimension for fully connected layers
        self.flat_dim = self.max_filter_size * self.final_height * self.final_width
        print(f"Final encoder output dimensions: {self.flat_dim} (flattened).")
        
        # Common encoder and decoder components
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # Create scheduler parameters
        self.initial_lr = args.lr

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr, weight_decay=1e-5)

        # Initialize scheduler based on args
        self._init_scheduler(args)

    def _init_scheduler(self, args):
        """Initialize the learning rate scheduler based on arguments"""
        if not hasattr(args, 'lr_schedule'):
            self.scheduler = None
            return
            
        if args.lr_schedule == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=args.epochs,
                eta_min=args.min_lr
            )
        elif args.lr_schedule == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=args.epochs // 3,  # decrease LR every 1/3 of total epochs
                gamma=0.1
            )
        elif args.lr_schedule == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.85
            )
        elif args.lr_schedule == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',          # reduce LR when metric stops decreasing
                factor=0.5,          # multiply LR by this factor
                patience=2,          # number of epochs with no improvement after which LR will be reduced
                min_lr=args.min_lr,  # lower bound on the learning rate
                verbose=True         # print message when LR is reduced
            )
        else:
            self.scheduler = None

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=4, stride=2, padding=1,
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
            nn.Upsample(size=(self.input_height, self.input_width), mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )

    def get_output_size(self):
        """Returns the expected output size of the autoencoder"""
        return (self.input_height, self.input_width)

    def _compute_loss(self, output: AutoencoderOutput, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
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
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self(images)
        
        # Compute loss
        total_loss, loss_components = self._compute_loss(output, images)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()

        return LossOutput(
            total_loss=total_loss.item(),
            components={k: v.item() if isinstance(v, torch.Tensor) else v 
                       for k, v in loss_components.items()}
        )
    
    def scheduler_step(self):
        """
        Step the learning rate scheduler. Call this after each epoch.
        """
        if self.scheduler is not None:
            self.scheduler.step()

    def get_current_lr(self):
        """
        Get the current learning rate
        """
        return self.optimizer.param_groups[0]['lr']

    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        """
        Forward pass of the model. Should be implemented by subclasses.
        Must return an AutoencoderOutput object.
        """
        raise NotImplementedError("Subclasses must implement forward method")


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

    def _compute_loss(self, output: AutoencoderOutput, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        batch_size = target.size(0)

        recon_loss = F.mse_loss(output.reconstruction, target, reduction='sum') / batch_size
        return recon_loss, {'recon_loss': recon_loss}


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

    def _compute_loss(self, output: AutoencoderOutput, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
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
        
        return total_loss, {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_div,
            'kl_weight': kl_weight
        }