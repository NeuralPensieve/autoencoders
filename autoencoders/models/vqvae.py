import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoders.models import BaseAutoencoder, AutoencoderOutput, LossOutput

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def _calculate_distances(self, flat_input):
        """Calculate distances between input vectors and embeddings."""
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        return distances

    def _get_encodings(self, distances, device):
        """Get encoding indices and one-hot encodings from distances."""
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=device)
        encodings.scatter_(1, encoding_indices, 1)
        return encodings, encoding_indices

    def _quantize(self, encodings, input_shape):
        """Quantize the input using encodings."""
        return torch.matmul(encodings, self.embedding.weight).view(input_shape)

    def _preprocess_inputs(self, inputs):
        """Convert inputs from BCHW to BHWC and flatten."""
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        return inputs, input_shape, flat_input

    def _postprocess_quantized(self, quantized, inputs):
        """Apply straight-through estimator and convert back to BCHW."""
        quantized = inputs + (quantized - inputs).detach()
        return quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs):
        raise NotImplementedError("Subclasses must implement forward method")


class VectorQuantizer(BaseVectorQuantizer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__(num_embeddings, embedding_dim, commitment_cost)

    def _compute_loss(self, quantized, inputs):
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        return loss

    def forward(self, inputs):
        # Preprocess inputs
        inputs, input_shape, flat_input = self._preprocess_inputs(inputs)
        
        # Calculate distances and get encodings
        distances = self._calculate_distances(flat_input)
        encodings, encoding_indices = self._get_encodings(distances, inputs.device)
        
        # Quantize
        quantized = self._quantize(encodings, input_shape)
        
        # Compute loss
        loss = self._compute_loss(quantized, inputs)
        
        # Postprocess and return
        quantized = self._postprocess_quantized(quantized, inputs)
        return quantized, loss, encoding_indices.view(input_shape[:-1])


def ema_inplace(moving_avg, new, decay):
    """Update exponential moving average in-place"""
    moving_avg.data.mul_(decay)
    moving_avg.data.add_((1 - decay) * new)


class VectorQuantizerEMA(BaseVectorQuantizer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super().__init__(num_embeddings, embedding_dim, commitment_cost)
        self.decay = decay
        self.epsilon = epsilon

        # Initialize EMA variables
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        self.register_buffer('ema_update_ready', torch.ones(1))

    def _update_ema(self, encodings, flat_input):
        """Update embeddings using exponential moving average."""
        if self.training and self.ema_update_ready:
            # Calculate cluster sizes
            ema_inplace(self.ema_cluster_size, encodings.sum(0), self.decay)
            
            # Calculate total size and apply Laplace smoothing
            n = self.ema_cluster_size.sum()
            cluster_size = ((self.ema_cluster_size + self.epsilon) /
                          (n + self.num_embeddings * self.epsilon) * n)
            
            # Update cluster centers
            dw = torch.matmul(encodings.t(), flat_input)
            ema_inplace(self.ema_w, dw, self.decay)
            
            # Update embedding weights
            self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)

    def forward(self, inputs):
        # Preprocess inputs
        inputs, input_shape, flat_input = self._preprocess_inputs(inputs)
        
        # Calculate distances and get encodings
        distances = self._calculate_distances(flat_input)
        encodings, encoding_indices = self._get_encodings(distances, inputs.device)
        
        # Update EMA
        self._update_ema(encodings, flat_input)
        
        # Quantize
        quantized = self._quantize(encodings, input_shape)
        
        # Compute loss (simpler than non-EMA version)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Postprocess and return
        quantized = self._postprocess_quantized(quantized, inputs)
        return quantized, loss, encoding_indices.view(input_shape[:-1])

    def init_ema(self):
        """Initialize EMA variables at the start of training"""
        self.ema_cluster_size.data.fill_(0)
        self.ema_w.data.copy_(self.embedding.weight.data)
        self.ema_update_ready.data.fill_(1)

    def toggle_ema_update(self, enabled=True):
        """Enable or disable EMA updates"""
        self.ema_update_ready.data.fill_(1 if enabled else 0)


class VQVAE(BaseAutoencoder):
    def __init__(self, input_size, latent_dim, args):
        super().__init__(input_size, latent_dim, args)
        
        # VQ-VAE specific parameters
        self.num_embeddings = args.num_embeddings
        self.commitment_cost = args.commitment_cost
        self.use_ema = args.use_ema
        self.ema_decay = args.ema_decay
        
        # Override encoder's final layer to match embedding dimension
        self.pre_quantization = nn.Conv2d(
            self.max_filter_size, latent_dim,
            kernel_size=1, stride=1
        )
        
        # Vector Quantizer (EMA or regular based on args)
        if self.use_ema:
            self.vector_quantizer = VectorQuantizerEMA(
                num_embeddings=self.num_embeddings,
                embedding_dim=latent_dim,
                commitment_cost=self.commitment_cost,
                decay=self.ema_decay
            )
        else:
            self.vector_quantizer = VectorQuantizer(
                num_embeddings=self.num_embeddings,
                embedding_dim=latent_dim,
                commitment_cost=self.commitment_cost
            )
        
        # Post quantization projection
        self.post_quantization = nn.Conv2d(
            latent_dim, self.max_filter_size,
            kernel_size=1, stride=1
        )

    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        # Encode
        encoded = self.encoder(x)
        
        # Pre-quantization projection
        z = self.pre_quantization(encoded)
        
        # Vector quantization
        quantized, vq_loss, encoding_indices = self.vector_quantizer(z)
        
        # Post-quantization projection
        x = self.post_quantization(quantized)
        
        # Decode
        reconstruction = self.decoder(x)
        
        return AutoencoderOutput(
            reconstruction=reconstruction,
            latent=quantized,
            parameters={
                'vq_loss': vq_loss,
                'encoding_indices': encoding_indices,
                'z': z
            }
        )

    def _compute_loss(self, output: AutoencoderOutput, target: torch.Tensor) -> LossOutput:
        batch_size = target.size(0)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(output.reconstruction, target, reduction='sum') / batch_size
        
        # VQ loss from the vector quantizer
        vq_loss = output.parameters['vq_loss']
        
        # Calculate perplexity
        encoding_indices = output.parameters['encoding_indices']
        avg_probs = torch.mean(
            F.one_hot(encoding_indices, self.num_embeddings).float(), dim=0
        )
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Calculate codebook usage
        unique_indices = torch.unique(encoding_indices)
        codebook_usage = len(unique_indices) / self.num_embeddings
        
        # Total loss
        total_loss = recon_loss + vq_loss
        
        return LossOutput(
            total_loss=total_loss,
            components={
                'recon_loss': recon_loss,
                'vq_loss': vq_loss,
                'perplexity': perplexity,
                'codebook_usage': codebook_usage
            }
        )