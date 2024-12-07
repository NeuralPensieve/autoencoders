import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoders.models import (
    AutoencoderOutput,
    LossOutput,
    BaseAutoencoder,
    BaseEncoder,
    BaseDecoder,
)


class VQVAEEncoder(BaseEncoder):
    def forward(self, x):
        return self._encode(x)


class VQVAEDecoder(BaseDecoder):
    def forward(self, z):
        return self._decode(z)


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return (
            loss,
            quantized.permute(0, 3, 1, 2).contiguous(),
            perplexity,
            encodings,
            encoding_indices,
        )


class VQVAE(BaseAutoencoder):
    def __init__(self, input_size, latent_dim, args):
        super().__init__(input_size, latent_dim, args)

        self.num_embeddings = args.num_embeddings

        self.vector_quantizer = VectorQuantizerEMA(
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            commitment_cost=args.commitment_cost,
            decay=args.ema_decay,
        )

        self.encoder = VQVAEEncoder(hidden_dims=args.hidden_dims)
        self.decoder = VQVAEDecoder(
            hidden_dims=args.hidden_dims, output_size=input_size
        )

        # Initialize optimizer after model is constructed
        self._init_optimizer()

    def forward(self, x):
        z = self.encoder(x)
        vq_loss, quantized, perplexity, encodings, encoding_indices = (
            self.vector_quantizer(z)
        )
        x_recon = self.decoder(quantized)

        # Calculate codebook usage
        unique_indices = torch.unique(encoding_indices)
        codebook_usage = len(unique_indices) / self.num_embeddings

        return AutoencoderOutput(
            reconstruction=x_recon,
            latent=quantized,
            parameters={
                "vq_loss": vq_loss,
                "encoding_indices": encoding_indices,
                "encodings": encodings,
                "perplexity": perplexity,
                "codebook_usage": codebook_usage,
            },
        )

    def get_embeddings(self, x):
        z = self.encoder(x)
        vq_loss, quantized, perplexity, encodings, encoding_indices = (
            self.vector_quantizer(z)
        )

        return quantized

    def get_reconstructions(self, z):
        return self.decoder(z)

    def _compute_loss(
        self, output: AutoencoderOutput, target: torch.Tensor
    ) -> LossOutput:
        recon_loss = F.mse_loss(output.reconstruction, target)
        vq_loss = output.parameters["vq_loss"]
        total_loss = recon_loss / self.data_variance + vq_loss

        return LossOutput(
            total_loss=total_loss,
            components={
                "recon_loss": recon_loss,
                "vq_loss": vq_loss,
                "perplexity": output.parameters["perplexity"],
                "codebook_usage": output.parameters["codebook_usage"],
            },
        )
