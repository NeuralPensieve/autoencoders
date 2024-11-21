import tyro
from dataclasses import dataclass

from autoencoders.training import train

@dataclass
class Args:
    """Configuration class for VAE training parameters."""
    model_name: str = 'vanilla'
    """Identifier for the model architecture. Options: 'vanilla', 'vae'"""
    data_folder: str = 'data/celeba_aligned'
    """Root directory containing the image dataset"""
    data_name: str = 'celebA'
    """Name of the dataset being used"""
    cuda: bool = True
    """Flag to enable CUDA training"""
    H: int = 218
    """Original image height dimension"""
    W: int = 178
    """Original image width dimension"""
    downsize: int = 2
    """Factor by which to reduce image dimensions"""
    epochs: int = 50
    """Number of training epochs"""
    batch_size: int = 256
    """Number of samples per training batch"""
    latent_dim: int = 256
    """Dimension of the latent space representation"""
    track: bool = False
    """Flag to enable training progress tracking"""
    limit: int = None
    """Optional limit on number of training samples (None for full dataset)"""
    checkpoint: bool = False
    """Flag to enable model checkpointing during training"""
    lr: float = 1e-3
    """Initial learning rate for optimization"""
    min_lr: float = 1e-5
    """Minimum learning rate threshold"""
    lr_schedule: str = 'plateau'
    """Learning rate scheduling strategy. Options: 'cosine', 'step', 'exponential', 'plateau'"""
    visualize_similar: bool = True
    """Flag to enable visualization of similar images"""
    visualize_latent: bool = False
    """Flag to enable visualization of latent space"""
    visualize_similar_sample: float = 0.1
    """Fraction of dataset to use for similarity visualization"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)