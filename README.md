# Autoencoders Package

This package provides a collection of autoencoder implementations for image processing tasks, with a focus on the CelebA dataset. The package includes various autoencoder architectures such as Vanilla Autoencoder, Variational Autoencoder (VAE), and VQ-VAE.

## Features
- Multiple autoencoder architectures (Vanilla, VAE, VQ-VAE)
- Common encoder and decoder architectures among all implementations, allowing to compare their performance
- Integration with Weights & Biases (wandb.ai) for experiment tracking
- Configurable model architectures and training parameters
- Various visualization tools for better understanding of the model's behavior

## Dataset
We utilize the CelebA dataset (aligned) for training and testing various autoencoders. You can download the dataset from:

https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Installation
To install the package, run the following command:
```bash
pip install -e .
```

## Usage
To train a model, use the training script with your desired configuration:

```bash
python scripts/train_model.py \
    --exp_name "my_experiment" \
    --model_name vae \
    --data_folder "path/to/celeba_aligned" \
    --wandb_project "autoencoders" \
    --epochs 30 \
    --batch_size 256 \
    --latent_dim 256 \
    --track True
```

### Key Parameters
- `model_name`: Choose between 'vanilla', 'vae', or 'vqvae'
- `data_folder`: Path to your dataset
- `latent_dim`: Dimension of the latent space (default: 256)
- `hidden_dims`: Architecture dimensions [32, 64, 128, 256]
- `lr`: Learning rate (default: 2e-4)
- `lr_schedule`: Learning rate scheduler ('cosine', 'step', 'exponential', 'plateau')
- `batch_size`: Training batch size (default: 256)
- `epochs`: Number of training epochs (default: 30)
- `track`: Enable Weights & Biases tracking
- `checkpoint`: Enable model checkpointing

## Contributing
Contributions are welcome! If you'd like to add a new autoencoder implementation or improve existing ones, please submit a pull request.

## License
This package is released under the MIT License.