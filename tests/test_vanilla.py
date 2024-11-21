import unittest
import torch
from autoencoders.models import AutoencoderOutput, VanillaAutoencoder

class Args:
    def __init__(self, device, lr, epochs, lr_schedule, min_lr):
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.lr_schedule = lr_schedule
        self.min_lr = min_lr


class TestVanillaAutoencoder(unittest.TestCase):
    def test_forward(self):
        input_size = (32, 32)
        latent_dim = 10
        args = Args(device=torch.device('cpu'), lr=0.001, epochs=10, lr_schedule='cosine', min_lr=0.0001)
        model = VanillaAutoencoder(input_size, latent_dim, args)
        images = torch.randn(1, 3, 32, 32)
        output = model(images)
        self.assertIsInstance(output, AutoencoderOutput)
        self.assertIsNotNone(output.reconstruction)
        self.assertIsNotNone(output.latent)
        self.assertIsNone(output.parameters)

    def test_compute_loss(self):
        input_size = (32, 32)
        latent_dim = 10
        args = Args(device=torch.device('cpu'), lr=0.001, epochs=10, lr_schedule='cosine', min_lr=0.0001)
        model = VanillaAutoencoder(input_size, latent_dim, args)
        images = torch.randn(1, 3, 32, 32)
        output = model(images)
        loss_output = model._compute_loss(output, images)
        self.assertIsNotNone(loss_output.total_loss)
        self.assertIsNotNone(loss_output.components)


if __name__ == '__main__':
    unittest.main()