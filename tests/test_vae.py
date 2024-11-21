import unittest
import torch
from autoencoders.models import AutoencoderOutput, VAE

class Args:
    def __init__(self, device, lr, epochs, lr_schedule, min_lr):
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.lr_schedule = lr_schedule
        self.min_lr = min_lr


class TestVAE(unittest.TestCase):
    def test_forward(self):
        input_size = (32, 32)
        latent_dim = 10
        args = Args(device=torch.device('cpu'), lr=0.001, epochs=10, lr_schedule='cosine', min_lr=0.0001)
        model = VAE(input_size, latent_dim, args)
        images = torch.randn(1, 3, 32, 32)
        output = model(images)
        self.assertIsInstance(output, AutoencoderOutput)
        self.assertIsNotNone(output.reconstruction)
        self.assertIsNotNone(output.latent)
        self.assertIsNotNone(output.parameters)
        self.assertIn('mu', output.parameters)
        self.assertIn('log_var', output.parameters)

    def test_reparameterize(self):
        input_size = (32, 32)
        latent_dim = 10
        args = Args(device=torch.device('cpu'), lr=0.001, epochs=10, lr_schedule='cosine', min_lr=0.0001)
        model = VAE(input_size, latent_dim, args)
        mu = torch.randn(1, latent_dim)
        log_var = torch.randn(1, latent_dim)
        z = model.reparameterize(mu, log_var)
        self.assertIsNotNone(z)
        self.assertEqual(z.shape, (1, latent_dim))

    def test_compute_loss(self):
        input_size = (32, 32)
        latent_dim = 10
        args = Args(device=torch.device('cpu'), lr=0.001, epochs=10, lr_schedule='cosine', min_lr=0.0001)
        model = VAE(input_size, latent_dim, args)
        images = torch.randn(1, 3, 32, 32)
        output = model(images)
        loss, loss_components = model._compute_loss(output, images)
        self.assertIsNotNone(loss)
        self.assertIsNotNone(loss_components)
        self.assertIn('total_loss', loss_components)
        self.assertIn('recon_loss', loss_components)
        self.assertIn('kl_loss', loss_components)
        self.assertIn('kl_weight', loss_components)


if __name__ == '__main__':
    unittest.main()