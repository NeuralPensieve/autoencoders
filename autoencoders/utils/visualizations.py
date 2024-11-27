import random
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import Subset


def plot_reconstructions(model, dataloader, device, epoch, track):
    """Plot 10 random samples and their reconstructions"""
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images[:10].to(device)
        output = model(images)
        reconstructed = output.reconstruction

        # Create figure
        fig, axes = plt.subplots(10, 2, figsize=(8, 20))
        plt.subplots_adjust(wspace=0.1)

        for i in range(10):
            # Original
            orig_img = images[i].cpu().numpy().transpose(1, 2, 0)
            axes[i, 0].imshow(orig_img)
            axes[i, 0].axis("off")
            if i == 0:
                axes[i, 0].set_title("Original")

            # Reconstructed
            recon_img = reconstructed[i].cpu().numpy().transpose(1, 2, 0)
            axes[i, 1].imshow(recon_img)
            axes[i, 1].axis("off")
            if i == 0:
                axes[i, 1].set_title("Reconstructed")

        # Save and log to wandb
        # plt.savefig(f'reconstructions_epoch_{epoch+1}.png')
        if track:
            wandb.log(
                {
                    "visualizations/reconstructions": wandb.Image(
                        plt, caption=f"Reconstructions at Epoch {epoch+1}"
                    )
                }
            )
        plt.close()


def visualize_similar_images(
    model, dataloader, device, epoch, track=False, n=5, sample_size=0.1
):
    """
    Creates a nxn collage with a random reference image and its n**2-1 most similar images
    based on latent space embeddings. Uses batch processing and data sampling for memory efficiency.

    Args:
        model: The VAE or autoencoder model
        dataloader: DataLoader containing the images
        device: torch device
        epoch: Current epoch number
        track: Whether to log to wandb
        sample_size: Fraction of dataset to sample (between 0 and 1)
    """
    model.eval()

    # TODO: Fix the use of model.encoder

    N = n**2 - 1

    # Calculate number of samples
    dataset_size = len(dataloader.dataset)
    num_samples = int(dataset_size * sample_size)

    # Randomly sample indices
    all_indices = list(range(dataset_size))
    sampled_indices = random.sample(all_indices, num_samples)

    # Create a subset of the dataset
    subset_dataset = Subset(dataloader.dataset, sampled_indices)
    subset_loader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
    )

    with torch.no_grad():
        # Lists to store embeddings and corresponding image indices
        all_embeddings = []
        all_image_indices = []

        # Process batches
        for batch_idx, (images, _) in enumerate(subset_loader):
            images = images.to(device)

            # Get embeddings
            if hasattr(model, "encode_mu"):
                embeddings = model.encode_mu(images)
            else:
                embeddings = model.encoder(images)

            # Flatten embeddings if needed (in case they're not already flattened)
            embeddings = embeddings.view(embeddings.size(0), -1)

            # Store embeddings and corresponding indices
            all_embeddings.append(embeddings.cpu())
            all_image_indices.extend(
                sampled_indices[
                    batch_idx * dataloader.batch_size : (batch_idx + 1)
                    * dataloader.batch_size
                ]
            )

        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Randomly select reference image index
        ref_idx = random.randint(0, len(all_embeddings) - 1)
        ref_embedding = all_embeddings[ref_idx]

        # Calculate cosine similarity efficiently
        ref_embedding_norm = F.normalize(ref_embedding.unsqueeze(0), p=2, dim=1)
        all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
        similarities = torch.mm(ref_embedding_norm, all_embeddings_norm.t())[0]

        # Get indices of top N most similar images
        _, top_indices = similarities.topk(N)  # Get 9 (including the reference)

        # Get the actual dataset indices for the similar images
        similar_dataset_indices = [all_image_indices[i] for i in top_indices]

        # Load the selected images
        selected_images = []
        for idx in similar_dataset_indices:
            img, _ = dataloader.dataset[idx]
            selected_images.append(img)

        # Create grid
        grid_images = torch.stack(selected_images)
        grid = make_grid(grid_images, nrow=3, normalize=True, padding=2)

        # Plot
        plt.figure(figsize=(3 * n, 3 * n))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title(
            f"Similar Images (Epoch {epoch+1})\nReference image (top-left) and its {N} nearest neighbors"
        )

        # Save or log to wandb
        if track:
            import wandb

            wandb.log({"similar_images": wandb.Image(plt), "epoch": epoch + 1})

        plt.close()


def visualize_latent_space(model, dataloader, device, epoch, track):
    """
    Visualize the latent space using PCA or t-SNE
    """
    model.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            output = model(images)
            latent_vectors.append(output.latent.cpu())
            labels.extend(lbls.numpy())

            if len(latent_vectors) * len(images) >= 1000:  # Limit to 1000 samples
                break

    latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
    labels = np.array(labels)

    # Perform t-SNE
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)

    # Plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap="tab20")
    plt.colorbar(scatter)
    plt.title(f"Latent Space Visualization (Epoch {epoch})")

    if track:
        wandb.log({"latent_space": wandb.Image(plt), "epoch": epoch})

    plt.close()
