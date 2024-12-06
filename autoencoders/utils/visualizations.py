import random
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
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


def create_interpolations(z, n):
    # Create n-1 interpolations between the first and last embeddings
    n += 1
    z_interpolations = [z[0]]
    for i in range(1, n):
        alpha = i / (n)
        z_interpolations.append((1 - alpha) * z[0] + alpha * z[-1])

    z_interpolations.append(z[-1])

    return torch.stack(z_interpolations)


def plot_interpolated_reconstrunctions(
    model, dataloader, device, epoch, track, image_ids, N=5
):
    """Plot N interpolation samples between all possible pairs of images.

    Args:
        model: The VAE or similar model
        dataloader: Dataset loader
        device: torch device
        epoch: Current epoch number
        track: Boolean for wandb tracking
        image_ids: List of image IDs to interpolate between
        N: Number of interpolation steps between each pair
    """
    from itertools import combinations

    M = len(image_ids)
    if M < 2:
        raise ValueError("Need at least 2 image IDs to perform interpolation")

    model.eval()
    with torch.no_grad():
        # Get all images from the dataset
        sample_images = []
        for img_id in image_ids:
            sample_images.append(dataloader.dataset.__getitem__(img_id)[0])

        # Convert to tensor
        sample_images = torch.stack(sample_images)

        # Get embeddings for all images
        z = model.get_embeddings(sample_images.to(device))

        # Calculate number of pairs
        num_pairs = (M * (M - 1)) // 2

        # Create figure
        # Each row needs N+4 columns (2 originals, N interpolations, 2 reconstructions)
        fig, axes = plt.subplots(num_pairs, N + 4, figsize=(3 * (N + 4), 3 * num_pairs))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

        # If only one row, wrap axes in list to make indexing consistent
        if num_pairs == 1:
            axes = [axes]

        # For each pair of images
        for row, (i, j) in enumerate(combinations(range(M), 2)):
            # Get embeddings for this pair
            z_pair = torch.stack([z[i], z[j]])

            # Create interpolations
            z_interp = create_interpolations(z_pair, N)

            # Get reconstructions
            reconstructions = model.get_reconstructions(z_interp)

            # Plot original first image
            orig_img = sample_images[i].cpu().numpy().transpose(1, 2, 0)
            axes[row][0].imshow(orig_img)
            axes[row][0].axis("off")
            axes[row][0].set_title(f"Image {i+1}")

            # Plot original second image
            orig_img = sample_images[j].cpu().numpy().transpose(1, 2, 0)
            axes[row][N + 3].imshow(orig_img)
            axes[row][N + 3].axis("off")
            axes[row][N + 3].set_title(f"Image {j+1}")

            # Plot reconstructions and interpolations
            for k in range(N + 2):
                recon_img = reconstructions[k].cpu().numpy().transpose(1, 2, 0)
                axes[row][k + 1].imshow(recon_img)
                axes[row][k + 1].axis("off")
                if k == 0 or k == N + 1:
                    axes[row][k + 1].set_title("Reconstructed")

        plt.suptitle(f"Interpolations between {M} images - Epoch {epoch+1}")

        # Save and log to wandb
        if track:
            wandb.log(
                {
                    "visualizations/interpolations": wandb.Image(
                        plt, caption=f"Interpolations at Epoch {epoch+1}"
                    )
                }
            )
        plt.close()

    # img = []
    # for i in range(25):
    #     img.append(dataloader.dataset.__getitem__(i)[0])

    # # plot the two images using matplotlib, and save them to wandb
    # fig, axes = plt.subplots(5, 5, figsize=(8, 4))
    # plt.subplots_adjust(wspace=0.1)
    # for i in range(25):
    #     axes[i // 5, i % 5].imshow(img[i].cpu().numpy().transpose(1, 2, 0))
    #     axes[i // 5, i % 5].axis("off")

    # # save image to file
    # plt.savefig(f"original_images.png")


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
