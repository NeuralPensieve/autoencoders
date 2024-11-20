import os
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np


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
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title('Original')
            
            # Reconstructed
            recon_img = reconstructed[i].cpu().numpy().transpose(1, 2, 0)
            axes[i, 1].imshow(recon_img)
            axes[i, 1].axis('off')
            if i == 0:
                axes[i, 1].set_title('Reconstructed')
        
        # Save and log to wandb
        # plt.savefig(f'reconstructions_epoch_{epoch+1}.png')
        if track:
            wandb.log({
                "visualizations/reconstructions": wandb.Image(
                    plt, caption=f'Reconstructions at Epoch {epoch+1}'
                )
            })
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
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab20')
    plt.colorbar(scatter)
    plt.title(f'Latent Space Visualization (Epoch {epoch})')
    
    if track:
        wandb.log({
            "latent_space": wandb.Image(plt),
            "epoch": epoch
        })
    
    plt.close()
