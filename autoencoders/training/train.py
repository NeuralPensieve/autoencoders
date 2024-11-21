import os
import wandb
import pprint
from tqdm import tqdm, trange
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from autoencoders.models.models import VAE, VanillaAutoencoder
from autoencoders.utils.datasets import CustomImageDataset
from autoencoders.utils.utils import (
    plot_reconstructions, 
    visualize_latent_space, 
    visualize_similar_images,
)


def train(args):
    # create unique run_name with date and time
    args.exp_name = f"{args.data_name}_{args.model_name}"
    args.run_name = f"{args.exp_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # pretty print args using tyro
    print(f"Training with the following args:")
    pprint.pprint(vars(args))

    model_map = {
        'vae': VAE,
        'vanilla': VanillaAutoencoder,
    }
    ae_model = model_map[args.model_name]

    # Initialize wandb with args
    if args.track:
         wandb.init(
             project="imagenet-vae", 
             config=args,
             name=args.run_name,
             )
         wandb.run.log_code("../")

    # Device configuration
    args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    new_size = (args.W // args.downsize, args.H // args.downsize)
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor()
    ])
    
    # Load dataset
    trainset = CustomImageDataset(root_dir=f"{args.data_folder}", transform=transform, limit=args.limit)
    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        persistent_workers=True,
        pin_memory=True,
        )
    
    # Initialize model with args instead of just lr
    model = ae_model(
        input_size=new_size, 
        latent_dim=args.latent_dim,
        args=args,  # Pass the entire args object
        ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params:.2e}")
    
    # Training loop
    for epoch in trange(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        # Create progress bar for each epoch
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for images, _ in pbar:
            # Training loop
            loss_output = model.training_step(images)
            loss = loss_output.total_loss
            extra_losses = loss_output.components
                
            # Update running losses
            epoch_loss += loss

            pbar.set_postfix({'loss': f'{loss/len(images):.4f}'})

            if args.model_name == 'vae':
                epoch_recon_loss += extra_losses['recon_loss']
                epoch_kl_loss += extra_losses['kl_loss']

            # log data to wandb
            if args.track:
                wandb.log({'train/loss': loss / len(images)})
                if args.model_name == 'vae':
                    wandb.log({
                        'train/recon_loss': extra_losses['recon_loss'] / len(images),
                        'train/kl_loss': extra_losses['kl_loss'] / len(images)
                    })

        # Step the scheduler after each epoch
        if model.scheduler:
            if isinstance(model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                model.scheduler.step(loss)
            else:
                model.scheduler.step()
        
        # Get current learning rate from the model
        current_lr = model.get_current_lr()
        
        # Log epoch metrics to wandb
        log_data = {
            'epoch': epoch + 1,
            'epoch/loss': epoch_loss / len(trainset),
            'epoch/learning_rate': current_lr  # Use the current learning rate from the model
        }
        if args.model_name == 'vae':
            log_data.update({
                'epoch/recon_loss': epoch_recon_loss / len(trainset),
                'epoch/kl_loss': epoch_kl_loss / len(trainset),
            })
        if args.track:
            wandb.log(log_data)
            
        # Generate and log visualizations
        if (epoch + 1) % 2 == 0:
            plot_reconstructions(model, trainloader, args.device, epoch, args.track)
            if args.visualize_latent:
                visualize_latent_space(model, trainloader, args.device, epoch, args.track)
            if args.visualize_similar:
                visualize_similar_images(
                    model, trainloader, args.device, epoch, 
                    args.track, sample_size=args.visualize_similar_sample
                    )

            if args.checkpoint:
                checkpoint_dir = f'artifacts/{args.run_name}'
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = f'{checkpoint_dir}/vae_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                if args.track:
                    wandb.save(checkpoint_path)