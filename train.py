import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import LunarDataset, transform
from GAN_GUI import UNetGenerator, PatchDiscriminator
import os

if __name__ == '__main__':
    # Paths
    clean_dir = r"D:\ARGON\TRAINING_DATA\clean" #replace with your system path
    noisy_dir = r"D:\ARGON\TRAINING_DATA\noisy" #replace with your system path

    # Dataset + Loader with Photon Counting enabled
    train_dataset = LunarDataset(
        clean_dir, 
        noisy_dir, 
        transform=transform,
        use_photon_counting=True  # ENABLE PHOTON COUNTING
    )

    # FIXED: num_workers=0 for Windows CPU training
    # Batch size adjusted for CPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4 if device.type == 'cpu' else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("Training samples:", len(train_dataset))
    print(f"Batch size: {batch_size}")

    # Models - Use smaller features for CPU training
    
    print(f"Using device: {device}")
    
    # Adjust model size based on device
    model_features = 32 if device.type == 'cpu' else 64
    print(f"Model features: {model_features}")

    G = UNetGenerator(features=model_features).to(device)
    D = PatchDiscriminator(features=model_features).to(device)

    # Loss Functions
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    criterion_MSE = nn.MSELoss()  # Additional perceptual loss

    # Optimizers with better hyperparameters
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))  # Lower D lr

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)

    # Training Loop - Adjust epochs based on device
    epochs = 15
    lambda_L1 = 100  # L1 loss weight
    lambda_MSE = 10  # MSE loss weight
    
    print(f"Training for {epochs} epochs with photon counting noise model...")
    print("Note: Training on CPU will be slow.")
    for epoch in range(epochs):
        G.train()
        D.train()
        
        epoch_loss_G = 0
        epoch_loss_D = 0
        
        for i, (noisy, clean) in enumerate(train_loader):
            if i >= 100:
                break
            noisy, clean = noisy.to(device), clean.to(device)
            batch_size = noisy.size(0)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Real images
            pred_real = D(clean)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

            # Fake images
            fake = G(noisy)
            pred_fake = D(fake.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()

            fake = G(noisy)
            pred_fake = D(fake)

            # Adversarial loss
            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

            # Pixel-wise L1 loss (preserves structure)
            loss_G_L1 = criterion_L1(fake, clean)

            # MSE loss (enhances quality)
            loss_G_MSE = criterion_MSE(fake, clean)

            # Combined generator loss
            loss_G = loss_G_GAN + lambda_L1 * loss_G_L1 + lambda_MSE * loss_G_MSE

            loss_G.backward()
            optimizer_G.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i+1}/100] "
                      f"Loss_G: {loss_G.item():.4f} (GAN: {loss_G_GAN.item():.4f}, "
                      f"L1: {loss_G_L1.item():.4f}, MSE: {loss_G_MSE.item():.4f}) "
                      f"Loss_D: {loss_D.item():.4f}")
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        avg_loss_G = epoch_loss_G / len(train_loader)
        avg_loss_D = epoch_loss_D / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs} Complete | Avg Loss_G={avg_loss_G:.4f} | Avg Loss_D={avg_loss_D:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), f"generator_epoch_{epoch+1}.pth")
            torch.save(D.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

    # Save final model
    torch.save(G.state_dict(), "generator_final.pth")
    torch.save(D.state_dict(), "discriminator_final.pth")

    print("Training complete! Models saved.")
