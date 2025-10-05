import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class LunarDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None, use_photon_counting=True):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.clean_images = sorted(os.listdir(clean_dir))
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.transform = transform
        self.use_photon_counting = use_photon_counting

        assert len(self.clean_images) == len(self.noisy_images), \
            f"Clean ({len(self.clean_images)}) and Noisy ({len(self.noisy_images)}) counts do not match!"

    def __len__(self):
        return len(self.clean_images)

    def apply_photon_counting(self, img_array):
        """
        Simulate photon counting statistics for lunar images.
        Applies Poisson noise model typical of low-light astronomical imaging.
        """
        # Normalize to [0, 1] range
        img_normalized = img_array / 255.0
        
        # Scale to photon counts (simulate low-light lunar conditions)
        # Typical lunar surface reflects ~10-15% of incident light
        photon_scale = 50.0  # Average photons per pixel
        photon_counts = img_normalized * photon_scale
        
        # Apply Poisson noise (photon counting statistics)
        noisy_photons = np.random.poisson(photon_counts)
        
        # Normalize back to [0, 255]
        noisy_img = np.clip(noisy_photons / photon_scale * 255.0, 0, 255)
        
        return noisy_img.astype(np.uint8)

    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])

        clean_img = Image.open(clean_path).convert("L")
        noisy_img = Image.open(noisy_path).convert("L")
        
        # Apply photon counting noise model if enabled
        if self.use_photon_counting:
            noisy_array = np.array(noisy_img)
            noisy_array = self.apply_photon_counting(noisy_array)
            noisy_img = Image.fromarray(noisy_array)

        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)

        return noisy_img, clean_img


# CRITICAL FIX: Resolution options based on hardware
# For GPU: Use 600x600 for best quality
# For CPU: Use 256x256 for reasonable training speed

# Choose ONE of these based on your hardware:

# Option 1: Full resolution (REQUIRES GPU, VERY SLOW ON CPU)
transform_full = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor()
])

# Option 2: Balanced (Works on CPU, decent quality)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Good balance of quality and speed
    transforms.ToTensor()
])

# Option 3: Fast training (CPU-friendly, lower quality)
transform_fast = transforms.Compose([
    transforms.Resize((128, 128)),  # Fast training on CPU
    transforms.ToTensor()
])