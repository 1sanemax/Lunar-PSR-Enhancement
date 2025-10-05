import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from GAN_GUI import UNetGenerator
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from model_loader import load_model_state

# Calculate quality metrics
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_ssim_pairs(enhanced_dir, noisy_dir, limit=200):
    # Get sorted list of images
    enhanced_images = sorted([os.path.join(enhanced_dir, f) for f in os.listdir(enhanced_dir) if f.endswith(".jpg")])[:limit]
    noisy_images = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith(".png") or f.endswith(".jpg")])[:limit]
    
    results = []
    for e_img, n_img in zip(enhanced_images, noisy_images):
        # Read both images
        img1 = cv2.imread(e_img, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(n_img, cv2.IMREAD_GRAYSCALE)

        # Resize noisy image to match enhanced size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute SSIM
        score, _ = ssim(img1, img2, full=True)
        results.append((os.path.basename(e_img), os.path.basename(n_img), score))
    
    return results

# Load model
G = UNetGenerator(features=32).to(device)
if load_model_state(G, "generator_final.pth", device):
    print("✅ Generator weights loaded successfully.")
else:
    raise RuntimeError("❌ Failed to load generator weights — check path/format or architecture.")
G.eval()

# Transform - MUST match training resolution
transform = transforms.Compose([
    transforms.Resize((600, 600)),  # Keep original 600x600 resolution
    transforms.ToTensor(),
])

# Load noisy image
#for i in range(0, 200):  # Change range for multiple images
img_path = rf"D:/ARGON/TRAINING_DATA/noisy/img188.jpg" #replace with lunar image path
img = Image.open(img_path).convert("L")
    #print(f"Original image size: {img.size}")

noisy_tensor = transform(img).unsqueeze(0).to(device)
    #print(f"Input tensor shape: {noisy_tensor.shape}")

    # Enhance
with torch.no_grad():
    enhanced = G(noisy_tensor)

   # print(f"Output tensor shape: {enhanced.shape}")

    # Convert back to image - preserve full resolution
inter_img = enhanced.squeeze(0).squeeze(0).cpu().numpy()

    # Denormalize from [-1, 1] to [0, 1]
inter_img = (inter_img + 1) / 2.0
inter_img = np.clip(inter_img, 0, 1)

    # Convert original image to numpy for fair comparison
noisy_np = np.array(img.resize((600, 600))) / 255.0

    # Display results at FULL resolution
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

axes[0].imshow(noisy_np, cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"Noisy Input", fontsize=14)
axes[0].axis('off')

    # Show difference
encd = np.abs(inter_img - noisy_np)
axes[1].imshow(encd, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Enhanced Output", fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Save enhanced image at full resolution
enhanced_pil = Image.fromarray((encd * 255).astype(np.uint8))
output_dir = r"D:/ARGON/OUTPUT" #replace with lunar image path where you want it stored

    # Make sure the folder exists
os.makedirs(output_dir, exist_ok=True)

    # Auto-increment filename
'''i = 1
while os.path.exists(os.path.join(output_dir, f"enhanced_output_{i}.jpg")):
    i += 1
filename = os.path.join(output_dir, f"enhanced_output_{i}.jpg")

    # Save file    
enhanced_pil.save(filename)'''
    #print(f"Enhanced image saved as '{filename}' with size {enhanced_pil.size}")

    # Optional: Calculate and display quality metrics
print(f"\nQuality Assessment:")
print(f"Enhanced image statistics:")
print(f"  Mean: {encd.mean():.4f}")    
print(f"  Std: {encd.std():.4f}")
print(f"  Min: {encd.min():.4f}")
print(f"  Max: {encd.max():.4f}")
print(f"PSNR (Noisy vs Enhanced): {calculate_psnr(noisy_np, encd):.4f} dB")
enhanced_dir = rf"D:/ARGON/OUTPUT/" #replace with lunar image path where it is stored
noisy_dir = rf"D:/ARGON/TRAINING_DATA/noisy/" #replace with lunar image path
ssim_results = compute_ssim_pairs(enhanced_dir, noisy_dir)
#display 188th image score
for e, n, score in ssim_results[:]:
    if "img188" in e:

        print(f"{e} vs {n} → SSIM: {score:.4f}")

