# 🌙 Lunar Image Enhancement using Photon Counting & GAN-based Denoising

## 📌 Overview

This project enhances noisy lunar images captured from **Permanently Shadowed Regions (PSRs)** of the Moon. Since these regions receive almost no sunlight, images suffer from **low visibility and high noise**.

Our approach combines:
- **Photon Counting Simulation** → adds realistic low-light Poisson noise
- **GAN-based Denoising** → U-Net Generator + PatchGAN Discriminator for detail-preserving enhancement
- **Evaluation Metrics** → PSNR & SSIM to measure quality improvements

---

## 📂 Project Structure

```
├── dataset_loader.py     # Loads dataset, applies photon counting, preprocessing
├── GAN_GUI.py            # Defines UNet Generator & PatchGAN Discriminator
├── train.py              # Training loop for GAN (with L1, MSE, GAN loss)
├── testing.py            # Testing pipeline + evaluation metrics (PSNR, SSIM)
├── ORION_SPACE_SYSTEMS_SIH_1732.pptx   # SIH presentation (reference)
└── README.md             # Documentation
```

---

## ⚙️ Setup

### 1. Install Requirements

```bash
pip install torch torchvision pillow opencv-python scikit-image matplotlib numpy
```

### 2. Dataset Structure

Organize your dataset like this:

```
TRAINING_DATA/
├── clean/   # Ground truth clean lunar images
└── noisy/   # Corresponding noisy images
```

**Important:** Filenames in both folders must match (e.g., `img1.jpg` in both).

---

## 🧠 Training

Run training with:

```bash
python train.py
```

**Training Configuration:**
- By default, uses **256×256 resolution** (CPU-friendly)
- For GPU training, switch to **600×600 resolution** inside `dataset_loader.py`
- Model checkpoints saved as:
  - `generator_epoch_X.pth`
  - `discriminator_epoch_X.pth`
  - Final models: `generator_final.pth`, `discriminator_final.pth`

---

## 🖼️ Testing & Evaluation

Enhance images and compute metrics:

```bash
python testing.py
```

**Testing Pipeline:**
- Loads `generator_final.pth`
- Enhances noisy lunar images and saves results in `OUTPUT/`
- Evaluates with **PSNR** (clarity) and **SSIM** (structural similarity)

**Example Output:**

```
enhanced_output_1.jpg vs img1.jpg → SSIM: 0.8234
PSNR (Noisy vs Enhanced): 28.42 dB
```

---

## 📊 Workflow

```
Dataset
   ↓
Photon Counting Simulation
   ↓
Preprocessing
   ↓
UNet Generator + PatchGAN Discriminator
   ↓
Training Loop
   ↓
Save Generator
   ↓
Testing on Noisy Images
   ↓
Enhanced Outputs
   ↓
Evaluation: PSNR & SSIM
```

---

## 📈 Results & Benefits

- Produces **clearer lunar images** from shadowed regions
- Preserves fine **surface details** while reducing noise
- **Low-cost, reproducible** using open-source tools
- Useful for **lunar research, exploration, and resource mapping**

---

## 📚 References

- Oxford University: *AI-driven lunar shadow imaging* -
  https://www.researchgate.net/publication/373270947_Rolling_bearing_fault_diagnosis_method_based_on_2D_grayscale_images_and_Wasserstein_Generative_Adversarial_Nets_under_unbalanced_sample_condition
- Photon Counting Models in Astronomy -
  https://www.ox.ac.uk/news/features/peering-moons-permanently-shadowed-regions-ai#:~:text=The%20Moon's%20polar%20regions%20are,resolution%20for%20the%20first%20time
- GAN-based Denoising research papers -
  https://ml4physicalsciences.github.io/2020/files/NeurIPS_ML4PS_2020_43.pdf


---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is open-source and available under the Apache 2.0 License.

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue in the repository.
