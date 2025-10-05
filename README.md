# 🌙 Lunar Image Enhancement using Photon Counting & GAN-based Denoising

## 📌 Overview

This project enhances noisy lunar images captured from **Permanently Shadowed Regions (PSRs)** of the Moon.  
Since these regions receive almost no sunlight, images suffer from **low visibility and high noise**.

Our approach combines:
- **Photon Counting Simulation** → adds realistic low-light Poisson noise  
- **GAN-based Denoising** → U-Net Generator + PatchGAN Discriminator for detail-preserving enhancement  
- **Evaluation Metrics** → PSNR & SSIM to measure quality improvements  

---

## 📂 Project Structure

├── dataset_loader.py # Loads dataset, applies photon counting, preprocessing
├── GAN_GUI.py # Defines UNet Generator & PatchGAN Discriminator
├── train.py # Training loop for GAN (with L1, MSE, GAN loss)
├── testing.py # Testing + PSNR/SSIM evaluation pipeline
├── model_loader.py # Robust model loading (handles CPU/GPU mismatch & missing keys)
└── README.md # Documentation


---

## ⚙️ Setup

### 1. Install Requirements

```
pip install torch torchvision pillow opencv-python scikit-image matplotlib numpy
```
### 2. Dataset Structure

Organize your dataset like this:

TRAINING_DATA/
├── clean/   # Ground truth clean lunar images
└── noisy/   # Corresponding noisy images

Note: Filenames in both folders must match (e.g., img1.jpg in both).
---
## 🧠 Training
### Run training with:

```python train.py```

### Training Configuration:

  Default: 256×256 resolution (CPU-friendly)
  
  GPU mode: change to 600×600 inside dataset_loader.py
  
  Model checkpoints saved as:
  
  generator_epoch_X.pth
  
  discriminator_epoch_X.pth
  
  Final: generator_final.pth, discriminator_final.pth
---
## 🖼️ Testing & Evaluation

### Run the testing pipeline:

```python testing.py```

**Testing Pipeline:** 
- Loads generator_final.pth
- Enhances noisy lunar images and saves results in OUTPUT/
- Evaluates with **PSNR** (clarity) and **SSIM** (structural similarity)
**Example Output:**
enhanced_output_1.jpg vs img1.jpg → SSIM: 0.8234
PSNR (Noisy vs Enhanced): 28.42 dB
---
## 📊 Workflow
![Flowchart](https://github.com/1sanemax/Lunar-PSR-Enhancement/blob/main/flowchart.jpg) 
--- 
## 📈 Results & Benefits 
- Produces **clearer lunar images** from shadowed regions
- Preserves fine **surface details** while reducing noise
- **Low-cost, reproducible** using open-source tools
- Useful for **lunar research, exploration, and resource mapping**
---
## 📚 References 
- A. K. Dagar, *Analysis of the Permanently Shadowed Region of Cabeus Crater*, Planetary and Space Science, 2023. [Link](https://www.sciencedirect.com/science/article/abs/pii/S0019103523003391)
- Oxford University, *Peering into the Moon's Permanently Shadowed Regions with AI*. [Link](https://www.ox.ac.uk/news/features/peering-moons-permanently-shadowed-regions-ai#:~:text=The%20Moon's%20polar%20regions%20are,resolution%20for%20the%20first%20time)
- ResearchGate, *Rolling bearing fault diagnosis method based on 2D grayscale images and Wasserstein Generative Adversarial Nets under unbalanced sample condition*. [Link](https://www.researchgate.net/publication/373270947_Rolling_bearing_fault_diagnosis_method_based_on_2D_grayscale_images_and_Wasserstein_Generative_Adversarial_Nets_under_unbalanced_sample_condition)
- ML4PhysicalSciences, *Wasserstein GANs for Physical Science Imaging (NeurIPS ML4PS 2020)*. [Link](https://ml4physicalsciences.github.io/2020/files/NeurIPS_ML4PS_2020_43.pdf)
---
## 🖼️ Results 
Here are some sample results from the project: 
### Noisy Input vs Enhanced Output 
| Noisy Image | Enhanced Image | 
|-------------|----------------| 
| ![Noisy](https://github.com/1sanemax/Lunar-PSR-Enhancement/blob/main/Noisy_example.jpg) | ![Enhanced](https://github.com/1sanemax/Lunar-PSR-Enhancement/blob/main/Enhanced_of_noisy.jpg) | 
### Side-by-Side Example 
![Comparison](https://github.com/1sanemax/Lunar-PSR-Enhancement/blob/main/Example_comparison_of_Noisy_vs_Enhanced.png) 
### Metrics of Enhanced Image 
![Metrics](https://github.com/1sanemax/Lunar-PSR-Enhancement/blob/main/Example_metrics_of_Enhanced.png) 
--- 
## 🧱 Build as Executable (Optional)
### To create a standalone .exe for offline use:
```pip install pyinstaller```
```pyinstaller --onefile testing.py```
Your executable will appear in the dist/ folder (e.g., dist/testing.exe).
You can run it directly to enhance lunar images without needing Python installed.
---
## 👩‍🚀 Authors 
This project is inspired by **Problem Statement ID 1732 (Enhancement of Permanently Shadowed Regions of Lunar Craters)** from **Smart India Hackathon 2024**. 
However, the current implementation is **independent work** and not part of the SIH submission. 
**Team:** - Surya Narayanan S - Shankar Balaji V 
--- 
## 🤝 Contributing 
Contributions are welcome! Please feel free to submit a Pull Request. 
--- 
## 📄 License
This project is open-source and available under the Apache License 2.0. 
--- 
## 📧 Contact
For questions, feedback, or collaboration opportunities, please open an issue in the repository. Or reach out directly at: 
- 📩 Email: suryanarayanan@ieee.org
- 🔗 LinkedIn: [SuryaNarayanan](https://www.linkedin.com/in/suryanarayanan3329/)
