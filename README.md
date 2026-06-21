<p align="center">
  <h1 align="center">RoWSFormer — Custom Reproduction</h1>
</p>

---

> **Disclaimer:** This is an **independent, custom reproduction** of the paper
> *"RoWSFormer: A Robust Watermarking Framework with Swin Transformer for Enhanced Geometric Attack Resilience"*
> by Weitong Chen & Yuheng Li ([arXiv:2409.14829](https://arxiv.org/abs/2409.14829)).
> This repository is **not** an official implementation by the original authors.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Training](#-training)
- [Citation](#-citation)

---

## 🔍 Overview

Most deep-learning-based watermarking methods rely on Convolutional Neural Networks (CNNs), which excel at capturing local spatial features but struggle with **geometric distortions** (rotation, scaling, affine transforms) due to their inherently limited receptive field.

**RoWSFormer** addresses this limitation by leveraging the **Swin Transformer** architecture, whose window-based self-attention mechanism captures **global and long-range spatial relationships**. This makes the watermark embedding and extraction pipeline significantly more resilient to geometric attacks while maintaining high imperceptibility (PSNR).

### What This Repo Does

This repository provides a clean, modular PyTorch reproduction of the RoWSFormer framework:
- **Encoder** — embeds a 64-bit binary watermark into a cover image using a U-Net-style architecture built with Swin Transformer blocks.
- **Noise Layer** — simulates real-world image distortions (geometric & non-geometric attacks) during training.
- **Decoder** — extracts the watermark from a (potentially attacked) watermarked image.
- **3-Phase Training** — progressive training strategy for stable convergence.

---

## 📁 Project Structure

```
RoWSFormer-custom-reproduction-main/
├── main.py                         # Entry point: 3-phase training loop + visualization
├── train.py                        # Training & validation logic with early stopping
├── requirements.txt                # Python dependencies
├── notebook/
│   └── rowsformer.ipynb            # Jupyter notebook for experiments (e.g., Colab)
├── models/
│   ├── blocks/
│   │   ├── encoder.py              # Encoder (U-Net architecture)
│   │   ├── decoder.py              # Decoder (watermark extractor)
│   │   └── noise_layer.py          # Differentiable noise/attack simulator
│   └── components/
│       ├── LCESTB.py               # Locally-Channel Enhanced Swin Transformer Block
│       ├── FETB.py                 # Frequency-Enhanced Transformer Block
│       ├── feature_extractor.py    # Local Feature Extractor (Conv2d)
│       ├── watermark_encoder.py    # Watermark → spatial feature map
│       └── layers/
│           ├── swin_transformer.py # Swin Transformer with window partition/reverse
│           ├── transformer.py      # Standard multi-head self-attention
│           └── window_attention.py # Window attention with relative position bias
└── utils/
    ├── dataset.py                  # CustomDataset (image loading + random watermark)
    ├── losses.py                   # WatermarkLoss (MSE + BCE + constraint)
    └── metrics.py                  # PSNR and bit accuracy metrics
```

---

## 📋 Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | 2.11.0 |
| torch-dct | 0.1.6 |
| NumPy | 2.0.2 |
| Kornia | 0.8.3 |
| tqdm | 4.67.3 |
| matplotlib | 3.10.0 |

---

## ⚙ Installation

```bash
# Clone the repository
git clone https://github.com/redezvous2004/RoWSFormer-custom-reproduction.git
cd RoWSFormer-custom-reproduction-main

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset

This reproduction uses the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) high-resolution image dataset.

> **Note:** Images are resized to **128 × 128** during training. The watermark is a random **64-bit** binary vector generated per sample. (For simple version)
---

## 🚀 Training

### 3-Phase Progressive Training Strategy

The training follows a progressive curriculum:

| Phase | Epochs | λ₁ (Image) | λ₂ (Watermark) | λ₃ (Constraint) | Attack | Goal |
|---|---|---|---|---|---|---|
| **Phase 1** | 10 | 0.0 | 1.0 | 0.5 | None | Learn watermark embedding/extraction only |
| **Phase 2** | 10 | 1.0 | 1.0 | 0.5 | None | Balance image quality and watermark accuracy |
| **Phase 3** | 15 | 5.0 | 1.0 | 0.5 | Specific | Add robustness against a specific attack |

### Running Training

```bash
python main.py
```

The script trains separate models for each attack type defined in `attack_list`:
```python
attack_list = ['none', 'salt_and_pepper_noise', 'rotation']
```

Each trained model is saved as `best_model_{attack_name}.pth`.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 2 × 10⁻⁴ |
| Weight Decay | 1 × 10⁻³ |
| LR Scheduler | Cosine Annealing (T_max=40, η_min=1 × 10⁻⁶) |
| Batch Size | 4 |
| Image Size | 128 × 128 |
| Watermark Length | 64 bits |
| Gradient Clipping | max_norm = 5.0 |
| Early Stopping | patience = 10 epochs |

### Model Architecture Hyperparameters

| Parameter | Value |
|---|---|
| Base Dimension (`dim`) | 64 |
| Number of Stages | 3 |
| Number of Attention Heads | 4 |
| Window Size (Swin) | 8 × 8 |
| Shift Size (SW-MSA) | 4 |

---

## 📚 Citation

If you use this code or find it helpful, please cite the original paper:

```bibtex
@article{chen2024rowsformer,
  title     = {RoWSFormer: A Robust Watermarking Framework with Swin Transformer 
               for Enhanced Geometric Attack Resilience},
  author    = {Chen, Weitong and Li, Yuheng},
  journal   = {arXiv preprint arXiv:2409.14829},
  year      = {2024},
  url       = {https://arxiv.org/abs/2409.14829}
}
```
