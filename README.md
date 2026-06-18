<p align="center">
  <h1 align="center">RoWSFormer — Custom Reproduction</h1>
  <p align="center">
    <b>A Robust Watermarking Framework with Swin Transformer for Enhanced Geometric Attack Resilience</b>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2409.14829"><img src="https://img.shields.io/badge/arXiv-2409.14829-b31b1b.svg" alt="arXiv"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg" alt="PyTorch"></a>
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </p>
</p>

---

> **Disclaimer:** This is an **independent, custom reproduction** of the paper
> *"RoWSFormer: A Robust Watermarking Framework with Swin Transformer for Enhanced Geometric Attack Resilience"*
> by Weitong Chen & Yuheng Li ([arXiv:2409.14829](https://arxiv.org/abs/2409.14829)).
> This repository is **not** an official implementation by the original authors.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Evaluation & Visualization](#-evaluation--visualization)
- [Supported Attacks](#-supported-attacks)
- [Results (Paper)](#-results-paper)
- [Implementation Notes & Differences](#-implementation-notes--differences)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

Most deep-learning-based watermarking methods rely on Convolutional Neural Networks (CNNs), which excel at capturing local spatial features but struggle with **geometric distortions** (rotation, scaling, affine transforms) due to their inherently limited receptive field.

**RoWSFormer** addresses this limitation by leveraging the **Swin Transformer** architecture, whose window-based self-attention mechanism captures **global and long-range spatial relationships**. This makes the watermark embedding and extraction pipeline significantly more resilient to geometric attacks while maintaining high imperceptibility (PSNR).

### What This Repo Does

This repository provides a clean, modular PyTorch reproduction of the RoWSFormer framework:
- **Encoder** — embeds a 64-bit binary watermark into a cover image using a U-Net-style architecture built with Swin Transformer blocks.
- **Noise Layer** — simulates real-world image distortions (geometric & non-geometric attacks) during training.
- **Decoder** — extracts the watermark from a (potentially attacked) watermarked image.
- **3-Phase Training** — follows the paper's progressive training strategy for stable convergence.

---

## ✨ Key Contributions

| Contribution | Description |
|---|---|
| **LCESTB** (Locally-Channel Enhanced Swin Transformer Block) | Combines Swin Transformer's window-based self-attention with a channel-attention enhancement module (depthwise conv + SE-style squeeze-excite) for both global spatial modeling and local channel recalibration. |
| **FETB** (Frequency-Enhanced Transformer Block) | Uses standard multi-head self-attention followed by a frequency-domain (DCT-based) channel reweighting mechanism to reinforce robustness in the frequency domain. |
| **U-Net Encoder Architecture** | A multi-scale encoder with skip connections that fuses image features and watermark features at every resolution level. |
| **Progressive 3-Phase Training** | A curriculum training strategy that gradually introduces image quality loss and attack robustness. |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENCODER (U-Net)                          │
│                                                                 │
│  Cover Image ──► LocalFeatureExtractor ──► [LCESTB × 2] ──┐    │
│                                              ↓ Downsample  │    │
│                                          [LCESTB × 2] ──┐  │    │
│                                              ↓ Downsample│  │    │
│                                          [LCESTB × 2] ──┤  │    │
│                                              ↓           │  │    │
│                                        ┌─── FETB ───┐   │  │    │
│                                        │ (Bottleneck)│   │  │    │
│                                        └─────────────┘   │  │    │
│                                              ↑ Upsample  │  │    │
│  Watermark ──► WatermarkEncoder ──►  Concat (skip + wm)  │  │    │
│                                          [LCESTB × 2]    │  │    │
│                                              ↑ Upsample  │  │    │
│                                        Concat (skip + wm)│  │    │
│                                          [LCESTB × 2]    │  │    │
│                                              ↑           │  │    │
│                                        Concat (skip + wm)│  │    │
│                                          [LCESTB × 2]    │  │    │
│                                              ↓           │  │    │
│                                     Conv2d(dim → 3)      │  │    │
│                                              ↓           │  │    │
│                                     δ (residual image)   │  │    │
│                                              ↓           │  │    │
│                              Watermarked = Cover + δ     │  │    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│                     NOISE LAYER                       │
│  Watermarked Image ──► Simulated Attack ──► Attacked  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│                      DECODER                          │
│  Attacked Image ──► LocalFeatureExtractor             │
│       ↓                                               │
│  [LCESTB × 2] ──► Downsample ──► [LCESTB × 2]        │
│       ↓ Downsample                                    │
│  [LCESTB × 2] ──► FETB (Bottleneck)                   │
│       ↓                                               │
│  InfoConv ──► AdaptiveAvgPool ──► FC ──► 64-bit WM    │
└──────────────────────────────────────────────────────┘
```

### Component Details

| Module | File | Description |
|---|---|---|
| `Encoder` | `models/blocks/encoder.py` | U-Net encoder with LCESTB down/up blocks, FETB bottleneck, and watermark injection at every decoder level via skip connections. |
| `Decoder` | `models/blocks/decoder.py` | Downsampling-only architecture ending in adaptive pooling + FC to extract the 64-bit watermark. |
| `NoiseLayer` | `models/blocks/noise_layer.py` | Differentiable attack simulator with 8 attack types. |
| `LCESTB` | `models/components/LCESTB.py` | Swin Transformer + Locally-Channel Enhanced Block (depthwise conv + channel SE attention). |
| `FETB` | `models/components/FETB.py` | Standard Transformer blocks + DCT-based Frequency-Enhanced Block. |
| `SwinTransformer` | `models/components/layers/swin_transformer.py` | Window-based self-attention with shifted windows (W-MSA / SW-MSA). |
| `WindowAttention` | `models/components/layers/window_attention.py` | Multi-head attention with learnable relative position bias within windows. |
| `Transformer` | `models/components/layers/transformer.py` | Standard multi-head self-attention used in FETB. |
| `WatermarkEncoder` | `models/components/watermark_encoder.py` | Projects a 64-bit binary vector to spatial feature maps via linear + reshape + bilinear interpolation + Conv2d. |
| `LocalFeatureExtractor` | `models/components/feature_extractor.py` | Simple 3×3 Conv2d to lift RGB input to `dim`-dimensional feature space. |
| `WatermarkLoss` | `utils/losses.py` | Combined loss: MSE (image quality) + BCE (watermark extraction) + pixel-range constraint. |

---

## 📁 Project Structure

```
RoWSFormer-reprod/
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
| TorchVision | 0.26.0 |
| torch-dct | 0.1.6 |
| NumPy | 2.0.2 |
| Kornia | 0.8.3 |
| tqdm | 4.67.3 |
| matplotlib | (for visualization) |

---

## ⚙ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/RoWSFormer-reprod.git
cd RoWSFormer-reprod

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
pip install matplotlib           # For visualization in main.py
```

---

## 📂 Dataset Preparation

This reproduction uses the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) high-resolution image dataset.

1. **Download** the DIV2K training and validation HR images:
   - [DIV2K_train_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (~800 images)
   - [DIV2K_valid_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (~100 images)

2. **Organize** the dataset directory:
   ```
   dataset/
   ├── DIV2K_train_HR/
   │   ├── 0001.png
   │   ├── 0002.png
   │   └── ...
   └── DIV2K_valid_HR/
       ├── 0801.png
       ├── 0802.png
       └── ...
   ```

3. **Update paths** in `main.py` to point to your local dataset:
   ```python
   train_dataset = CustomDataset(img_dir='path/to/DIV2K_train_HR', img_size=128, bit_length=64)
   valid_dataset = CustomDataset(img_dir='path/to/DIV2K_valid_HR', img_size=128, bit_length=64)
   ```

> **Note:** Images are resized to **128 × 128** during training. The watermark is a random **64-bit** binary vector generated per sample.

---

## 🚀 Training

### 3-Phase Progressive Training Strategy

The training follows a progressive curriculum as described in the paper:

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
| FETB Blocks (Bottleneck) | 2 |

---

## 📊 Evaluation & Visualization

After training, `main.py` automatically runs a visualization that shows for each attack model:

1. **Original image** vs. **Watermarked image** (with PSNR) vs. **Attacked image** (with accuracy)
2. **Bit-level comparison** — original watermark, extracted watermark, and match/mismatch per bit

```
┌─────────────┬───────────────────┬────────────────────┐
│  Original   │   Watermarked     │   Attacked         │
│             │   PSNR=XX.XdB     │   Acc=XX.X%        │
├─────────────┴───────────────────┴────────────────────┤
│  Original WM:    ████░░░████░░████...                │
│  Extracted WM:   ████░░░████░░████...                │
│  Match (62/64):  ████████████████...                 │
└──────────────────────────────────────────────────────┘
```

---

## 🛡 Supported Attacks

### Non-Geometric Attacks

| Attack | Training | Testing |
|---|---|---|
| Gaussian Noise | σ ∈ [0.001, 0.04] | σ ∈ [0.01, 0.05] |
| Salt & Pepper Noise | ratio ∈ [0.001, 0.04] | ratio ∈ [0.01, 0.05] |
| Gaussian Blur | kernel=5×5, σ=2 (fixed) | σ ∈ [0.0001, 2] |
| Median Blur | kernel=7×7 (fixed) | kernel ∈ {3×3, 5×5, 7×7} |

### Geometric Attacks

| Attack | Training | Testing |
|---|---|---|
| Cropout | ratio = 0.4 | ratio ∈ [0.1, 0.5] |
| Dropout | ratio = 0.4 | ratio ∈ [0.2, 0.6] |
| Rotation | angle ∈ [-30°, 30°] | angle ∈ [-30°, 30°] |
| Scaling | ratio ∈ [0.7, 1.5] | ratio ∈ [0.6, 2.0] |

---

## 📈 Results (Paper)

As reported in the original paper, RoWSFormer achieves the following improvements over prior SOTA methods:

| Attack Category | PSNR Improvement | Extraction Accuracy |
|---|---|---|
| Non-geometric attacks | **+3 dB** (same accuracy) | Comparable to prior SOTA |
| Geometric attacks (rotation, scaling, affine) | **+6 dB** | **>97%** |

> ⚠ **Note:** These are the results from the original paper. This reproduction may not exactly replicate these numbers due to differences in implementation details, hardware, hyperparameter tuning, and randomness.

---

## 📝 Implementation Notes & Differences

This reproduction includes some modifications from the original paper:

1. **Cropout & Dropout Fix:** The original noise layer implementations replaced cropped/dropped regions with pixels from the original (cover) image, which leaks the cover signal to the decoder. This reproduction replaces removed regions with the **channel-wise mean** of the watermarked image to avoid information leakage.

2. **Watermark Encoder:** The watermark encoder projects the 64-bit vector through a linear layer to a spatial grid, then applies bilinear interpolation and convolution. The intermediate grid size (`l1=512`) may differ from the paper.

3. **Decoder Design:** The decoder uses adaptive average pooling to `4×4` followed by fully-connected layers, rather than global average pooling. BatchNorm is added in downsampling layers for gradient stability.

4. **Weight Initialization:** The decoder uses Kaiming initialization for Conv2d layers and truncated normal initialization for Linear layers, which helps with gradient flow.

5. **No Affine Attack:** The current noise layer does not include an affine transformation attack. Only rotation and scaling are implemented as geometric attacks.

6. **Image Clamping:** The encoder only clamps output to `[0, 1]` during inference (not training), allowing the constraint loss to softly regularize the output range.

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

---

## 🙏 Acknowledgements

- Original paper by **Weitong Chen** & **Yuheng Li** — [arXiv:2409.14829](https://arxiv.org/abs/2409.14829)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) by Microsoft Research
- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) for high-resolution image training
- [Kornia](https://github.com/kornia/kornia) for differentiable geometric transforms
- [torch-dct](https://github.com/zh217/torch-dct) for differentiable DCT operations

---

<p align="center">
  <i>Built as a learning exercise to understand Transformer-based robust watermarking.</i>
</p>
