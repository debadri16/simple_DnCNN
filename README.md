# Lightweight CNN for Gaussian Image Denoising

[![Python](https://img.shields.io/badge/python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.1-red)]()

## Overview

This project implements a **lightweight Convolutional Neural Network (CNN)** for **removing Gaussian noise from images**. The model is trained on **BSD500 patches** and can handle a **range of noise levels**.

The goal is to provide a **small, efficient denoiser** suitable for:

- mobile or edge devices
- real-time applications
- low-memory environments

> Inspired by AI-based denoising pipelines used in professional tools such as Adobe Photoshop.

---

## Model Architecture

The network is based on a **7–8 layer CNN** with **BatchNorm and ReLU** activations:
```
Conv2d(3 → 64) → ReLU
[Conv2d(64 → 64) → BatchNorm → ReLU] × 7
Conv2d(64 → 3)
```

- **Input:** RGB image (3 channels)  
- **Output:** Residual gaussian noise  
- **Patch size for training:** 50×50  
- **Weight initialization:** Kaiming normal  

The network predicts the **residual noise**.

---

## Training Details

- **Dataset:** BSD500 (10,000 patches extracted)  
- **Noise:** Gaussian noise with σ randomly sampled in [0.01, 0.2]  
- **Optimizer:** Adam  
- **Learning rate:** 1e-3  
- **Epochs:** 300  
- **Loss:** Mean Squared Error (MSE)

> **Best PSNR achieved:** 31.743  

---

## Requirements

```bash
python >=3.10
torch >=2.1
torchvision
numpy
matplotlib
```

---

## Usage
- Refer to the **.ipynb** file for details

---

## Properties
- The model removes Gaussian noise effectively while preserving textures.
- Lightweight (~1 MB), suitable for low-memory applications.
