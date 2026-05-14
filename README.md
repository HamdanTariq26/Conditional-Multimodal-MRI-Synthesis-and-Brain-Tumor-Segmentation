# Conditional Multimodal MRI Synthesis & Brain Tumor Segmentation

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![TPU](https://img.shields.io/badge/Hardware-TPU%20v3--8-008080.svg)](https://cloud.google.com/tpu)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end medical AI pipeline for the **BraTS 2020 Challenge**. This project implements a dual-model system: a high-precision **ResNet-encoded U-Net** for tumor segmentation and a **Conditional Diffusion Model (DDPM)** for synthesizing realistic, multi-modal MRI volumes from segmentation masks.

---

## 🌟 Key Features

### 1. Segmentation Pipeline (ResNet-U-Net)
*   **Architecture**: U-Net backbone with a Residual Encoder (ResNet) for superior feature extraction.
*   **Sub-region Detection**: Automated delineation of Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).
*   **Performance**: Achieved mean Dice coefficients of **0.80 (WT)**, **0.76 (TC)**, and **0.86 (ET)**.

### 2. Synthesis Pipeline (Conditional DDPM)
*   **Architecture**: UNet2DModel conditioned on categorical segmentation masks.
*   **Modalities**: Synthesizes T1, T2, T1ce, and FLAIR volumes simultaneously.
*   **Optimization**: Implements **v-prediction** objective and **DDIM sampling** (50 steps) for high-fidelity, accelerated inference.

---

## 🚀 Technical Innovations

This project features advanced engineering solutions for training generative models on specialized hardware:

*   💎 **CPU-EMA Weight Tracking**: Solved BFloat16 precision underflow on TPUs by offloading Exponential Moving Average shadow weights to CPU RAM in FP32.
*   🧠 **Brain-Masked Normalization**: Implemented foreground-specific Z-score normalization to prevent "washed-out" syntheses caused by dark background pixels.
*   🎨 **Composite Loss Function**: A hybrid objective combining **MSE**, **SSIM**, and **VGG Perceptual Loss** to optimize for pixel accuracy, structural integrity, and clinical texture.
*   ⚡ **XLA Optimization**: Custom distributed loading strategy to prevent communication deadlocks during TPU multi-core execution.

---

## 📊 Dataset: BraTS 2020
Trained and validated on the **Multimodal Brain Tumor Segmentation Challenge 2020** dataset.
*   **369 Patient Cases**: Co-registered T1, T2, T1ce, and FLAIR modalities.
*   **Expert Labels**: NCR (Necrotic), ED (Edema), and ET (Enhancing).

---

## 🛠️ Installation & Usage

### Prerequisites
*   Python 3.9+
*   Kaggle or Google Cloud TPU (Recommended for Synthesis)

### Setup
```bash
[git clone https://github.com/[Remaining]/brats-mri-diffusion.git](https://github.com/HamdanTariq26/Conditional-Multimodal-MRI-Synthesis-and-Brain-Tumor-Segmentation.git)
cd brats-mri-diffusion
pip install -r requirements.txt
```

---

## 🖼️ Results

### Segmentation Performance
| Region | Dice Score |
| :--- | :--- |
| **Whole Tumor** | 0.80 |
| **Tumor Core** | 0.76 |
| **Enhancing Tumor** | 0.86 |

*(Insert your `brats_final_sharp.png` here)*

---

## 👥 Authors
*   **Hamdan Tariq** - [GitHub](https://github.com/your-username)
*   **Mubashara Anees**

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**National University of Sciences and Technology (NUST)**
*Semester — Department of Computer Science*
