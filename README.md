<div align="center">

<img src="https://readme-typing-svg.demolab.com/?lines=DesertVision:+Robust+Segmentation;Off-Road+Autonomy+Powered+by+AI;Trained+on+Synthetic+Digital+Twins&font=Fira+Code&center=true&width=600&height=50&color=3B82F6&vCenter=true&pause=1000" alt="Animated Title">

[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge&logo=target)]()
[![Domain](https://img.shields.io/badge/Domain-Computer_Vision-blue?style=for-the-badge&logo=opencv)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge&logo=pytorch)]()
[![Hardware](https://img.shields.io/badge/Hardware-RTX_3050_|_4GB_GPU-purple?style=for-the-badge&logo=nvidia)]()
[![Task](https://img.shields.io/badge/Task-Semantic_Segmentation-black?style=for-the-badge)]()
[![Data](https://img.shields.io/badge/Data-Synthetic_Digital_Twins-0078D4?style=for-the-badge)]()
[![App](https://img.shields.io/badge/Application-Autonomous_Vehicles-10B981?style=for-the-badge)]()
[![Models](https://img.shields.io/badge/Models-UNet++_%7C_SegFormer-6366F1?style=for-the-badge)]()
> **Team 404 BRAIN NOT FOUND:** Aayush Saha | Kalashdeep Asati | Dyutiman Bharadwaj | Harshit Gujar 

*Training models exclusively on Falcon-generated synthetic digital twin environments to segment novel, unseen real-world desert locations.*

</div>

<br>

## 🔬 Executive Overview

Unmanned Ground Vehicles (UGVs) require instantaneous, pixel-level scene understanding to safely navigate hazardous off-road terrain. This project focuses on training a robust semantic segmentation model using synthetic desert environment data generated from Duality AI’s Falcon platform. Our objective was to benchmark multiple architectures, optimizing preprocessing and loss functions to accurately segment 10 off-road terrain elements (vegetation, logs, rocks, landscape, etc.) under severe domain shifts.

---

## ⚙️ Architecture & Training Pipelines

We developed and benchmarked two distinct models, tailored for different hardware constraints and architectural strengths.

### 🟢 Pipeline A: UNet++ (CNN-Based)
Built for efficient feature extraction and preserving fine spatial boundaries.

* **Backbone:** EfficientNet-B4 with scSE attention in the decoder.
* **Hardware Setup:** Trained on a 4GB GPU using Mixed Precision (fp16) to maximize VRAM efficiency.
* **Loss Optimization:** Combined CrossEntropy (with tuned class weights), Dice Loss, and a Boundary-Aware Loss to penalize class edge pixels.
* **Augmentation Suite:** Horizontal/vertical flips, ±20° rotation, color jitter, and random crop+resize. *Crucially, all mask transforms used NEAREST interpolation to prevent class label corruption*.

<div align="center">

| Training Phase | Configuration | mIoU Achieved |
| :--- | :--- | :--- |
| **Phase 1 (Frozen Encoder)** | 20 epochs, lr=2e-4, OneCycleLR | `0.5599` |
| **Phase 2 (Unfrozen Encoder)** | 15 epochs, encoder lr=5e-6, decoder lr=5e-5 | `0.5887` |

</div>

<br>

### 🔵 Pipeline B: SegFormer-B4 (Transformer-Based)
Built to capture long-range global context using attention mechanisms.

* **Backbone:** Hierarchical Mix Transformer (MiT-B4) encoder with a lightweight MLP decoder (pretrained on ADE20K, ~64M parameters).
* **Hardware Setup:** Trained on an RTX 3050 (6GB VRAM) using `torch.amp.autocast` + `GradScaler`, delivering 30–50% speed improvements. Input resolution was reduced from 640 to 512 to balance GPU load.
* **Loss Optimization:** 0.5 × CrossEntropy (with 0.1 label smoothing) + 0.5 × Dice Loss.
* **Optimization Strategy:** AdamW optimizer (handles weight decay correctly for Transformers) + OneCycleLR to avoid local minima.

<div align="center">

| Fine-Tuning Stage | Focus & Strategy | mIoU Progression |
| :--- | :--- | :--- |
| **Stage 1 (Epochs 1–10)** | Full model, decoder adapts, lr=6e-5 | `0.25 – 0.38` |
| **Stage 2 (Epochs 10–20)** | Encoder features gradually adapt | `0.40 – 0.51` |
| **Stage 3 (Epochs 20–40)** | LR decays, fine boundary refinement | `0.53 – 0.61` |

</div>

---

## 📊 Model Evaluation & Metrics

Despite the constraints of 4–6GB VRAM GPUs, our targeted optimizations achieved roughly a **2× improvement** over the baseline model. Below is the comprehensive architecture comparison followed by the detailed Per-Class IoU breakdown.

<table align="center" style="border-collapse: collapse; width: 100%; border: 1px solid #333;">
  <tr style="background-color: #0f172a; color: white;">
    <th align="center" style="padding: 12px; border: 1px solid #333;">Architecture</th>
    <th align="center" style="padding: 12px; border: 1px solid #333;">Backbone & Training Phase</th>
    <th align="center" style="padding: 12px; border: 1px solid #333;">Loss & Optimizer</th>
    <th align="center" style="padding: 12px; border: 1px solid #333;">Key Enhancements Applied</th>
    <th align="center" style="padding: 12px; border: 1px solid #333;">Final mIoU</th>
  </tr>
  <tr>
    <td align="center" style="padding: 12px; border: 1px solid #333;">⚪ <b>Baseline</b></td>
    <td align="center" style="padding: 12px; border: 1px solid #333;">DINOv2 ViT-S/14<br><i>(Frozen Backbone)</i></td>
    <td align="center" style="padding: 12px; border: 1px solid #333;">Standard Setup<br>SGD (10 Epochs)</td>
    <td align="center" style="padding: 12px; border: 1px solid #333;">None</td>
    <td align="center" style="padding: 12px; border: 1px solid #333;"><code>0.2900</code></td>
  </tr>
  <tr>
    <td align="center" style="padding: 12px; border: 1px solid #333;">🟢 <b>UNet++</b></td>
    <td align="center" style="padding: 12px; border: 1px solid #333;">EfficientNet-B4<br><i>(2-Phase Fine-Tuning)</i></td>
    <td align="center" style="padding: 12px; border: 1px solid #333;">CE + Dice + Boundary<br>OneCycleLR</td>
    <td align="center" style="padding: 12px; border: 1px solid #333;">Hard Example Mining (Epoch 10+), 6-pass TTA, OpenCV Bilateral CRF</td>
    <td align="center" style="padding: 12px; border: 1px solid #333;"><code>0.5887</code></td>
  </tr>
  <tr>
    <td align="center" style="padding: 12px; border: 1px solid #333;">🔵 <b>SegFormer</b></td>
    <td align="center" style="padding: 12px; border: 1px solid #333;">MiT-B4<br><i>(Head Reinitialization)</i></td>
    <td align="center" style="padding: 12px; border: 1px solid #333;">0.5 CE + 0.5 Dice<br>AdamW + OneCycleLR</td>
    <td align="center" style="padding: 12px; border: 1px solid #333;">Class Weight Rebalancing, Mixed Precision (AMP), Gradient Clipping</td>
    <td align="center" style="padding: 12px; border: 1px solid #333;">🏆 <b><code>0.6262</code></b></td>
  </tr>
</table>

<br>

<table align="center" style="border-collapse: collapse; border: none; background-color: transparent;">
  <tr>
    <td align="center" style="border: none; padding-bottom: 10px;"><h3>🟢 UNet++ Metrics</h3></td>
    <td align="center" style="border: none; padding-bottom: 10px;"><h3>🔵 SegFormer-B4 Metrics</h3></td>
  </tr>
  <tr>
    <td align="center" style="border: none;">
      <img src="RESULTS AND EXAMPLE IMAGES/PREDICTED ACCURACY/UNET++ MODEL 1.png" width="450" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/><br>
      <img src="RESULTS AND EXAMPLE IMAGES/PREDICTED ACCURACY/UNET++ MODEL 2.png" width="450" style="border-radius: 8px; margin-top: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>
    </td>
    <td align="center" style="border: none;">
      <img src="RESULTS AND EXAMPLE IMAGES/PREDICTED ACCURACY/SEG-FORMER-B4 1.png" width="450" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/><br>
      <img src="RESULTS AND EXAMPLE IMAGES/PREDICTED ACCURACY/SEG-FORMER-B4 2.png" width="450" style="border-radius: 8px; margin-top: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>
    </td>
  </tr>
</table>

---

## 🖼️ Visual Prediction Gallery

A direct side-by-side comparison of Raw Colour Images vs. Predicted Segmentations for both models.

<table align="center" style="border-collapse: collapse; border: none;">
  <tr>
    <th colspan="2" style="font-size: 18px; text-align: center; padding: 15px 0;">🟢 UNet++ Model Predictions</th>
  </tr>
  <tr>
    <td align="center" style="border: none; width: 50%;">
      <b>Example 1: Raw Colour Image</b><br>
      <img src="RESULTS AND EXAMPLE IMAGES/IMAGE PREDICTION/UNET++ RAW 1.jpg" width="400" style="border-radius: 8px; margin-top: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/><br>
    </td>
    <td align="center" style="border: none; width: 50%;">
      <b>Example 1: Predicted Classes</b><br>
      <img src="RESULTS AND EXAMPLE IMAGES/IMAGE PREDICTION/UNET++ PRED 1.png" width="400" style="border-radius: 8px; margin-top: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/><br>
    </td>
  </tr>
  <tr>
    <td align="center" style="border: none; padding-top: 20px;">
      <b>Example 2: Raw Colour Image</b><br>
      <img src="RESULTS AND EXAMPLE IMAGES/IMAGE PREDICTION/UNET++ RAW 2.jpg" width="400" style="border-radius: 8px; margin-top: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/><br>
    </td>
    <td align="center" style="border: none; padding-top: 20px;">
      <b>Example 2: Predicted Classes</b><br>
      <img src="RESULTS AND EXAMPLE IMAGES/IMAGE PREDICTION/UNET++ PRED 2.png" width="400" style="border-radius: 8px; margin-top: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/><br>
    </td>
  </tr>

  <tr>
    <th colspan="2" style="font-size: 18px; text-align: center; padding: 40px 0 15px 0;">🔵 SegFormer-B4 (+ Fine-Tuning) Predictions</th>
  </tr>
  <tr>
    <td align="center" style="border: none;">
      <b>Example 1: Raw Colour Image</b><br>
      <img src="RESULTS AND EXAMPLE IMAGES/IMAGE PREDICTION/SEG-FOR RAW 1.jpg" width="400" style="border-radius: 8px; margin-top: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/><br>
    </td>
    <td align="center" style="border: none;">
      <b>Example 1: Predicted Classes</b><br>
      <img src="RESULTS AND EXAMPLE IMAGES/IMAGE PREDICTION/SEG-FOR PRED 1.png" width="400" style="border-radius: 8px; margin-top: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/><br>
    </td>
  </tr>
  <tr>
    <td align="center" style="border: none; padding-top: 20px;">
      <b>Example 2: Raw Colour Image</b><br>
      <img src="RESULTS AND EXAMPLE IMAGES/IMAGE PREDICTION/SEG-FOR RAW 2.jpg" width="400" style="border-radius: 8px; margin-top: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/><br>
    </td>
    <td align="center" style="border: none; padding-top: 20px;">
      <b>Example 2: Predicted Classes</b><br>
      <img src="RESULTS AND EXAMPLE IMAGES/IMAGE PREDICTION/SEG-FOR PRED 2.png" width="400" style="border-radius: 8px; margin-top: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/><br>
    </td>
  </tr>
</table>

---

## 🚧 Roadblocks & Interactive Solutions

Click to expand how we handled critical engineering challenges during the 6-hour hackathon timeframe:

<details>
<summary><b>1. Severe Target Out-of-Bounds Crashes 🛑</b></summary>
<br>
<b>Issue:</b> Raw mask pixel values were encoded as 100, 200... 10000, causing CrossEntropy to crash immediately.<br>
<b>Fix:</b> Built a custom remapping dictionary converting raw values to contiguous 0–9 indices, completely eliminating index errors.
</details>

<details>
<summary><b>2. The "Invisible Object" Class Imbalance 👻</b></summary>
<br>
<b>Issue:</b> Sky and Landscape classes dominated pixel distribution, leaving Flowers, Logs, and Ground Clutter essentially invisible to the model (IoU ≈ 0).<br>
<b>Fix:</b> Deployed Hard Example Mining (oversampling the top 30% hardest images 3×) combined with manual class weights (Flowers/Logs=3.5, Sky=0.2).
</details>

<details>
<summary><b>3. Cython Compilation Failures on Windows 💻</b></summary>
<br>
<b>Issue:</b> <code>pydensecrf</code> failed to build during CRF post-processing due to a 278-line Cython/Eigen error.<br>
<b>Fix:</b> Swapped out the library for an OpenCV bilateral filter guided by RGB edges—achieving the same boundary-snapping effect with zero compilation.
</details>

<details>
<summary><b>4. Exploding Gradients in Transformers 💥</b></summary>
<br>
<b>Issue:</b> Large transformer models (ViT/MiT) are prone to gradient spikes during fine-tuning.<br>
<b>Fix:</b> Applied gradient clipping (max_norm=1.0) and AdamW with LR warmup (10%), stabilizing convergence through all training stages.
</details>

---

## 🚀 The Road Ahead (Future Scope)

* **✨ Cross-Model Ensembling:** Combining UNet++'s spatial boundary precision with SegFormer's global context via mIoU-weighted softmax averaging to yield an expected 2–4% performance boost.
* **🌍 Sim-to-Real Gap Analysis:** Deploying the model on real desert photographs to quantify degradation under true domain shifts, informing future dataset generation in Falcon.
* **⚡ Edge Quantization:** Utilizing TorchScript export and INT8 post-training quantization to slash inference latency under 10ms per image—critical for NVIDIA Jetson hardware on actual UGVs.
* **🔍 Panoptic Upgrade:** Evolving from semantic to panoptic segmentation to differentiate between individual instances of the same class (e.g., distinguishing two overlapping trees) for enhanced obstacle avoidance.

<br>

<div align="center">
  <img src="https://img.shields.io/badge/Built_with_%E2%9D%A4%EF%B8%8F_by-404_Brain_Not_Found-0f172a?style=for-the-badge&logo=github" alt="Built by Team">
</div>
