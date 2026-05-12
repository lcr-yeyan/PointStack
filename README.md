<div align="center">

# PointStack

### Attention-Enhanced PointNet for Stacked Object Perception and Hierarchical Reasoning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Overview

**PointStack** is a 3D deep vision framework for robotic grasping in stacked object scenarios. It extends PointNet with channel attention, position attention, and multi-scale feature fusion to achieve high-precision semantic segmentation of point clouds. On top of segmentation, it performs instance clustering, hierarchical relationship reasoning, and grasp order planning — forming a complete perception-to-action pipeline.

> This project is part of an undergraduate thesis: *"Robotic Arm Grasping of Stacked Objects Based on 3D Depth Vision"* at China Jiliang University.

### Key Features

- **Attention-Enhanced PointNet++** — SE-Net channel attention in every Set Abstraction and Feature Propagation layer, plus a dedicated Position Attention module
- **Multi-Scale Feature Fusion** — Global (1024D) + mid-level (256D) + local (128D) features fused for richer point-wise representation
- **6-Channel Input Normalization** — XYZ coordinates + 3 complementary Z-normalization features tailored for tabletop scenes
- **Instance Segmentation** — Z-layered clustering with nearby-merge, height-split, and fragment-reconnect refinement
- **Hierarchical Reasoning** — Support edge detection via Z-gap + XY-overlap analysis, transitive closure for indirect support, topological sorting for grasp order
- **PyBullet Simulation** — Realistic depth rendering with ToF noise simulation (Gaussian jitter, center dropout, edge outliers, speckle)
- **End-to-End Pipeline** — Simulation → Semantic Segmentation → Instance Clustering → Hierarchy Inference → Grasp Planning

---

## Pipeline

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  PyBullet    │───▶│  PointStack     │───▶│  Instance        │
│  Simulation  │    │  Semantic Seg   │    │  Clustering      │
└──────────────┘    └─────────────────┘    └────────┬─────────┘
                                                    │
┌──────────────┐    ┌─────────────────┐             │
│  Grasp Order │◀───│  Hierarchy      │◀────────────┘
│  Planning    │    │  Reasoning      │
└──────────────┘    └─────────────────┘
```

---

## Model Architecture

PointStack builds upon the PointNet encoder-decoder framework with the following enhancements:

| Component | Standard PointNet | PointStack (Ours) |
|-----------|--------------------|---------------------|
| Input Channels | 3 (XYZ) or 6 (XYZ+RGB) | **6** (XYZ + 3 Z-normalizations) |
| SA Channel Attention | None | **SE-Net** per SA layer |
| FP Channel Attention | None | **SE-Net** per FP layer |
| Position Attention | None | **Position Attention** (QKV) |
| Multi-Scale Fusion | None | **Global + Mid + Local → 128D** |
| Global SE | None | **SE-Global** (1024→64→1024) |
| Mid-Level SE | None | **SE-Mid** (256→32→256) |
| Dropout | None | **Double Dropout** (0.3 + 0.2) |
| Parameters | ~1.5M | **2.18M** |

```
Input: (B, 2048, 6)
  │
  ├── SA1 (512 pts, r=0.2) ── [64,64,128] + ChannelAttn ──▶ feat1 (128D)
  ├── SA2 (128 pts, r=0.4) ── [128,128,256] + ChannelAttn ──▶ feat2 (256D)
  ├── SA3 (global) ── [256,512,1024] + ChannelAttn ──▶ feat3 (1024D)
  │                                                    │
  │                              SE-Global ◀────────────┤
  │                                                    │
  ├── FP3 (feat3 + feat2) ── [256,256] + ChannelAttn ──▶ fp3_out (256D)
  │                                                    │
  │                              SE-Mid ◀───────────────┤
  │                                                    │
  ├── FP2 (fp3_out + feat1) ── [256,256,128] + ChannelAttn ──▶ fp2_out (128D)
  ├── FP1 (fp2_out + input) ── [128,128,128] + ChannelAttn ──▶ fp1_out (128D)
  │
  ├── Position Attention ──▶ attn_out (128D)
  │
  ├── Multi-Scale Fusion ──▶ [fp1_out + feat3_broadcast + fp3_out_broadcast] → 128D
  │
  └── Classifier ── Conv1d(128→256) → Dropout(0.3) → Conv1d(256→128) → Dropout(0.2) → Conv1d(128→2)
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.4 (recommended)
- PyTorch 2.5+

### Setup

```bash
git clone https://github.com/lcr-yeyan/PointStack.git
cd PointStack

pip install -r requirements.txt
```

### Requirements

```
torch>=1.10.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
open3d>=0.14.0
matplotlib>=3.5.0
pyyaml>=6.0
omegaconf>=2.1.0
tqdm>=4.62.0
loguru>=0.5.0
pybullet>=3.2.0
```

---

## Quick Start

### 1. Generate Training Data

Generate 800 training + 200 validation scenes with random cuboid stacking:

```bash
python generate_training_data.py
```

### 2. Train the Model

```bash
python train_sem_seg.py
```

Training artifacts are saved to `training_data/train_logs/`:
- `best_model.pth` — best checkpoint
- `loss_curves.png` — loss and mIoU curves
- `metrics.json` — per-epoch detailed metrics
- `training_report.txt` — training summary
- `sample_predictions/` — validation set predictions

### 3. Test on Preview Scenes

```bash
python test_model.py
```

Evaluates the model on 8 carefully designed stacking scenarios. Results saved to `test_results/`.

### 4. Run Full Inference Pipeline

Semantic segmentation → instance clustering → hierarchy reasoning → visualization:

```bash
python simulate_inference.py
```

Results saved to `data_preview/_hierarchy_results/`.

### 5. Real-Time Simulation Test

```bash
python realtime_sim_test.py
```

Interactive mode: press `n` for new random scene, `s` to save, `q` to quit.

For automated batch testing:

```bash
python auto_realtime_test.py
```

---

## Results

### Semantic Segmentation (8 Test Scenes)

| Scene | Description | mIoU | Object IoU | Latency |
|-------|-------------|------|------------|---------|
| S1 | Side-by-side placement | 0.9957 | 0.9917 | 1489ms |
| S2 | 25% overlap stacking | 0.9903 | 0.9814 | 356ms |
| S3 | 50% overlap stacking | **1.0000** | **1.0000** | 374ms |
| S4 | 75% overlap stacking | **1.0000** | **1.0000** | 349ms |
| S5 | Centered full stack | **1.0000** | **1.0000** | 499ms |
| S6 | Offset full stack | **1.0000** | **1.0000** | 530ms |
| S7 | Cross stack (90° rotated) | **1.0000** | **1.0000** | 585ms |
| S8 | Aligned stack (different sizes) | **1.0000** | **1.0000** | 566ms |
| **Avg** | | **0.9982** | | **594ms** |

### Training Summary

| Metric | Value |
|--------|-------|
| Best Validation mIoU | **0.9998** |
| Best Epoch | 13 |
| Total Epochs (early stop) | 33 |
| Training Scenes | 800 |
| Validation Scenes | 200 |
| Model Parameters | 2,179,459 |

### Hierarchical Reasoning (8 Test Scenes)

| Metric | Value |
|--------|-------|
| Accuracy | **87.5%** (7/8) |
| Failure case | Cross-stack (S7) — insufficient Z-gap |

### Real-Time Simulation (30 Random Scenes)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 73.33% |
| Stacking Detection Accuracy | **95.24%** |
| Average Latency | 986.8ms |

---

## Project Structure

```
PointStack/
├── models/                          # Neural network definitions
│   ├── pointnet_seg.py              # PointStack (attention-enhanced PointNet++)
│   ├── sem_seg_net.py               # Baseline semantic segmentation networks
│   ├── instance_seg_net.py          # Instance segmentation networks
│   └── stack_layer_net.py           # Stacking layer prediction network
├── modules/                         # Core algorithm modules
│   ├── postprocess.py               # Instance clustering (Z-layered + refinement)
│   ├── hierarchy.py                 # Hierarchical relationship reasoning
│   └── preprocessing.py             # Point cloud preprocessing
├── configs/                         # Configuration files
│   └── default_config.yaml          # Default pipeline configuration
├── data_preview/                    # Preview/test scene data
│   └── _hierarchy_results/          # Hierarchy reasoning outputs
├── paper_figures/                   # Paper figures and plots
├── realtime_results/                # Real-time simulation results
├── sim_test_results/                # Simulation test results
├── train_sem_seg.py                 # Training script
├── test_model.py                    # Model evaluation on 8 test scenes
├── simulate_inference.py            # Full inference pipeline
├── generate_training_data.py        # Training data generation
├── generate_stacking_data.py        # Test scene generation
├── realtime_sim_test.py             # Interactive real-time simulation
├── auto_realtime_test.py            # Automated batch simulation test
├── generate_figures.py              # Paper figure generation
├── visualize_labeled_pcd.py         # Labeled point cloud visualization
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Configuration

All pipeline parameters are centralized in `configs/default_config.yaml`:

- **Camera**: resolution, FPS, filters
- **Preprocessing**: ROI, outlier removal, voxel downsampling, normalization
- **Model**: architecture selection, input channels, number of classes
- **Training**: batch size, epochs, optimizer, scheduler, data augmentation
- **Postprocessing**: DBSCAN/clustering parameters, instance filters
- **Hierarchy**: contact thresholds, Z-gap tolerance, indirect support depth

---

## Citation

If you use PointStack in your research, please cite:

```bibtex
@software{PointStack,
  author    = {Liu, Changrui},
  title     = {PointStack: Attention-Enhanced PointNet++ for Stacked Object Perception and Hierarchical Reasoning},
  year      = {2026},
  url       = {https://github.com/lcr-yeyan/PointStack}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with PyTorch · Open3D · PyBullet</sub>
</div>