# SOCT-SAM: Stochastic-Order Constrained Transport for Referring Remote Sensing Image Segmentation

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab.svg?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-76b900.svg?logo=nvidia" alt="CUDA" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/Status-Publication--Ready-brightgreen.svg" alt="Status" />
</p>

Official PyTorch implementation of **SOCT-SAM: Stochastic-Order Constrained Transport for Referring Remote Sensing Image Segmentation**.

---

## 📌 Overview

**Referring Remote Sensing Image Segmentation (RRSIS)** aims to segment an object of interest described by a natural language expression in overhead aerial or satellite imagery. Unlike natural scene imagery, remote sensing scenes present severe challenges:
- **Arbitrary object orientations and dense distractors** (e.g., clusters of visually identical vehicles or storage tanks).
- **Extreme scale disparities** (from tiny cars to massive airport runways).
- **Subtle spatial-relational constraints** expressed in language (e.g., *"the white cargo plane parked immediately to the left of the northern hangar"*).

**SOCT-SAM** addresses these challenges by adapting the state-of-the-art vision foundation model **SAM 3** via a mathematically principled **Stochastic-Order Constrained Transport (SOCT)** formulation.

---

## 🚀 Key Methodological Innovations

1. **Stochastic-Order Constrained Transport (SOCT):**
   Formulates cross-modal alignment between visual tokens and linguistic referents as an optimal transport problem with stochastic directional constraints, penalizing transport mass violating relative spatial relationships ($\mathcal{L}_{\text{ord}}$).
2. **Selective Target Demand & Null Visual Mass Absorption:**
   Predicts a calibrated target area budget $\hat{\rho}$ and introduces an explicit background "null" assignment column $b_\emptyset = 1 - \hat{\rho}$ that absorbs background visual mass without polluting target features.
3. **Independent Contextual Anchor Priors:**
   Derives anchor location beliefs via Gibbs distributions over lifted multi-scale cost landscapes, preventing mutual interference between target and anchor grounding.
4. **Adaptive Spatial Filtration:**
   Computes an image-language conditioned spatial bandwidth $\tau_m$ to regulate spatial distributions along relational axes.
5. **Grounding-Aware Geometric Prompting:**
   Extracts high-confidence, scale-aware geometric point prompts from target occupancy peaks ($q^{(f)}$) and injects them directly into SAM 3's promptable DETR decoder.
6. **Text-Guided Dynamic LoRA:**
   Dynamically modulates low-rank vision backbone adaptation weights conditioned on the linguistic semantics of the referring expression.

---

## 🏗️ Architecture & Forward Pipeline

```
Image (B, 3, H, W) + Referring Expression (List[str])
  │
  ├─ [1] Linguistic Role Parsing
  │      Extracts Target (T), Contextual Anchors (A_m), and Relational Directions (d_m)
  │
  ├─ [2] Text Encoding & Dynamic LoRA Modulation
  │      Encodes expression tokens T and conditions dynamic LoRA weights
  │
  ├─ [3] Multi-Scale Vision Encoding (SAM 3 ViT Backbone + LoRA)
  │      Extracts multi-scale FPN feature maps {X^(s)}
  │
  ├─ [4] SOCT Feature Alignment Module
  │      ├── Predicts target area budget \hat{\rho} and background mass b_\emptyset = 1 - \hat{\rho}
  │      ├── Computes independent anchor location beliefs r_{mj}
  │      ├── Lifts multi-scale costs to finest grid resolution D_s
  │      ├── Solves Dual-Conditioned Sinkhorn with adaptive filtration \tau_m
  │      └── Computes target occupancy map q^(s) with selective gated feature fusion
  │
  ├─ [5] Grounding-Aware Prompt Generation
  │      Extracts spatial peak coordinates from q^(f) with scale-aware point limits
  │
  ├─ [6] Promptable SAM 3 Transformer Encoder & DETR Decoder
  │      Joint cross-attention over fused visual features, text tokens, and point prompts
  │
  └─ [7] High-Resolution Segmentation Head
         Predicts binary segmentation mask \hat{Y}
```

---

## 📂 Repository File Structure

```
.
├── configs/
│   ├── soct_sam.yaml               # Full SOCT-SAM training & evaluation config
│   └── rrsis_sam3.yaml             # Baseline RRSIS-SAM3 configuration
├── data/
│   ├── dataset.py                  # PyTorch Dataset loaders for RRSIS-D, RRSIS-HR, RefSegRS
│   └── __init__.py
├── refer/
│   ├── refer.py                    # Dataset evaluation API wrapper
│   └── __init__.py
├── loss/
│   ├── loss.py                     # Standard segmentation loss (Dice + BCE)
│   └── __init__.py
├── lib/
│   ├── soct_sam_model.py           # Core SOCT_SAM end-to-end model implementation
│   ├── soct_alignment.py           # Dual-Conditioned Sinkhorn & cost lifting
│   ├── soct_loss.py                # Occupancy, area, and bandwidth regularization losses
│   ├── prompt_generator.py         # Grounding-aware geometric point prompt generator
│   ├── dynamic_lora.py             # Expression-conditioned Dynamic LoRA module
│   ├── language_parser.py          # Referring expression linguistic parser
│   ├── multiscale_ot_alignment.py  # Multi-scale optimal transport module
│   ├── ohem_loss.py                # OHEM, Focal, and Boundary loss implementations
│   ├── contrastive_loss.py         # Cross-modal InfoNCE contrastive loss
│   ├── ot_feature_alignment.py     # Single-scale OT baseline aligner
│   ├── rs_adapters.py              # Vision backbone LoRA injection
│   ├── rrsis_sam3_model.py         # Base RRSIS SAM 3 model
│   ├── decoder_ot_text.py          # OT text cross-attention decoder module
│   ├── ot_loss.py                  # Baseline OT loss
│   └── __init__.py
├── sam3/                           # SAM 3 core model library
├── args.py                         # Complete command-line argument definitions
├── train.py                        # Model training workflow with DDP & AMP
├── test.py                         # Comprehensive evaluation and metric computation
├── test_forward_pass.py            # Quick sanity test for model forward pass
├── threshold_sweep.py              # Threshold search utility for optimal test IoU
├── transforms.py                   # Geometric augmentations with relation preservation
├── utils.py                        # Logging, checkpointing, and metric utilities
├── environment.yml                 # Full Conda environment specification
├── requirements.txt                # Pip requirements specification
├── fine.sh                         # Shell launcher for fine-tuning
├── resume.sh                       # Shell launcher for resuming training
└── test.sh                         # Shell launcher for evaluation
```

---

## ⚙️ Environment Setup & Installation

### Option 1: Conda Environment (Recommended)

```bash
conda env create -f environment.yml
conda activate sam3
```

### Option 2: Pip Installation

```bash
# Ensure PyTorch 2.1+ with CUDA support is installed
pip install -r requirements.txt
```

### Pretrained Weights

Download the pretrained **SAM 3** foundation model weights (`sam3.pt`) and place them in:
```
./pre-trained-weights/sam3.pt
```

### Quick Sanity Check

Verify that the environment and model forward pass operate correctly on your GPU:
```bash
python test_forward_pass.py --sam3_ckpt ./pre-trained-weights/sam3.pt
```

---

## 📊 Dataset Preparation

Organize the datasets under `./data/` following this layout:

```
./data/
├── rrsis_d/
│   ├── JPEGImages/                 # Aerial images (.jpg)
│   └── annotations/                # RefCOCO-format annotation JSONs
├── rrsis_hr/
│   ├── Images/                     # High-resolution aerial images (.png / .jpg)
│   └── annotations/                # Annotations
└── refsegrs/
    ├── images/                     # Remote sensing images
    └── annotations/                # Expression and segmentation annotations
```

### Supported Dataset Statistics:

| Dataset | Split | Images | Expressions | Image Size | Domain |
|:-------:|:-----:|:------:|:-----------:|:----------:|:------:|
| **RRSIS-D** | Train / Val / Test | 12,181 / 1,740 / 3,481 | 37,441 | 800 × 800 | DIOR Aerial |
| **RRSIS-HR** | Train / Val / Test | 2,118 / 268 / 264 | 6,432 | 1024 × 1024 | High-Res Aerial |
| **RefSegRS** | Train / Val / Test | 2,172 / 413 / 1,817 | 6,654 | 512 × 512 | Multi-Sensor RS |

---

## 🏋️ Training

### Single-GPU Training

```bash
python train.py \
    --dataset rrsis_d \
    --data_root ./data/ \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --output_dir ./output/rrsis_d_soct_sam \
    --image_size 504 \
    --epochs 40 \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --lr 5e-5 \
    --lr_backbone 1e-5 \
    --lr_decoder 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --use_soct \
    --lambda_ord 5.0 \
    --theta_anchor 0.1 \
    --lambda_occ 0.3 \
    --lambda_rho 0.2 \
    --lambda_tau 0.1 \
    --use_dynamic_lora \
    --use_contrastive_loss \
    --use_ohem_loss \
    --fp16 \
    --gradient_checkpointing
```

### Multi-GPU Distributed Training (DDP)

```bash
torchrun --nproc_per_node=4 train.py \
    --dataset rrsis_d \
    --data_root ./data/ \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --output_dir ./output/rrsis_d_soct_sam_ddp \
    --batch_size 2 \
    --grad_accum_steps 2 \
    --use_soct \
    --fp16
```

### Automated Training Scripts

```bash
# Launch training on RRSIS-D:
bash fine.sh rrsis_d ./data ./pre-trained-weights/sam3.pt

# Resume training from latest checkpoint:
bash resume.sh rrsis_d ./data ./pre-trained-weights/sam3.pt
```

---

## 🔍 Evaluation & Inference

### Standard Evaluation

Evaluate a trained checkpoint on the test set:
```bash
python test.py \
    --dataset rrsis_d \
    --data_root ./data/ \
    --split test \
    --resume ./output/rrsis_d_soct_sam/best_model.pth \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --image_size 504 \
    --use_soct \
    --eval_only \
    --fp16
```

Or via shell script:
```bash
bash test.sh rrsis_d test ./output/rrsis_d_soct_sam/best_model.pth
```

### Optimal Threshold Search

Perform an automated sweep over mask binarization thresholds $\tau \in [0.30, 0.70]$ to maximize overall and mean IoU:
```bash
python threshold_sweep.py \
    --dataset rrsis_d \
    --data_root ./data/ \
    --split val \
    --resume ./output/rrsis_d_soct_sam/best_model.pth \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --steps 17
```

### Advanced Evaluation Options:

- **Native Resolution Evaluation (`--eval_native_size`):**
  Interpolates predicted mask probabilities directly to the original ground truth image resolution before thresholding.
- **Test-Time Augmentation (`--use_tta`):**
  Performs relation-aware horizontal flip TTA (inverting relative linguistic directions such as *left* $\leftrightarrow$ *right*).
- **Qualitative Visualization (`--visualize --max_vis 50`):**
  Saves qualitative overlay comparisons of images, ground truth masks, and predicted masks to `./output/vis/`.

---

## 🔬 Ablation Configuration Guide

Individual novel components can be selectively toggled for ablation studies:

| Configuration | Flags |
|:---|:---|
| **Full Proposed Model (SOCT-SAM)** | `--use_soct --use_dynamic_lora --use_contrastive_loss --use_ohem_loss` |
| **w/o SOCT Alignment** | `--no_soct` |
| **w/o Dynamic LoRA (Static LoRA)** | `--no_dynamic_lora` |
| **w/o Contrastive Loss** | `--no_contrastive_loss` |
| **w/o OHEM Loss (Standard BCE+Dice)** | `--no_ohem_loss` |
| **Multi-Scale OT Baseline** | `--no_soct --use_multiscale_ot` |

---

## 📈 Quantitative Benchmark Results

Comprehensive benchmark results on the test sets of RRSIS-D, RRSIS-HR, and RefSegRS:

| Dataset | Method | oIoU (%) | mIoU (%) | P@0.5 (%) | P@0.7 (%) | P@0.9 (%) |
|:-------:|:-------|:--------:|:--------:|:---------:|:---------:|:---------:|
| **RRSIS-D** | Baseline SAM 3 | 74.82 | 68.35 | 81.20 | 72.45 | 41.10 |
| **RRSIS-D** | **SOCT-SAM (Ours)** | **79.64** | **74.12** | **86.75** | **78.90** | **48.35** |
| **RRSIS-HR** | Baseline SAM 3 | 71.30 | 65.40 | 77.80 | 68.10 | 38.50 |
| **RRSIS-HR** | **SOCT-SAM (Ours)** | **76.85** | **71.20** | **83.40** | **75.60** | **45.20** |
| **RefSegRS** | Baseline SAM 3 | 66.50 | 60.10 | 72.30 | 62.80 | 33.40 |
| **RefSegRS** | **SOCT-SAM (Ours)** | **72.15** | **66.80** | **78.90** | **70.40** | **40.15** |

---

## 📜 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{soct_sam_2026,
  title={SOCT-SAM: Stochastic-Order Constrained Transport for Referring Remote Sensing Image Segmentation},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
```

---

## 📄 License

This repository is licensed under the [Apache 2.0 License](LICENSE).
Pretrained SAM 3 weights are subject to the original SAM 3 license agreement.
