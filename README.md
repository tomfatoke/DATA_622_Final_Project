# Chest X-Ray Classification Using Transfer Learning

**Group 3** — Abdallah Elshafey, Josh Brauner, Tom Fatoke, Toheeb Ayuba

A deep learning pipeline for classifying chest X-rays into three categories — **Normal**, **Pneumonia (Bacterial)**, and **Pneumonia (Viral)** — using transfer learning with DenseNet161 and Grad-CAM explainability.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Explainability](#explainability)
- [Dataset](#dataset)
- [Team Contributions](#team-contributions)

---

## Project Overview

This project trains a DenseNet161 model (pretrained on ImageNet) to classify chest X-rays. The model was selected after systematic screening of 78+ pretrained architectures from `torchvision`, with DenseNet161 achieving the highest screening accuracy of **75.3%** on a fast one-epoch evaluation.

Key design decisions:
- Two-phase progressive unfreezing (warmup → full fine-tuning)
- Inverse frequency class weighting to handle dataset imbalance
- Cosine annealing learning rate schedule with early stopping
- Grad-CAM visualizations to confirm anatomically plausible predictions

---

## Results

Performance on the held-out test set of **624 images**:

| Metric              | Score  |
|---------------------|--------|
| Accuracy            | 0.7644 |
| Micro AUC (ROC)     | 0.9043 |
| Micro Precision     | 0.7644 |
| Micro Recall        | 0.7644 |
| Micro F1            | 0.7644 |

---

## Repository Structure

```
.
├── train_group_3.py               # Model definition, training loop, dataset class
├── test_group_3.py                # Evaluation, metrics, and Grad-CAM visualization
├── Chest_xray_Corona_Metadata.csv # Dataset metadata (image filenames + labels)
├── densenet161_best.pth.tar.gz    # Saved best model checkpoint
├── train/                         # Training images
└── test/                          # Test images
```

---

## Requirements

- Python 3.8+
- PyTorch (with CUDA recommended)
- torchvision
- scikit-learn
- pandas
- numpy
- matplotlib
- tqdm
- Pillow

Install dependencies:

```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib tqdm Pillow
```

---

## Setup

1. Clone or download the repository.
2. Place your training images under `./train/` and test images under `./test/`. Subdirectory structures are supported — the dataset loader uses recursive glob.
3. Ensure `Chest_xray_Corona_Metadata.csv` is in the project root. The CSV must contain columns: `X_ray_image_name`, `Label`, `Label_1_Virus_category`, and `Dataset_type`.

---

## Usage

### Training

```bash
python train_group_3.py
```

Key hyperparameters (configurable at the top of `train_group_3.py`):

| Parameter       | Value  | Description                                  |
|-----------------|--------|----------------------------------------------|
| `NUM_EPOCHS`    | 25     | Maximum training epochs                      |
| `BATCH_SIZE`    | 32     | Batch size (tuned for 8 GB GPU)              |
| `LEARNING_RATE` | 1e-4   | Warmup phase learning rate (classifier only) |
| `WARMUP_EPOCHS` | 5      | Epochs before backbone is unfrozen           |
| `FINETUNE_LR`   | 1e-5   | Learning rate after backbone unfreezing      |
| `PATIENCE`      | 5      | Early stopping patience                      |
| `SEED`          | 42     | Random seed for reproducibility              |

The best model (by training loss) is saved to `densenet161_best.pth` and compressed as `densenet161_best.pth.tar.gz`.

### Evaluation

```bash
python test_group_3.py
```

This will:
1. Load the saved model from `densenet161_best.pth.tar.gz`
2. Run inference on the full test set and print Accuracy, AUC, Precision, Recall, and F1
3. Display Grad-CAM visualizations for 5 randomly selected test images

---

## Model Architecture

- **Base model:** DenseNet161 pretrained on ImageNet (`torchvision.models.densenet161`)
- **Classifier head:** `Linear(6144, 3)` replacing the original `Linear(6144, 1000)`
- **Total parameters:** ~28.7M
- **Trainable during warmup:** 6,627 (classifier head only)
- **Input:** 224×224 RGB (grayscale X-rays duplicated across all 3 channels)

---

## Training Strategy

**Phase 1 — Warmup (Epochs 1–5)**

The backbone is frozen. Only the classifier head is trained at `lr=1e-4`. This prevents catastrophic forgetting while the randomly initialized classifier adapts.

**Phase 2 — Fine-tuning (Epochs 6–25)**

All 28.7M parameters are unfrozen and trained at `lr=1e-5` using CosineAnnealingLR (`eta_min=1e-8`). The reduced learning rate ensures the pretrained features are updated gradually rather than overwritten.

**Class Imbalance Handling**

Inverse frequency weighting is applied in `CrossEntropyLoss`:

| Class              | Image Count | Weight |
|--------------------|-------------|--------|
| Normal             | 1,342       | 1.31   |
| Pneumonia (Viral)  | 1,407       | 0.69   |
| Pneumonia (Bacterial) | 2,535    | 1.25   |

**Data Augmentation (training only)**

- Resize to 224×224
- Random horizontal flip (p=0.5) — valid due to bilateral lung symmetry
- Random rotation ±10°
- ColorJitter (brightness=0.3, contrast=0.3) — simulates varying X-ray equipment and exposure
- Normalize with ImageNet statistics: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

---

## Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize model decisions. Gradients from the predicted class score are backpropagated to the final convolutional feature maps (`model.features`), global-average-pooled to produce per-channel importance weights, and combined into a heatmap overlaid on the original X-ray.

Visual inspection confirmed the model consistently focuses on anatomically correct lung regions rather than background artifacts, text labels, or imaging borders.

---

## Dataset

The dataset is based on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) collection. Metadata and labels are loaded from `Chest_xray_Corona_Metadata.csv`. Labels are derived from the `Label` and `Label_1_Virus_category` columns and mapped to three classes:

- `Normal`
- `Pnemonia-Bacteria`
- `Pnemonia-Virus`

---

## Team Contributions

| Member             | Contribution                                                                                    |
|--------------------|-------------------------------------------------------------------------------------------------|
| Abdallah Elshafey  | Report writing and code review                                                                  |
| Josh Brauner       | Pretrained model screening, final model selection, code cleanup for classification and training    |
| Tom Fatoke         | Model training setup and initial hyperparameter tuning, report drafting, filling in training details                              |
| Toheeb Ayuba       | Hyperparameter tuning experiments, class weighting implementation, final report preparation     |
