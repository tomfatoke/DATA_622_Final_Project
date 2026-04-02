#Using the densenet161 as it was the best model from the fast_model_screening file
# ============================================================
# IMPORTS
# ============================================================
import os
import copy
import time
import random
import numpy as np
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

import torchvision.models as models
from torchvision.models import DenseNet161_Weights
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support
)

print("All imports successful")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}") 


# ============================================================
# HYPERPARAMETERS
# ============================================================


DATA_DIR        = "Coronahack-Chest-XRay-Dataset"
TRAIN_DIR       = os.path.join(DATA_DIR, "train")
TEST_DIR        = os.path.join(DATA_DIR, "test")
METADATA_PATH   = "Chest_xray_Corona_Metadata.csv"
MODEL_SAVE_PATH = "densenet161_best.pth"

#  Classes 
CLASS_NAMES = ["Normal", "Pnemonia-Bacteria", "Pnemonia-Virus"]
NUM_CLASSES = 3

#  Hyperparameters 
NUM_EPOCHS    = 25
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3    # Used during warmup (head only)
WARMUP_EPOCHS = 3       # Epochs before unfreezing backbone
FINETUNE_LR   = 1e-5   # Lower LR after unfreezing
PATIENCE      = 5       # Early stopping patience
SEED          = 42

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

print(f"Device: {DEVICE}")
print(f"Model will be saved to: {MODEL_SAVE_PATH}") 

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print("Seeds set")