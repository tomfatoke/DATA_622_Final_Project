#Using the densenet161 as it was the best model from the fast_model_screening file
# IMPORTS
import copy
import tarfile
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

print("All imports successful")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Set paths
TRAIN_DIR       = "./train"
METADATA_PATH   = "Chest_xray_Corona_Metadata.csv"
MODEL_SAVE_PATH = "densenet161_best.pth"

#  Classes 
CLASS_NAMES = ["Normal", "Pnemonia-Bacteria", "Pnemonia-Virus"]
NUM_CLASSES = 3

#  Hyperparameters 
NUM_EPOCHS    = 25
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4    #only used for the warmup 
WARMUP_EPOCHS = 5       #epochs before unfreezing the backbone
FINETUNE_LR   = 1e-5   #LR after backbone unfrozen
PATIENCE      = 5       #when to stop if there has been no improvement
SEED          = 42

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): 
    torch.backends.cudnn.benchmark = True
USE_AMP = torch.cuda.is_available() 

print(f"Device: {DEVICE}")
print(f"Model will be saved to: {MODEL_SAVE_PATH}") 

# REPRODUCIBILITY
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print("Seeds set") 

# METADATA HELPERS
# Reading the CSV and maps rows to one of the three classes we have
def normalize_text(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()

def derive_final_label(label, virus_category):
    label     = normalize_text(label)
    virus_cat = normalize_text(virus_category)
    
    # Handles common misspellings/variants
    if label == "normal":
        return "Normal"
    if label in {"pnemonia", "pneumonia"}:
        if virus_cat == "bacteria":
            return "Pnemonia-Bacteria"
        elif virus_cat == "virus":
            return "Pnemonia-Virus"
    return None  

#cleaning the data by turning non-strings into strings and removes accidental spaces before/after the text
#also applies final labels to the data based on the rules defined above
def load_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    df["Dataset_type_norm"] = df["Dataset_type"].astype(str).str.strip().str.upper()
    df["final_label"] = df.apply(
        lambda row: derive_final_label(row["Label"], row["Label_1_Virus_category"]),
        axis=1
    )
    return df
#maps image filenames to their paths and loops through one at a time
#also checks subfolders with rglob and image types like .png
def build_image_path_lookup(base_dir):
    lookup = {}
    for p in Path(base_dir).rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            lookup[p.name] = str(p)
    return lookup

print("Metadata helpers defined") 

# DATASET CLASS
# Loads images on demand and does not load everything into RAM
class ChestXrayDataset(Dataset):
    def __init__(self, metadata_df, dataset_type, class_names,
                 train_dir, transform=None):
        self.transform    = transform
        self.class_names  = class_names #saves class names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        dataset_type      = dataset_type.upper()

        image_lookup = build_image_path_lookup(train_dir)
        df = metadata_df[metadata_df["Dataset_type_norm"] == dataset_type].copy()

        #holds image records and lets us know how many were skipped
        self.rows = []
        skipped   = 0

        for _, row in df.iterrows():
            final_label = row["final_label"]
            image_name  = str(row["X_ray_image_name"]).strip()

            if final_label is None:
                skipped += 1
                continue
            image_path = image_lookup.get(image_name)
            if image_path is None:
                skipped += 1
                continue

            self.rows.append({
                "image_name": image_name,
                "image_path": image_path,
                "label_name": final_label,
                "label_idx":  self.class_to_idx[final_label],
            })
        #summary to verify the data loaded correctly
        print(f"{dataset_type}: {len(self.rows)} images loaded, {skipped} skipped")

    def __len__(self):
        return len(self.rows)
    #loading images in RGB because densenet161 requires 3 channels for colour input
    def __getitem__(self, idx):
        row   = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["label_idx"]

print("Dataset class defined") 

# TRANSFORMS
#images pass through this transformation pipeline 
train_transforms = transforms.Compose([ 
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(p=0.5), #randomly mirrors the image
    transforms.RandomRotation(degrees=10), #Anywhere between 10 and 20 should work based on the transforms page
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), #Not 100% sure if saturation and hue will improve the model but can remove it if necessary
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])  #based on the documentation for transforms
])

print("Transforms defined") 

# LOADING DATA
# This is to verify the paths are correct before we train the model
metadata_df = load_metadata(METADATA_PATH) 

train_dataset = ChestXrayDataset( 
    metadata_df, "TRAIN", CLASS_NAMES, 
    TRAIN_DIR, transform=train_transforms
)

train_loader = DataLoader( 
    train_dataset, batch_size = BATCH_SIZE, 
    shuffle = True, num_workers = 2, pin_memory = True
)

#to verify the distribution of classes in the data
train_counts = Counter(row["label_name"] for row in train_dataset.rows)
print(f"Train class counts: {dict(train_counts)}")

# MODEL SETUP
def create_model(num_classes, device): 
    model = models.densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)  #uses model pretrained weights for detecting edges, shapes etc etc

    in_features = model.classifier.in_features 
    model.classifier = nn.Linear(in_features, num_classes) #replaces final layer of 1000 with 3 for the xray classes

    #only final classifying layer is used for training
    for name, param in model.named_parameters(): 
        if "classifier" not in name: 
            param.requires_grad = False 
        
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print(f"Trainable parameters: {trainable:} (backbone frozen)") 
    return model.to(device) 

def unfreeze_backbone(model, device): 
    for param in model.parameters(): 
        param.requires_grad = True 
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print(f"Backbone unfrozen, trainable parameters: {trainable:}") #to let us know the full number of params

def compute_class_weights(dataset, num_classes, device): 
    counts = Counter(row["label_idx"] for row in dataset.rows) 
    total = sum(counts.values()) 
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)] #so the classes with fewer images will be rare and get a higher weighting
    print(f"Class weights: {[round(w, 2) for w in weights]}") 
    return torch.tensor(weights, dtype=torch.float).to(device) 

model = create_model(NUM_CLASSES, DEVICE)
class_weights = compute_class_weights(train_dataset, NUM_CLASSES, DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
) 

scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6) 
scaler = torch.amp.GradScaler("cuda") if USE_AMP else None 
print("Model setup complete") 

# TRAINING (to update weights)
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    pbar = tqdm(loader, desc="Training", leave=False) #just the progress bar for training
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) #clears gradients for each batch of 32

        if USE_AMP:
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds    = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total   += inputs.size(0)

        pbar.set_postfix({"loss": f"{running_loss/total:.4f}",
                          "acc":  f"{correct/total:.4f}"})

    return running_loss / total, correct / total

def main():
    global optimizer, scheduler
    # FULL TRAINING LOOP
    # with an unfrozen backbone and early stopping based on validation loss
    # This does the proper training necessary
    best_val_loss   = float("inf")
    epochs_no_improve = 0

    print("training")
    print(f"Epochs: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | Device: {DEVICE}")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        if epoch == WARMUP_EPOCHS + 1:
            print("\nWarmup complete — unfreezing backbone")
            unfreeze_backbone(model, DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=NUM_EPOCHS - WARMUP_EPOCHS,
                                          eta_min=1e-8)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE
        )
        scheduler.step()
        #how long it takes for each epoch
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"time: {epoch_time/60:.1f}min"
        )

        # need to save the best model based on val loss and add early stopping based on patience hyperparameter
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, MODEL_SAVE_PATH)
            with tarfile.open(""+MODEL_SAVE_PATH+".tar.gz", "w:gz") as tar:
                tar.add(MODEL_SAVE_PATH)
            tar.close()
            print(f" -> Best model saved to {MODEL_SAVE_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f" -> No improvement ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("\nTraining complete!")

if __name__ == "__main__":
    main()