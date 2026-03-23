import os
import csv
import time
import copy
import random
import traceback
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm

import torchvision
from torchvision.models import list_models, get_model, get_model_weights

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# =========================================================
# USER SETTINGS
# =========================================================
DATA_DIR = r"Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Path to your metadata file
METADATA_PATH = "Chest_xray_Corona_Metadata.csv"

RESULTS_CSV = "fast_model_screening_results.csv"

# Force-skip any models you do not want to try
MANUALLY_SKIPPED_MODELS = {
    "regnet_y_128gf",
    "vit_h_14"
}

# Final 3 classes
CLASS_NAMES = ["Normal", "Pnemonia-Bacteria", "Pnemonia-Virus"]
NUM_CLASSES = len(CLASS_NAMES)

# Fast screening settings
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_WORKERS = 4
TRAIN_FRACTION = 0.25
TEST_FRACTION = 0.50
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

ALL_MODELS = list_models(module=torchvision.models)


# =========================================================
# REPRODUCIBILITY / PERFORMANCE
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# =========================================================
# RESULTS RESUME HELPERS
# =========================================================
def get_completed_models_from_csv(csv_path):
    completed = set()

    if not os.path.exists(csv_path):
        return completed

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row.get("model_name", "").strip()
            status = row.get("status", "").strip().lower()
            if model_name and status in {"success", "failed"}:
                completed.add(model_name)

    return completed


def get_models_to_run(all_models, csv_path, manually_skipped_models):
    completed_models = get_completed_models_from_csv(csv_path)
    remaining = [
        m for m in all_models
        if m not in completed_models and m not in manually_skipped_models
    ]
    return remaining, completed_models


def append_result_to_csv(csv_path, row):
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(row)

def sort_results_csv_by_accuracy(csv_path, output_path=None):
    """
    Sort results by test_accuracy descending.
    If output_path is None, overwrite the original file.
    """
    if not os.path.exists(csv_path):
        print(f"Cannot sort results: file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "test_accuracy" not in df.columns:
        print("Cannot sort results: 'test_accuracy' column not found.")
        return

    df["test_accuracy"] = pd.to_numeric(df["test_accuracy"], errors="coerce")

    # Optional tie-breakers
    sort_cols = ["test_accuracy"]
    ascending = [False]

    if "test_f1_weighted" in df.columns:
        df["test_f1_weighted"] = pd.to_numeric(df["test_f1_weighted"], errors="coerce")
        sort_cols.append("test_f1_weighted")
        ascending.append(False)

    if "train_time_sec" in df.columns:
        df["train_time_sec"] = pd.to_numeric(df["train_time_sec"], errors="coerce")
        sort_cols.append("train_time_sec")
        ascending.append(True)   # faster models win ties

    df = df.sort_values(by=sort_cols, ascending=ascending, na_position="last")

    if output_path is None:
        output_path = csv_path

    df.to_csv(output_path, index=False)
    print(f"Sorted results saved to: {output_path}")


# =========================================================
# METADATA HELPERS
# =========================================================
def normalize_text(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()


def derive_final_label(label, Label_1_Virus_category):
    """
    Maps metadata rows to one of:
      - Normal
      - Pnemonia-Bacteria
      - Pnemonia-Virus

    Returns None for rows that do not cleanly map.
    """
    label = normalize_text(label)
    virus_cat = normalize_text(Label_1_Virus_category)

    # Handle common misspellings/variants
    if label in {"normal"}:
        return "Normal"

    if label in {"pnemonia", "pneumonia"}:
        if virus_cat == "bacteria":
            return "Pnemonia-Bacteria"
        elif virus_cat == "virus":
            return "Pnemonia-Virus"
        else:
            return None

    return None


def load_metadata(metadata_path):
    df = pd.read_csv(metadata_path)

    required_cols = {
        "X_ray_image_name",
        "Label",
        "Dataset_type",
        "Label_1_Virus_category",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata file is missing required columns: {sorted(missing)}")

    df = df.copy()

    df["Dataset_type_norm"] = df["Dataset_type"].astype(str).str.strip().str.upper()
    df["final_label"] = df.apply(
        lambda row: derive_final_label(row["Label"], row["Label_1_Virus_category"]),
        axis=1
    )

    return df


def build_image_path_lookup(train_dir, test_dir):
    """
    Builds a lookup from image filename -> full path.
    Assumes image names are unique across train/test as listed in metadata.
    """
    lookup = {}

    for base_dir in [train_dir, test_dir]:
        for p in Path(base_dir).rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                lookup[p.name] = str(p)

    return lookup


# =========================================================
# DATASET
# =========================================================
class ChestXrayMetadataDataset(Dataset):
    def __init__(self, metadata_df, dataset_type, class_names, train_dir, test_dir, transform=None):
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.dataset_type = dataset_type.upper()

        image_lookup = build_image_path_lookup(train_dir, test_dir)

        df = metadata_df.copy()
        df = df[df["Dataset_type_norm"] == self.dataset_type].copy()

        rows = []
        skipped_missing_label = 0
        skipped_missing_file = 0

        for _, row in df.iterrows():
            final_label = row["final_label"]
            image_name = str(row["X_ray_image_name"]).strip()

            if final_label is None:
                skipped_missing_label += 1
                continue

            image_path = image_lookup.get(image_name)
            if image_path is None:
                skipped_missing_file += 1
                continue

            rows.append({
                "image_name": image_name,
                "image_path": image_path,
                "label_name": final_label,
                "label_idx": self.class_to_idx[final_label],
            })

        if not rows:
            raise ValueError(f"No usable rows found for dataset type {self.dataset_type}")

        self.rows = rows
        self.skipped_missing_label = skipped_missing_label
        self.skipped_missing_file = skipped_missing_file

        print(
            f"{self.dataset_type}: kept {len(self.rows)} rows | "
            f"skipped unlabeled/invalid {self.skipped_missing_label} | "
            f"skipped missing files {self.skipped_missing_file}"
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = row["label_idx"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# =========================================================
# DATA HELPERS
# =========================================================
def summarize_dataset_labels(dataset):
    counts = Counter()

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        for idx in dataset.indices:
            label_name = base_dataset.rows[idx]["label_name"]
            counts[label_name] += 1
    else:
        for row in dataset.rows:
            counts[row["label_name"]] += 1

    return counts


def make_subset(dataset, fraction, seed):
    if fraction >= 1.0:
        return dataset

    n_total = len(dataset)
    n_keep = max(1, int(n_total * fraction))

    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    chosen = indices[:n_keep]

    return Subset(dataset, chosen)


def build_dataloaders(weights, metadata_df, train_dir, test_dir, class_names,
                      batch_size, num_workers, train_fraction=1.0, test_fraction=1.0, seed=42):
    preprocess = weights.transforms()

    full_train_dataset = ChestXrayMetadataDataset(
        metadata_df=metadata_df,
        dataset_type="TRAIN",
        class_names=class_names,
        train_dir=train_dir,
        test_dir=test_dir,
        transform=preprocess
    )

    full_test_dataset = ChestXrayMetadataDataset(
        metadata_df=metadata_df,
        dataset_type="TEST",
        class_names=class_names,
        train_dir=train_dir,
        test_dir=test_dir,
        transform=preprocess
    )

    train_dataset = make_subset(full_train_dataset, train_fraction, seed)
    test_dataset = make_subset(full_test_dataset, test_fraction, seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )

    return full_train_dataset, full_test_dataset, train_dataset, test_dataset, train_loader, test_loader


# =========================================================
# MODEL HELPERS
# =========================================================
def get_default_weights(model_name):
    weights_enum = get_model_weights(model_name)
    return weights_enum.DEFAULT


def replace_classifier(model, model_name, num_classes):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        layers = list(model.classifier.children())
        for i in range(len(layers) - 1, -1, -1):
            if isinstance(layers[i], nn.Linear):
                in_features = layers[i].in_features
                layers[i] = nn.Linear(in_features, num_classes)
                model.classifier = nn.Sequential(*layers)
                return model

    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        if isinstance(model.heads.head, nn.Linear):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
            return model

    raise ValueError(f"Could not replace classifier for model: {model_name}")


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

    unfroze_any = False

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        for param in model.fc.parameters():
            param.requires_grad = True
        unfroze_any = True

    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            for param in model.classifier.parameters():
                param.requires_grad = True
            unfroze_any = True
        elif isinstance(model.classifier, nn.Sequential):
            for layer in model.classifier:
                if isinstance(layer, nn.Linear):
                    for param in layer.parameters():
                        param.requires_grad = True
                    unfroze_any = True

    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        if hasattr(model.heads.head, "parameters"):
            for param in model.heads.head.parameters():
                param.requires_grad = True
            unfroze_any = True

    if not unfroze_any:
        raise ValueError("Could not identify classifier head to unfreeze.")

    return model


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# TRAINING / EVALUATION
# =========================================================
def train_one_model(model, train_loader, device, num_epochs, learning_rate,
                    model_name, model_idx, total_models):
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)

    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        pbar = tqdm(
            train_loader,
            desc=f"Model {model_idx}/{total_models} | {model_name} | Epoch {epoch+1}/{num_epochs}",
            leave=True,
            dynamic_ncols=True
        )

        for batch_idx, (inputs, labels) in enumerate(pbar, start=1):
            batch_start = time.time()

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            batch_size_now = inputs.size(0)
            running_loss += loss.item() * batch_size_now
            total_samples += batch_size_now

            preds = torch.argmax(outputs, dim=1)
            running_correct += (preds == labels).sum().item()

            avg_loss = running_loss / total_samples
            avg_acc = running_correct / total_samples
            batch_time = time.time() - batch_start
            elapsed = time.time() - epoch_start
            eta_seconds = (elapsed / batch_idx) * max(0, len(train_loader) - batch_idx)

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{avg_acc:.4f}",
                "batch_s": f"{batch_time:.2f}",
                "eta_m": f"{eta_seconds / 60:.1f}",
            })

        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples
        epoch_time = time.time() - epoch_start

        print(
            f"    Epoch {epoch+1}/{num_epochs} complete | "
            f"loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} | "
            f"time: {epoch_time/60:.2f} min"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    total_elapsed = time.time() - total_start
    model.load_state_dict(best_model_wts)

    return model, best_loss, total_elapsed


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    eval_start = time.time()

    pbar = tqdm(
        test_loader,
        desc="Evaluating",
        leave=False,
        dynamic_ncols=True
    )

    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)

        if USE_AMP:
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
        else:
            outputs = model(inputs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = labels.numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

    eval_time = time.time() - eval_start

    accuracy = accuracy_score(all_labels, all_preds)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "eval_time_sec": eval_time
    }


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)

    metadata_df = load_metadata(METADATA_PATH)

    models_to_run, completed_models = get_models_to_run(
        all_models=ALL_MODELS,
        csv_path=RESULTS_CSV,
        manually_skipped_models=MANUALLY_SKIPPED_MODELS
    )

    print("=" * 90)
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"AMP enabled: {USE_AMP}")
    print(f"Metadata path: {METADATA_PATH}")
    print(f"Total torchvision models available: {len(ALL_MODELS)}")
    print(f"Already completed from CSV: {len(completed_models)}")
    print(f"Manually skipped: {len(MANUALLY_SKIPPED_MODELS)}")
    print(f"Remaining to run now: {len(models_to_run)}")
    print(f"Final classes: {CLASS_NAMES}")
    print(f"Epochs per model: {NUM_EPOCHS}")
    print(f"Train fraction: {TRAIN_FRACTION}")
    print(f"Test fraction: {TEST_FRACTION}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 90)

    if not models_to_run:
        print("\nNo models left to run.")
        return

    first_weights = None
    first_model_name = None
    for candidate in models_to_run:
        try:
            first_weights = get_default_weights(candidate)
            first_model_name = candidate
            break
        except Exception:
            continue

    if first_weights is None:
        raise RuntimeError("Could not find any remaining model with default weights.")

    full_train_dataset, full_test_dataset, train_dataset, test_dataset, _, _ = build_dataloaders(
        weights=first_weights,
        metadata_df=metadata_df,
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        class_names=CLASS_NAMES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_fraction=TRAIN_FRACTION,
        test_fraction=TEST_FRACTION,
        seed=SEED
    )

    print("\nDataset summary")
    print("-" * 90)
    print(f"Sanity-check model weights source: {first_model_name}")
    print(f"Full train usable images: {len(full_train_dataset)}")
    print(f"Full test usable images: {len(full_test_dataset)}")
    print(f"Train subset used per model: {len(train_dataset)}")
    print(f"Test subset used per model: {len(test_dataset)}")
    print("Full train label counts:", dict(summarize_dataset_labels(full_train_dataset)))
    print("Full test label counts:", dict(summarize_dataset_labels(full_test_dataset)))
    print("-" * 90)

    run_start = time.time()
    completed_times = []

    for idx, model_name in enumerate(models_to_run, start=1):
        model_start = time.time()
        print(f"\n[{idx}/{len(models_to_run)}] Testing model: {model_name}")

        try:
            weights = get_default_weights(model_name)

            _, _, train_subset, test_subset, train_loader, test_loader = build_dataloaders(
                weights=weights,
                metadata_df=metadata_df,
                train_dir=TRAIN_DIR,
                test_dir=TEST_DIR,
                class_names=CLASS_NAMES,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                train_fraction=TRAIN_FRACTION,
                test_fraction=TEST_FRACTION,
                seed=SEED
            )

            print(f"    Train samples: {len(train_subset)} | Test samples: {len(test_subset)}")

            model = get_model(model_name, weights=weights)
            model = replace_classifier(model, model_name, NUM_CLASSES)
            model = freeze_backbone(model)

            trainable_params = count_trainable_parameters(model)
            print(f"    Trainable parameters: {trainable_params:,}")

            model, best_train_loss, train_time_sec = train_one_model(
                model=model,
                train_loader=train_loader,
                device=DEVICE,
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                model_name=model_name,
                model_idx=idx,
                total_models=len(models_to_run)
            )

            metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                device=DEVICE
            )

            model_elapsed = time.time() - model_start
            completed_times.append(model_elapsed)
            avg_model_time = sum(completed_times) / len(completed_times)
            models_left = len(models_to_run) - idx
            eta_remaining_min = (avg_model_time * models_left) / 60

            result_row = {
                "model_name": model_name,
                "weights": str(weights),
                "num_classes": NUM_CLASSES,
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "train_fraction": TRAIN_FRACTION,
                "test_fraction": TEST_FRACTION,
                "train_samples_used": len(train_subset),
                "test_samples_used": len(test_subset),
                "trainable_params": trainable_params,
                "best_train_loss": round(best_train_loss, 6),
                "train_time_sec": round(train_time_sec, 2),
                "eval_time_sec": round(metrics["eval_time_sec"], 2),
                "test_accuracy": round(metrics["accuracy"], 6),
                "test_precision_weighted": round(metrics["precision_weighted"], 6),
                "test_recall_weighted": round(metrics["recall_weighted"], 6),
                "test_f1_weighted": round(metrics["f1_weighted"], 6),
                "test_precision_macro": round(metrics["precision_macro"], 6),
                "test_recall_macro": round(metrics["recall_macro"], 6),
                "test_f1_macro": round(metrics["f1_macro"], 6),
                "status": "success",
                "error": ""
            }

            append_result_to_csv(RESULTS_CSV, result_row)

            print(
                f"    Done | "
                f"acc: {metrics['accuracy']:.4f} | "
                f"f1_w: {metrics['f1_weighted']:.4f} | "
                f"train: {train_time_sec/60:.2f} min | "
                f"eval: {metrics['eval_time_sec']/60:.2f} min | "
                f"ETA remaining: {eta_remaining_min:.1f} min"
            )

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            model_elapsed = time.time() - model_start
            completed_times.append(model_elapsed)
            avg_model_time = sum(completed_times) / len(completed_times)
            models_left = len(models_to_run) - idx
            eta_remaining_min = (avg_model_time * models_left) / 60

            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"    Failed | {error_msg}")
            traceback.print_exc()

            result_row = {
                "model_name": model_name,
                "weights": "",
                "num_classes": NUM_CLASSES,
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "train_fraction": TRAIN_FRACTION,
                "test_fraction": TEST_FRACTION,
                "train_samples_used": "",
                "test_samples_used": "",
                "trainable_params": "",
                "best_train_loss": "",
                "train_time_sec": "",
                "eval_time_sec": "",
                "test_accuracy": "",
                "test_precision_weighted": "",
                "test_recall_weighted": "",
                "test_f1_weighted": "",
                "test_precision_macro": "",
                "test_recall_macro": "",
                "test_f1_macro": "",
                "status": "failed",
                "error": error_msg
            }

            append_result_to_csv(RESULTS_CSV, result_row)
            print(f"    ETA remaining: {eta_remaining_min:.1f} min")
            torch.cuda.empty_cache()

    total_run_min = (time.time() - run_start) / 60

    sort_results_csv_by_accuracy(
        RESULTS_CSV,
        output_path="fast_model_screening_results_sorted.csv"
    )

    print("\n" + "=" * 90)
    print("Run complete.")
    print(f"Results appended to: {RESULTS_CSV}")
    print(f"Elapsed this run: {total_run_min:.2f} minutes")
    print("=" * 90)


if __name__ == "__main__":
    main()