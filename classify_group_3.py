# imports
import collections
import tarfile
import torch.nn.functional as F
import random
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from train_group_3 import (
    METADATA_PATH, CLASS_NAMES, NUM_CLASSES, SEED, set_seed, DEVICE, load_metadata,
    ChestXrayDataset, USE_AMP, criterion, BATCH_SIZE
)

# constants
TEST_DIR = "./test"
MODEL_DIR = "densenet161_best.pth.tar.gz"

# functions

# evaluate model using test data, returning accuracy, micro AUC, micro precision, micro recall, and micro F1
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels_device = labels.to(device)

        if USE_AMP:
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels_device)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels_device)

        running_loss += loss.item() * inputs.size(0)

        probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs,
                        multi_class="ovr", average="micro")
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", zero_division=0
    )

    return {
        "loss": running_loss / len(all_labels),
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Convert a normalized tensor back to displayable image values in [0, 1]
def denormalize_image(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0, 1)

# Returns a 2D grayscale image for plotting from a 3-channel tensor
def get_display_image(img_tensor):
    img = denormalize_image(img_tensor).detach().cpu()
    img = img.permute(1, 2, 0).numpy()  # HWC
    # If the x-ray was duplicated into 3 channels, just average them for display
    gray = img.mean(axis=2)
    return gray

# Randomly choose test images, display original and Grad-CAM overlay,
# and show true/predicted labels.
def visualize(model, num_images=5):
    set_seed(SEED)

    metadata_df = load_metadata(METADATA_PATH)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ChestXrayDataset(
        metadata_df,
        "TEST",
        CLASS_NAMES,
        TEST_DIR,
        transform=test_transform
    )

    n = min(num_images, len(test_dataset))
    chosen_indices = random.sample(range(len(test_dataset)), n)

    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = np.array([axes])

    model.eval()

    for row, idx in enumerate(chosen_indices):
        image, true_label = test_dataset[idx]

        pred_idx, pred_name, pred_prob, cam = explain_image(model, image)

        base_img = get_display_image(image)

        # Left: original image
        axes[row, 0].imshow(base_img, cmap="gray")
        axes[row, 0].set_title(
            f"Original\nTrue: {CLASS_NAMES[true_label]}",
            fontsize=11
        )
        axes[row, 0].axis("off")

        # Right: Grad-CAM overlay
        axes[row, 1].imshow(base_img, cmap="gray")
        axes[row, 1].imshow(cam, cmap="jet", alpha=0.4)
        axes[row, 1].set_title(
            f"Pred: {pred_name} ({pred_prob:.3f})",
            fontsize=11
        )
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.show()

# Generate Grad-CAM explainability for a single image tensor
def explain_image(model, image):
    model.eval()

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        # Clone to avoid DenseNet's inplace ReLU issue
        output = output.clone()
        activations.append(output.detach())

        def grad_hook(grad):
            gradients.append(grad.detach())

        output.register_hook(grad_hook)
        return output

    # Hook the final feature tensor before classifier
    handle = model.features.register_forward_hook(forward_hook)

    try:
        x = image.unsqueeze(0).to(DEVICE)
        x.requires_grad_(True)

        model.zero_grad()

        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)

        pred_idx = probs.argmax(dim=1).item()
        pred_prob = probs[0, pred_idx].item()
        pred_name = CLASS_NAMES[pred_idx]

        score = outputs[0, pred_idx]
        score.backward()

        # Shapes: [1, C, H, W]
        acts = activations[0]
        grads = gradients[0]

        # Global average pooling of gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Resize to original transformed image size
        cam = F.interpolate(
            cam,
            size=(image.shape[1], image.shape[2]),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return pred_idx, pred_name, pred_prob, cam

    finally:
        handle.remove()

# main
def main():
    set_seed(SEED)
    metadata_df = load_metadata(METADATA_PATH)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ChestXrayDataset(metadata_df, "TEST", CLASS_NAMES,
                                    TEST_DIR, transform=test_transform)

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    with tarfile.open(MODEL_DIR, "r:gz") as tar:
        members = tar.getmembers()
        target = next(
            m for m in members
            if m.name.endswith(".pt") or m.name.endswith(".pth")
        )

        with tar.extractfile(target) as f:
            checkpoint = torch.load(f, map_location=DEVICE)

    # Build model architecture
    model = models.densenet161(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)

    # Load weights
    if isinstance(checkpoint, collections.OrderedDict):
        model.load_state_dict(checkpoint)
    elif isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # In case the whole dict is actually the state dict
            model.load_state_dict(checkpoint)
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    model = model.to(DEVICE)

    final_metrics = evaluate(model, test_loader, criterion, DEVICE)

    print("\n" + "=" * 60)
    print("FINAL TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy:       {final_metrics['accuracy']:.4f}")
    print(f"Micro AUC:      {final_metrics['auc']:.4f}")
    print(f"Micro Precision:{final_metrics['precision']:.4f}")
    print(f"Micro Recall:   {final_metrics['recall']:.4f}")
    print(f"Micro F1:       {final_metrics['f1']:.4f}")

    visualize(model, num_images=5)

if __name__ == "__main__":
    main()