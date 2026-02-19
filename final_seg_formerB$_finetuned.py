import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast, GradScaler

plt.switch_backend("Agg")

# ==================================================
# CONFIG
# ==================================================

NUM_CLASSES = 10
BATCH_SIZE = 2
EPOCHS = 40
LR = 6e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"D:\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset"
OUTPUT_DIR = "./train_stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Using device:", DEVICE)

# ==================================================
# MASK VALUE MAPPING
# ==================================================

value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr

# ==================================================
# DATASET
# ==================================================

class OffroadDataset(Dataset):
    def __init__(self, root, split):
        self.img_dir = os.path.join(root, split, "Color_Images")
        self.mask_dir = os.path.join(root, split, "Segmentation")
        self.files = os.listdir(self.img_dir)

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.NEAREST)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))

        img = self.transform(img)
        mask = self.mask_transform(mask)

        mask = convert_mask(mask)
        mask = torch.from_numpy(mask).long()

        return img, mask

train_loader = DataLoader(
    OffroadDataset(DATA_ROOT, "train"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,      # Windows safe
    pin_memory=True
)

val_loader = DataLoader(
    OffroadDataset(DATA_ROOT, "val"),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,      # Windows safe
    pin_memory=True
)

# ==================================================
# MODEL
# ==================================================

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)

# ==================================================
# LOSS
# ==================================================

class_weights = torch.tensor([
    1.5, 1.5, 1.0, 1.5, 2.0,
    3.0, 3.0, 2.5, 0.3, 0.2
]).to(DEVICE)

ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, NUM_CLASSES).permute(0,3,1,2).float()
    intersection = (pred * target_onehot).sum((2,3))
    union = pred.sum((2,3)) + target_onehot.sum((2,3))
    return 1 - ((2*intersection + smooth) / (union + smooth)).mean()

def combined_loss(pred, target):
    return 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)

# ==================================================
# OPTIMIZER
# ==================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
)

scaler = GradScaler("cuda")

# ==================================================
# METRIC
# ==================================================

def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1)
    ious = []
    for cls in range(NUM_CLASSES):
        inter = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        if union == 0:
            continue
        ious.append((inter/union).item())
    return np.mean(ious)

# ==================================================
# FULL 3-STAGE TRAINING PIPELINE
# ==================================================

best_iou = 0

print("\n==============================")
print("STAGE 1: MAIN TRAINING (20 EPOCHS)")
print("==============================\n")

# -------------------------------
# STAGE 1 : FAST LEARNING
# -------------------------------

for epoch in range(20):

    model.train()

    for imgs, masks in tqdm(train_loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(pixel_values=imgs).logits
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear")
            loss = combined_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    # Validation
    model.eval()
    val_iou = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with autocast("cuda"):
                outputs = model(pixel_values=imgs).logits
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear")

            val_iou.append(compute_iou(outputs, masks))

    avg_val_iou = np.mean(val_iou)
    print(f"Stage1 Epoch {epoch+1}/20 | Val IoU: {avg_val_iou:.4f}")

    if avg_val_iou > best_iou:
        best_iou = avg_val_iou
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")

print("\nStage 1 Complete. Best IoU:", best_iou)


# -------------------------------
# STAGE 2 : REFINEMENT
# -------------------------------

print("\n==============================")
print("STAGE 2: REFINEMENT (15 EPOCHS)")
print("==============================\n")

model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth"))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(15):

    model.train()

    for imgs, masks in tqdm(train_loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(pixel_values=imgs).logits
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear")
            loss = combined_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Validation
    model.eval()
    val_iou = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with autocast("cuda"):
                outputs = model(pixel_values=imgs).logits
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear")

            val_iou.append(compute_iou(outputs, masks))

    avg_val_iou = np.mean(val_iou)
    print(f"Stage2 Epoch {epoch+1}/15 | Val IoU: {avg_val_iou:.4f}")

    if avg_val_iou > best_iou:
        best_iou = avg_val_iou
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")

print("\nStage 2 Complete. Best IoU:", best_iou)


# -------------------------------
# STAGE 3 : ULTRA REFINEMENT
# -------------------------------

print("\n==============================")
print("STAGE 3: ULTRA REFINEMENT (10 EPOCHS)")
print("==============================\n")

model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth"))

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

for epoch in range(10):

    model.train()

    for imgs, masks in tqdm(train_loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(pixel_values=imgs).logits
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear")
            loss = combined_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Validation
    model.eval()
    val_iou = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with autocast("cuda"):
                outputs = model(pixel_values=imgs).logits
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear")

            val_iou.append(compute_iou(outputs, masks))

    avg_val_iou = np.mean(val_iou)
    print(f"Stage3 Epoch {epoch+1}/10 | Val IoU: {avg_val_iou:.4f}")

    if avg_val_iou > best_iou:
        best_iou = avg_val_iou
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")
print("Final Best IoU:", best_iou)
