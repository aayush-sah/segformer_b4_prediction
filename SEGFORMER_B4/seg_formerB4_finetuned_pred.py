import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from sklearn.metrics import confusion_matrix

# ==================================================
# CONFIG
# ==================================================

NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"D:\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset"
MODEL_PATH = "./train_stats/best_model.pth"

SAVE_DIR = "./prediction_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs",
    "Rocks", "Landscape", "Sky"
]

COLORS = np.array([
    [34,139,34],
    [0,255,0],
    [189,183,107],
    [160,82,45],
    [139,69,19],
    [255,105,180],
    [101,67,33],
    [128,128,128],
    [210,180,140],
    [135,206,235]
])

print("Using device:", DEVICE)

# ==================================================
# LOAD MODEL
# ==================================================

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model Loaded.")

# ==================================================
# LOSS FUNCTIONS
# ==================================================

ce_loss = nn.CrossEntropyLoss()

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, NUM_CLASSES).permute(0,3,1,2).float()
    intersection = (pred * target_onehot).sum((2,3))
    union = pred.sum((2,3)) + target_onehot.sum((2,3))
    return 1 - ((2*intersection + smooth) / (union + smooth)).mean()

# ==================================================
# MASK CONVERSION
# ==================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return new_arr

# ==================================================
# TRANSFORM
# ==================================================

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================================================
# COLORIZE
# ==================================================

def colorize(mask):
    return COLORS[mask]

# ==================================================
# VALIDATION LOOP
# ==================================================

val_img_dir = os.path.join(DATA_ROOT, "val", "Color_Images")
val_mask_dir = os.path.join(DATA_ROOT, "val", "Segmentation")

files = os.listdir(val_img_dir)

conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
total_loss = total_ce = total_dice = 0

start_time = time.time()

print("\nRunning Evaluation...\n")

for name in tqdm(files):

    img = Image.open(os.path.join(val_img_dir, name)).convert("RGB")
    mask = Image.open(os.path.join(val_mask_dir, name))

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    gt = convert_mask(mask)
    gt = np.array(Image.fromarray(gt).resize((512,512), Image.NEAREST))
    gt_tensor = torch.from_numpy(gt).long().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor).logits
        outputs = F.interpolate(outputs, size=(512,512), mode="bilinear")

        ce = ce_loss(outputs, gt_tensor)
        dice = dice_loss(outputs, gt_tensor)
        loss = 0.5 * ce + 0.5 * dice

    total_loss += loss.item()
    total_ce += ce.item()
    total_dice += dice.item()

    pred = torch.argmax(outputs, dim=1)[0].cpu().numpy()

    conf_matrix += confusion_matrix(
        gt.flatten(),
        pred.flatten(),
        labels=list(range(NUM_CLASSES))
    )

    # Save colored prediction
    color_pred = colorize(pred).astype(np.uint8)
    Image.fromarray(color_pred).save(
        os.path.join(SAVE_DIR, f"{name}_color.png")
    )

    # Save overlay
    resized_img = np.array(img.resize((512,512)))
    overlay = (0.6 * resized_img + 0.4 * color_pred).astype(np.uint8)
    Image.fromarray(overlay).save(
        os.path.join(SAVE_DIR, f"{name}_overlay.png")
    )

# ==================================================
# COMPUTE METRICS
# ==================================================

ious = []

for cls in range(NUM_CLASSES):
    intersection = conf_matrix[cls, cls]
    union = (
        conf_matrix[cls, :].sum() +
        conf_matrix[:, cls].sum() -
        intersection
    )
    iou = intersection / union if union != 0 else 0
    ious.append(iou)

mean_iou = np.mean(ious)
val_loss = total_loss / len(files)
val_ce = total_ce / len(files)
val_dice = total_dice / len(files)

elapsed = int(time.time() - start_time)

# ==================================================
# PRINT FORMATTED OUTPUT
# ==================================================

print("\n")
print(f"P2 [15/15] loss={val_loss:.4f} (ce={val_ce:.3f} dice={val_dice:.3f} bnd=0.000) val={val_loss:.4f} mIoU={mean_iou:.4f} ({elapsed}s)\n")

for name, score in zip(CLASS_NAMES, ious):
    print(f"{name:<15} : {score:.4f}")

# ==================================================
# SAVE GRAPHS
# ==================================================

# IoU Bar Graph
plt.figure(figsize=(10,6))
plt.bar(CLASS_NAMES, ious)
plt.xticks(rotation=45)
plt.ylabel("IoU")
plt.title(f"Per-Class IoU (mIoU={mean_iou:.4f})")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "iou_bar_graph.png"))
plt.close()

# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.close()

print("\nGraphs and predictions saved in:", SAVE_DIR)
print("Evaluation Complete.")
