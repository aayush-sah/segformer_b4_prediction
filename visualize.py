import cv2
import numpy as np
import os
from pathlib import Path

# ==================================================
# INPUT / OUTPUT
# ==================================================

input_folder = "./prediction_outputs"   # your predicted masks folder
output_folder = os.path.join(input_folder, "colorized")

os.makedirs(output_folder, exist_ok=True)

# ==================================================
# FIXED COLOR MAP FOR YOUR 10 CLASSES
# ==================================================

COLOR_MAP = {
    0: [34,139,34],     # Trees
    1: [0,255,0],       # Lush Bushes
    2: [189,183,107],   # Dry Grass
    3: [160,82,45],     # Dry Bushes
    4: [139,69,19],     # Ground Clutter
    5: [255,105,180],   # Flowers
    6: [101,67,33],     # Logs
    7: [128,128,128],   # Rocks
    8: [210,180,140],   # Landscape
    9: [135,206,235]    # Sky
}

# ==================================================
# LOAD MASK FILES
# ==================================================

image_extensions = ['.png']
image_files = [f for f in Path(input_folder).iterdir()
               if f.is_file() and f.suffix.lower() in image_extensions]

print(f"Found {len(image_files)} mask files to process")

# ==================================================
# PROCESS
# ==================================================

for image_file in sorted(image_files):

    print(f"Processing: {image_file.name}")

    mask = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)

    if mask is None:
        print(f"  Skipped: Could not read {image_file.name}")
        continue

    # Create RGB output image
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, color in COLOR_MAP.items():
        color_mask[mask == class_id] = color

    output_path = os.path.join(output_folder, image_file.name)
    cv2.imwrite(output_path, color_mask)

    print(f"  Saved: {output_path}")

print("\nColorization Complete.")
print(f"Saved to: {output_folder}")
