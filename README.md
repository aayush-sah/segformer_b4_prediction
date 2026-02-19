Duality AI – Offroad Semantic Segmentation
Robust Scene Understanding for Desert Autonomy

Team: 404 BRAIN NOT FOUND
Track: Semantic Segmentation
Goal: Train a model on synthetic desert data and generalize to a novel unseen environment.

1. Problem

Unmanned Ground Vehicles require pixel-level scene understanding to safely navigate off-road desert terrain.
The challenge:
Train on synthetic desert dataset
Evaluate on a novel desert environment
Segment 10 terrain classes
Maintain robustness under distribution shift

Main risks:
Class imbalance (rare objects like Logs)
Occlusion
Boundary inconsistencies
Limited hardware resources

2. Methodology

We trained two independent models on different devices, applying targeted optimization strategies.

Model A — UNet++ (CNN-Based)

Backbone: EfficientNet-B4
Device: 4GB GPU

Training Setup:
Dice + CrossEntropy Loss
Class weighting
Mixed Precision (FP16)
Adam optimizer
Gradient clipping

Fine-Tuning:
Hard example mining
Boundary-aware adjustments
Test-Time Augmentation (TTA)
Bilateral smoothing for edges

Model B — SegFormer MiT-B4 (Transformer-Based)

Backbone: MiT-B4 (pretrained)
Device: RTX 3050 (6GB)

Training Setup:
AdamW optimizer
OneCycleLR scheduler
Dice + CrossEntropy Loss
Label smoothing
Mixed Precision

Fine-Tuning
Stage-wise training (decoder → full network)
Strong geometric augmentation
Class imbalance correction
Learning rate refinement

3. Results & Performance

| Model        | Device   | Final mIoU |
| ------------ | -------- | ---------- |
| UNet++       | 4GB GPU  | 0.5887     |
| SegFormer-B4 | RTX 3050 | **0.61**   |

Key Observations

Transformer improved global context understanding:
CNN handled rare class recovery better after weighting
Dice loss improved recall
Augmentation improved generalization

Metrics evaluated:
Mean IoU
Per-class IoU
Confusion Matrix
Pixel Accuracy

5. Conclusion

Achieved 0.61 mIoU on novel environment
Demonstrated CNN vs Transformer comparison
Maintained strict separation of train/val/test sets
Trained exclusively on provided dataset
Synthetic digital twin data proved effective for segmentation robustness in off-road autonomy.

6. Future Work

Ensemble CNN + Transformer
Domain adaptation to real-world desert footage
Model quantization for edge deployment
Larger backbone experimentation
