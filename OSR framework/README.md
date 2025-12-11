# Unified Open-Set Recognition (OSR) Framework

This architecture focuses on **backbone-agnostic OSR evaluation**, enabling controlled and fair comparison across convolutional and transformer-based encoders under identical experimental conditions.

---

## Repository Structure
```
OSR framework/
├── Data/
│   └── CIFAR10_loader.py        # CIFAR-10 loading and OSR split logic
├── Models/
│   ├── Backbone.py              # Backbone definitions (ResNet, ConvNeXt, CLIP)
│   └── Linear_head.py           # Linear classifier head
├── Utils/
│   ├── Training.py              # Feature extraction & linear head training
│   ├── Scoring.py               # OSR scoring methods
│   ├── Metrics.py               # AUROC, AUPR, FPR@95, OSCR computation
│   ├── Visualization.py         # ROC curves, histograms, OSCR plots
│   └── misc.py                  # Utilities (seeding, helpers)
├── main_pipeline.py             # End-to-end OSR experiment pipeline                
├── Results.txt                  # Tabulated experimental results
├── Requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```
## Installation
```
pip install -r Requirements.txt
```
## Quick Start
Run the full OSR pipeline (train linear head + evaluate OSR scores + generate plots)

### RESNET 50 backbone
```
python main_pipeline.py --known-classes 6 --backbone resnet50 --epochs 50
```
### ConvNeXt-Tiny 
```
python main_pipeline.py --known-classes 6 --backbone convnext_tiny --epochs 50
```
### CLIP ViT-B/16
```
python Scripts/main_pipeline.py \                                                                                                 
  --known-classes 6 \
  --backbone clip_vit_b16 \
  --epochs 50 \
  --batch-size 256
```
## Configuration
```
Key command-line arguments:
	•	--known-classes : Number of known classes (remaining CIFAR-10 classes are treated as OOD)
	•	--backbone: Backbone encoder
	•	resnet50
	•	convnext_tiny
	•	clip_vit
	•	--epochs: Number of epochs for linear head training
	•	--batch-size: Batch size (default: 256)
	•	--lr: Learning rate for linear classifier (default: 1e-2)
	•	--seed: Random seed for reproducibility
```
## Outputs

### Console Metrics

The pipeline reports the following OSR metrics for each scoring method:
- AUROC: Area Under ROC Curve
- AUPR: Area Under Precision–Recall Curve
- PR@95: False Positive Rate when OOD detection is 95%
- OSCR: Open-Set Classification Rate


### Saved Plots

All visualizations are saved to the plots/ directory, including:
- roc_curves.png — ROC curves for OSR scoring methods
- oscr_curve.png — OSCR curve
- hist_msp.png — Score histogram (MSP)
- hist_energy.png — Score histogram (Energy)
- hist_mahalanobis.png — Score histogram (Mahalanobis)
- hist_knn.png — Score histogram (kNN)

These plots are generated using Utils/Visualization.py.

## Dataset Download
CIFAR-10 is automatically downloaded on first run to the local ./data directory.
