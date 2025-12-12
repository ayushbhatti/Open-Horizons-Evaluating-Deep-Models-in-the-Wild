# CIFAR-10 Few-Shot Class-Incremental Learning (FSCIL)

## Overview

Implementation of SPPR for few-shot class-incremental learning.

**Task**: 7 base classes → incrementally add 3 new classes with 1/5/10 shots

**Strategies**:

- `SPPR` - Self Prompted Prototype Refinment

## Installation

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn
```

## Training

```bash
cd Open-Horizons-Evaluating-Deep-Models-in-the-Wild/CIL Framework/SPPR

# Train all
python train.py --epochs 50 --batch_size 64 --lr 0.001 --dropout 0.5 --patience 20 --n_way 5 --k_shot 10 --use_ress

```

**Args**: `--epochs`, `--batch_size`,`--lr`,`--dropout`,`--patience`,`--n_way 5`,`--k_shot 10`,`--use_ress`

## Testing

```bash
# Test
python test.py --checkpoint ./checkpoints/best_model.pth --k_shot 5
```

**Args**: `--checkpoint`, `--k_shot`

**Outputs**: confusion matrix, per-class accuracy chart, prototype distribution,results.txt


## Files

- `train.py` - Training script
- `test.py` - Testing with visualizations
- `self_prompting.py` - Resnet50 with frozen backbone