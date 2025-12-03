# CIFAR-10 Few-Shot Class-Incremental Learning (FSCIL)

## Overview

Implementation of BASELINE, ORCO, CONCM, and CONCM_FULL strategies for few-shot class-incremental learning.

**Task**: 7 base classes → incrementally add 3 new classes with 1/5/10 shots

**Strategies**:

- `BASELINE` - No incremental learning (demonstrates problem)
- `ORCO` - Orthogonal pseudo-targets + contrastive losses
- `CONCM` - Simple mean calibration
- `CONCM_FULL` - Full ConCM with MPC + DSM modules

## Installation

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn
```

## Training

```bash
cd Open-Horizons-Evaluating-Deep-Models-in-the-Wild/Code_Deniz

# Train all
python train.py

# Train specific strategy
python train.py --strategy ORCO
python train.py --strategy CONCM_FULL
python train.py --strategy BASELINE --experiment 2
```

**Args**: `--strategy` (all/BASELINE/ORCO/CONCM/CONCM_FULL), `--experiment` (1/2/3)

## Testing

```bash
# Generate fixed test samples (once)
python generate_test_samples_CIFAR10.py

# Test
python test.py \
    --checkpoint ../../checkpoints_cifar_resnet50/ORCO_exp1_pretrained_frozen_best.pt \
    --strategy ORCO \
    --k_shot 5 \
    --pretrained \
    --fixed_samples fixed_test_samples.csv

# Batch testing
bash run_test_example.sh
```

**Args**: `--checkpoint`, `--strategy`, `--k_shot`, `--pretrained`, `--fixed_samples`

**Outputs**: confusion matrix, per-class accuracy chart, results.txt

## Experiments

| Exp | Encoder    | Layers Trained        | Strategies                        |
| --- | ---------- | --------------------- | --------------------------------- |
| 1   | Pretrained | Classifier only       | ORCO, CONCM, CONCM_FULL           |
| 2   | Scratch    | All                   | BASELINE, ORCO, CONCM, CONCM_FULL |
| 3   | Pretrained | Classifier + layer3/4 | ORCO, CONCM, CONCM_FULL           |

## Strategy Details

| Strategy   | Training                  | Testing                   |
| ---------- | ------------------------- | ------------------------- |
| BASELINE   | CE only                   | No calibration            |
| ORCO       | CE + 3 contrastive losses | Blend with pseudo-targets |
| CONCM      | CE only                   | Blend with base mean      |
| CONCM_FULL | CE + DSM alignment        | MPC + DSM calibration     |

## Files

- `train.py` - Training script
- `test.py` - Testing with visualizations
- `model.py` - ResNet50 + loss functions
- `concm_full.py` - Full ConCM framework
- `generate_test_samples_CIFAR10.py` - Fixed samples generator
