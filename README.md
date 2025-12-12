# Open Horizons: Evaluating Deep Models in the Wild

<img width="720" height="405" alt="image" src="https://github.com/user-attachments/assets/3585df02-78a5-4f84-bda4-42f56ef55c30" />


## Overview

**Open Horizons** is a research-driven project that investigates how modern deep visual representations behave in *open-world* settings, where deployed models must transcend the closed-world assumption. In realistic environments, models encounter previously unseen classes, must decide when to abstain from confident predictions, and are often required to incorporate and learn new categories over time. This repository studies these challenges through the complementary lenses of **Open-Set Recognition (OSR)** and **Few-Shot Class-Incremental Learning (FSCIL)**.

Rather than proposing a new architecture, the project provides a **controlled empirical analysis** that isolates the role of representation quality, scoring functions, and incremental learning mechanisms. The emphasis is on understanding *why* certain models generalize better to unknowns and *how* different learning paradigms interact under open-world constraints.

---

## Conceptual Motivation

Traditional supervised learning assumes that all test samples belong to classes seen during training. This assumption breaks down in real-world deployments, where systems must:

- Correctly classify known classes  
- Reject unknown or out-of-distribution inputs  
- Adapt to new classes without catastrophic forgetting  

OSR and CIL address different aspects of this challenge but are often studied independently. **Open Horizons** treats them as interconnected components of open-world learning, asking how representation quality and learning dynamics affect both unknown detection and incremental adaptation.

---

## Open-Set Recognition Perspective

In the OSR component, the project evaluates how different pretrained visual backbones behave when exposed to unknown classes at test time. Three frozen encoders—**ResNet-50**, **ConvNeXt-Tiny**, and **CLIP ViT-B/16**—are paired with a shared linear classification head and standard post-hoc scoring functions (MSP, Energy, Mahalanobis, and kNN).

The analysis focuses on:

- Separability of known vs. unknown samples  
- Robustness of confidence scores  
- Operating-point behavior under strict OOD rejection constraints  

Metrics such as AUROC, AUPR, FPR@95, and OSCR are used to quantify performance, with additional emphasis on decision behavior when a fixed proportion of unknown samples must be rejected. This setup highlights the impact of representation geometry and calibration rather than task-specific fine-tuning.

---

## Class-Incremental Learning Perspective

Complementing OSR, the FSCIL component studies how models incorporate new classes with limited supervision while preserving previously learned knowledge. Using a session-based CIFAR-10 protocol, the project compares **SPPR**, **OrCo**, and **ConCM** under identical conditions.

This analysis emphasizes:

- Stability–plasticity trade-offs  
- Prototype evolution across incremental sessions  
- Sensitivity to the number of shots per novel class  

By examining confusion patterns and overall accuracy across incremental sessions, the project sheds light on how different prototype-based strategies mitigate catastrophic forgetting and maintain class separability over time.

---

## Key Observations

Across both OSR and FSCIL experiments, several consistent themes emerge:

- **Representation quality dominates performance**: Transformer-based CLIP embeddings provide substantially better separability for unknown detection and stronger foundations for proximity-based scoring.
- **Post-hoc scoring remains fragile**: OSR performance is sensitive to calibration and threshold selection, particularly at high OOD rejection operating points.
- **Incremental learning benefits from structured prototypes**: Methods that explicitly enforce separation and consistency (e.g., OrCo and ConCM) achieve superior stability and final accuracy.
- **OSR and CIL are tightly coupled**: Embeddings that support strong unknown detection also facilitate cleaner incremental updates, suggesting shared underlying requirements.

---

## Broader Research Direction and Future Work

Open Horizons is intended as a stepping stone toward **unified open-world learning systems**. Future research directions motivated by this work include:

- Joint frameworks that integrate OSR and CIL, enabling simultaneous unknown rejection and incremental class acquisition  
- Evaluation on larger-scale and domain-shifted benchmarks to stress-test representation robustness  
- Exploration of OSR-aware training objectives, adaptive thresholding, and prototype calibration strategies  
- Extending incremental learning beyond class addition to handle concept drift and evolving class semantics  

Ultimately, the goal is to move beyond isolated evaluations and toward systems that can **recognize, reject, and learn in the wild**.

---

## Project Philosophy

This repository prioritizes **clarity, reproducibility, and analysis** over black-box performance gains. It is designed for researchers interested in understanding open-world behavior rather than optimizing for a single benchmark. Code organization and execution details are intentionally separated from this high-level overview and documented within relevant subdirectories.
