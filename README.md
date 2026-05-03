# DIPN — Domain-Invariant Prototype Network

**Domain-Invariant Prototype Network for Cross-Domain Breast Lesion Classification in Mammograms**

## Overview

DIPN is a two-phase unsupervised domain adaptation framework for breast lesion classification in mammography. It aligns target features to frozen source class prototypes without requiring target annotations or pseudo-labels.

**Full objective (Eq. 7):**

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{src} + \lambda_a \mathcal{L}_\text{align} + \gamma_d \mathcal{L}_\text{div}$$

with $\lambda_a = \gamma_d = 1.0$ fixed across all target domains.


## Data Layout

Each domain must follow this structure:

```
vindr_source/
├── cancer/      # labeled cancer images
└── healthy/     # labeled healthy images

<target>_target/
└── (flat folder of unlabeled images, any extension)

<target>_target_eval/
├── cancer/
└── healthy/
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

**Single run:**
```bash
python train.py \
    --config configs/dipn.yaml \
    --source_dir  /path/to/vindr_source \
    --target_dir  /path/to/inbreast_target \
    --target_eval_dir /path/to/inbreast_target_eval \
    --seed 0 \
    --save_subdir DIPN_VinDr_to_INbreast/run_0 \
    --device cuda:0
```

**Multi-seed run (3 seeds):**
```bash
TARGET_RAW="INbreast Dataset original" \
TARGET_KEY="inbreast" \
DEVICE=cuda:0 \
bash scripts/run.sh
```

## Repository Structure

```
DIPN/
├── train.py                  # main entry point
├── evaluate.py               # standalone evaluation
├── configs/
│   └── dipn.yaml             # full hyperparameter configuration
├── data/
│   └── dataset.py            # SourceDataset, TargetDataset
├── models/
│   └── backbone.py           # EfficientNet-B0 (+ timm alternatives)
├── training/
│   ├── phase1.py             # source consolidation
│   ├── phase2.py             # prototypical domain adaptation
│   └── prototypes.py         # prototype computation
├── losses/
│   ├── focal.py              # FocalLoss
│   ├── soft_align.py         # SoftPrototypeAlignmentLoss
│   ├── diversity.py          # DiversityLoss (KL to uniform)
│   └── mixup.py              # source Mixup utilities
├── utils/
│   ├── distribution.py       # Saerens-style label-shift corrector
│   ├── metrics.py            # AUC, sensitivity, specificity, F1
│   └── checkpoint.py         # save/load helpers
└── scripts/
    └── run.sh                # multi-seed launcher
```

## Key Design Choices

| Component | Paper Section | Details |
|---|---|---|
| Backbone | §4.3 | EfficientNet-B0, ImageNet pretrained, 1280-d features |
| Phase 1 | §3.2 | Focal loss (γ=2), source Mixup (p=0.5, α=0.4), early stopping on source val AUC |
| Prototypes | §3.2 Eq.1 | L2-normalized class means, frozen during Phase 2 |
| Phase 2 loss | §3.3.5 Eq.7 | L_src + λ_a·L_align + γ_d·L_div, λ_a=γ_d=1.0 |
| Alignment | §3.3.2 Eq.3 | Soft cosine attraction to frozen prototypes |
| Diversity | §3.3.3 Eq.4 | KL(uniform ‖ batch-mean prediction) |
| Correction | §3.3.4 Eq.6 | Saerens-style, gated by prior gap τ_π |
| Optimizer | §4.3 | Adam, weight_decay=1e-4, cosine-annealing LR |
| Image size | §4.3 | 224×224 (train: resize; eval: resize→center-crop) |

## Citation

If you use this code, please cite:

```bibtex
@article{hassan2025dipn,
  title={Domain-Invariant Prototype Network for Cross-Domain Breast Lesions Classification in Mammograms},
  author={Hassan, Mariam M. and others},
  year={2025}
}
```
