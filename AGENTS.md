# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-18
**Commit:** 36d77f9
**Branch:** main

## OVERVIEW
Kaggle competition: Predicting pasture biomass (dry weight) from top-view images using PyTorch + DINOv3 + timm. Multi-output regression with physics constraints. Deadline: 2026-01-21.

## STRUCTURE
```
Image2Biomass/
├── src/                # Core ML modules (models, dataset, losses, trainer)
├── notebooks/          # Numbered experiments (01-26+), train/infer pairs
├── data/               # Train images (357), test images (gitignored)
├── docs/               # Strategy docs, figures
└── CLAUDE.md           # Competition context & conventions
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Model architecture | `src/models.py` | DINOv2 backbone, FiLM fusion, physics head |
| Dataset/transforms | `src/dataset.py` | albumentations, tabular encoding |
| Loss functions | `src/losses.py` | Zero-inflated, distillation, competition metric |
| Training loop | `src/trainer.py` | Teacher-student KD pipeline |
| Latest experiment | `notebooks/26_*.py` | OOF + calibration |
| Baseline reference | `notebooks/20_train.py` | Best performing baseline |
| Hyperparameter tuning | `notebooks/16_*.py`, `17_*.py` | Optuna integration |

## CONVENTIONS
- `#%%` cell markers for Jupytext compatibility
- Shape assertions: `assert x.shape == (B, C, H, W)`
- Type hints for function signatures
- Experiment naming: `NN_description.py` (train) / `NN_description_infer.py`
- Physics constraints: `GDM = Green + Clover`, `Total = GDM + Dead`

## ANTI-PATTERNS (THIS PROJECT)
- NEVER suppress type errors with `as any`
- NEVER predict GDM/Total directly (derive from independent targets)
- NEVER use random split (use StratifiedGroupKFold by State+Month)
- NEVER apply aggressive color augmentation (hue_jitter ≤ 0.02)

## TARGET ORDER & WEIGHTS
```python
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
TARGET_WEIGHTS = {'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1, 'GDM_g': 0.2, 'Dry_Total_g': 0.5}
# Dry_Total_g dominates metric (50% weight)
```

## KEY MODEL PATTERNS
```python
# 1. Independent prediction → derived targets
green = softplus(head_green(feat))
clover = softplus(head_clover(feat))
dead = softplus(head_dead(feat))
gdm = green + clover   # Physics constraint
total = gdm + dead     # Physics constraint

# 2. FiLM modulation (tabular → image)
gamma, beta = film(tabular_feat)
modulated = img_feat * (1 + gamma) + beta

# 3. Zero-inflated clover (37.8% zeros)
p_positive = sigmoid(classifier(feat))
amount = softplus(regressor(feat))
prediction = p_positive * amount
```

## COMMANDS
```bash
# Train (Colab/Kaggle)
python notebooks/26_train_oof.py

# Inference
python notebooks/26_infer_calibrated.py

# Local test
python test_environment.py
```

## NOTES
- Images are 70cm × 30cm quadrats (split left/right for dual-view)
- DINOv3 weights via `kagglehub.dataset_download('kbsooo/pretrained-weights-biomass')`
- WandB project: `kbsoo0620-/csiro`
- Best CV: ~0.65 (v20 baseline), target LB: >0.70
