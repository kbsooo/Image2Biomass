# src/ - Core ML Modules

## OVERVIEW
Reusable PyTorch components: models, dataset, losses, trainer. Import via `from src.models import ...`

## WHERE TO LOOK
| File | Purpose |
|------|---------|
| `config.py` | `CFG` dataclass with all hyperparameters |
| `models.py` | DINOv2Backbone, TabularEncoder, FiLMFusion, PhysicsHead, Teacher/Student models |
| `dataset.py` | BiomassDataset, transforms (train/val/TTA), fold creation |
| `losses.py` | ZeroInflatedLoss, DistillationLoss, TeacherLoss, competition_metric |
| `trainer.py` | train_teacher, generate_soft_targets, train_student, inference |

## KEY CLASSES
- `TeacherModel`: Image + tabular → predictions (training only)
- `StudentModel`: Image only → predictions (inference)
- `PhysicsConstrainedHead`: Predicts 3 independent targets → derives GDM, Total

## CONVENTIONS
- All predictions non-negative via `nn.Softplus()`
- Target order: `[Green, Dead, Clover, GDM, Total]` (competition format)
- Independent order: `[Green, Clover, Dead]` (internal)

## ANTI-PATTERNS
- NEVER import models directly in notebooks (notebooks self-contain for Kaggle)
- NEVER return 5-dim tensor for loss (use 3 independent targets)
