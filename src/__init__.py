# Hybrid Approach: DINOv2 + Knowledge Distillation for Biomass Prediction
# CSIRO Biomass Competition

from .config import CFG
from .dataset import BiomassDataset, create_folds
from .models import TeacherModel, StudentModel
from .losses import DistillationLoss
from .trainer import train_teacher, train_student, generate_soft_targets

__all__ = [
    "CFG",
    "BiomassDataset",
    "create_folds",
    "TeacherModel",
    "StudentModel",
    "DistillationLoss",
    "train_teacher",
    "train_student",
    "generate_soft_targets",
]
