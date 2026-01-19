"""
Configuration for Hybrid Approach: DINOv2 + Knowledge Distillation
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import torch


@dataclass
class CFG:
    """Hyperparameters and paths for training."""

    # ===== Paths =====
    # Kaggle paths
    DATA_PATH: Path = Path("/kaggle/input/csiro-biomass")
    OUTPUT_DIR: Path = Path("/kaggle/working")
    WEIGHTS_PATH: Path = Path("/kaggle/input/pretrained-weights-biomass")
    
    # Local development paths (override in runtime)
    LOCAL_DATA_PATH: Path = Path("data")
    LOCAL_WEIGHTS_PATH: Path = Path.home() / "kaggle_weights" / "dinov2"

    # ===== Model =====
    backbone: str = "dinov2_vitb14"
    backbone_dim: int = 768  # ViT-B/14 output dimension
    input_size: int = 518  # DINOv2 optimal (divisible by 14)
    freeze_backbone: bool = True

    # Tabular encoder
    n_states: int = 4  # NSW, Vic, Tas, WA
    n_species: int = 30  # 25+ species in dataset
    embed_dim: int = 16
    tabular_hidden_dim: int = 128

    # ===== Teacher =====
    teacher_epochs: int = 20
    teacher_lr: float = 2e-4
    use_auxiliary_tasks: bool = True
    aux_weight: float = 0.2

    # ===== Student =====
    student_epochs: int = 25
    student_lr: float = 1e-4
    kd_alpha: float = 0.5  # Balance: 0=soft only, 1=hard only
    kd_temperature: float = 2.0

    # ===== Zero-Inflated Clover =====
    use_zero_inflated_clover: bool = True
    clover_cls_weight: float = 0.5

    # ===== Training =====
    n_folds: int = 5
    batch_size: int = 16
    weight_decay: float = 1e-4
    num_workers: int = 4

    # ===== Augmentation =====
    use_mixup: bool = True
    mixup_alpha: float = 0.2

    # ===== TTA =====
    use_tta: bool = True
    tta_transforms: int = 4  # original + hflip + vflip + both

    # ===== Misc =====
    seed: int = 42
    debug: bool = False

    # ===== Targets =====
    target_cols: List[str] = field(default_factory=lambda: [
        "Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"
    ])
    independent_targets: List[str] = field(default_factory=lambda: [
        "Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"  # Only predict these 3
    ])
    
    # Competition weights
    target_weights: Dict[str, float] = field(default_factory=lambda: {
        "Dry_Green_g": 0.1,
        "Dry_Dead_g": 0.1,
        "Dry_Clover_g": 0.1,
        "GDM_g": 0.2,
        "Dry_Total_g": 0.5,
    })

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def get_data_path(self) -> Path:
        """Return appropriate data path based on environment."""
        if self.DATA_PATH.exists():
            return self.DATA_PATH
        return self.LOCAL_DATA_PATH

    def get_weights_path(self) -> Path:
        """Return appropriate weights path based on environment."""
        if self.WEIGHTS_PATH.exists():
            return self.WEIGHTS_PATH
        return self.LOCAL_WEIGHTS_PATH

    def __post_init__(self):
        if self.debug:
            self.teacher_epochs = 2
            self.student_epochs = 2
            self.n_folds = 2
