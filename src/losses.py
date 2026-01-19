"""
Loss functions for Hybrid Approach.

- DistillationLoss: Combined hard and soft loss for KD
- ZeroInflatedLoss: BCE + MSE for clover
- competition_metric: Weighted R² calculation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .config import CFG


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate R² score."""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def competition_metric(
    y_true: torch.Tensor,  # [B, 5]
    y_pred: torch.Tensor,  # [B, 5]
    target_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Competition metric: Weighted R².
    
    Target order: [Green, Dead, Clover, GDM, Total]
    """
    if target_weights is None:
        # Default competition weights
        weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=y_pred.device)
    else:
        weights = torch.tensor([
            target_weights.get('Dry_Green_g', 0.1),
            target_weights.get('Dry_Dead_g', 0.1),
            target_weights.get('Dry_Clover_g', 0.1),
            target_weights.get('GDM_g', 0.2),
            target_weights.get('Dry_Total_g', 0.5),
        ], device=y_pred.device)
    
    weighted_r2 = 0.0
    for i in range(5):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        weighted_r2 += weights[i] * r2
    
    return weighted_r2


class ZeroInflatedLoss(nn.Module):
    """
    Combined loss for zero-inflated regression (Clover).
    
    L = α * BCE(p_positive, is_positive) + (1-α) * MSE(amount, target | target > 0)
    """
    
    def __init__(self, cls_weight: float = 0.5):
        super().__init__()
        self.cls_weight = cls_weight
    
    def forward(
        self,
        p_positive: torch.Tensor,   # [B, 1] - sigmoid output
        amount: torch.Tensor,        # [B, 1] - softplus output
        targets: torch.Tensor,       # [B, 1]
    ) -> torch.Tensor:
        """Returns scalar loss."""
        
        # Binary targets
        is_positive = (targets > 0).float()
        
        # Classification loss (BCE)
        cls_loss = F.binary_cross_entropy(p_positive, is_positive)
        
        # Regression loss (only on positive samples)
        positive_mask = targets > 0
        if positive_mask.sum() > 0:
            reg_loss = F.mse_loss(
                amount[positive_mask],
                targets[positive_mask]
            )
        else:
            reg_loss = torch.tensor(0.0, device=targets.device)
        
        return self.cls_weight * cls_loss + (1 - self.cls_weight) * reg_loss


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    
    L_total = α * L_hard + (1 - α) * L_soft + λ * L_clover
    
    Where:
    - L_hard: MSE with ground truth labels
    - L_soft: MSE with teacher's predictions (soft targets)
    - L_clover: Zero-inflated loss for clover
    """
    
    def __init__(
        self,
        alpha: float = 0.5,           # Hard vs soft balance
        clover_weight: float = 0.3,   # Extra weight for clover loss
        target_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.clover_weight = clover_weight
        self.zi_loss = ZeroInflatedLoss()
        self.target_weights = target_weights
    
    def forward(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,  # [B, 3] - Green, Clover, Dead
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            student_output: dict from StudentModel
            teacher_output: dict from TeacherModel (or cached targets)
            targets: Ground truth [Green, Clover, Dead]
            
        Returns:
            dict with 'total', 'hard', 'soft', 'clover' losses
        """
        # Student predictions [Green, Clover, Dead]
        pred = student_output['independent']
        
        # Hard loss (ground truth)
        hard_loss = F.mse_loss(pred, targets)
        
        # Soft loss (teacher's knowledge)
        if 'independent' in teacher_output:
            teacher_pred = teacher_output['independent']
            if not isinstance(teacher_pred, torch.Tensor):
                teacher_pred = teacher_pred.detach()
            else:
                teacher_pred = teacher_pred.detach()
            soft_loss = F.mse_loss(pred, teacher_pred)
        else:
            soft_loss = torch.tensor(0.0, device=pred.device)
        
        # Zero-inflated clover loss
        clover_loss = torch.tensor(0.0, device=pred.device)
        if 'clover_prob' in student_output and student_output['clover_prob'] is not None:
            clover_targets = targets[:, 1:2]  # Clover is at index 1
            clover_loss = self.zi_loss(
                student_output['clover_prob'],
                student_output['clover_amount'],
                clover_targets,
            )
        
        # Combine losses
        total_loss = (
            self.alpha * hard_loss +
            (1 - self.alpha) * soft_loss +
            self.clover_weight * clover_loss
        )
        
        return {
            'total': total_loss,
            'hard': hard_loss,
            'soft': soft_loss,
            'clover': clover_loss,
        }


class TeacherLoss(nn.Module):
    """
    Loss for training the teacher model.
    
    L = L_main + λ_aux * (L_ndvi + L_height + L_state + L_species)
    """
    
    def __init__(
        self,
        cfg: CFG,
        aux_weight: float = 0.2,
    ):
        super().__init__()
        self.cfg = cfg
        self.aux_weight = aux_weight
        self.zi_loss = ZeroInflatedLoss()
    
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        targets: torch.Tensor,       # [B, 3] - Green, Clover, Dead
        continuous: torch.Tensor,    # [B, 2] - NDVI, Height (for aux)
        state: torch.Tensor,         # [B] - for aux
        species: torch.Tensor,       # [B] - for aux
    ) -> Dict[str, torch.Tensor]:
        """Returns dict with total and component losses."""
        
        # Main loss on independent targets
        pred = output['main']['independent']  # [Green, Clover, Dead]
        main_loss = F.mse_loss(pred, targets)
        
        # Zero-inflated clover loss
        clover_loss = torch.tensor(0.0, device=pred.device)
        if 'clover_prob' in output['main']:
            clover_loss = self.zi_loss(
                output['main']['clover_prob'],
                output['main']['clover_amount'],
                targets[:, 1:2],
            )
        
        # Auxiliary losses
        aux_loss = torch.tensor(0.0, device=pred.device)
        if self.cfg.use_auxiliary_tasks and 'aux_ndvi' in output:
            # NDVI regression
            ndvi_loss = F.mse_loss(output['aux_ndvi'].squeeze(), continuous[:, 0])
            # Height regression
            height_loss = F.mse_loss(output['aux_height'].squeeze(), continuous[:, 1])
            # State classification
            state_loss = F.cross_entropy(output['aux_state'], state)
            # Species classification
            species_loss = F.cross_entropy(output['aux_species'], species)
            
            aux_loss = ndvi_loss + height_loss + state_loss + species_loss
        
        total_loss = main_loss + 0.3 * clover_loss + self.aux_weight * aux_loss
        
        return {
            'total': total_loss,
            'main': main_loss,
            'clover': clover_loss,
            'aux': aux_loss,
        }
