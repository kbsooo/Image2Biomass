"""
Training routines for Hybrid Approach.

- train_teacher: Phase 1 - Train teacher with full information
- generate_soft_targets: Phase 2 - Create soft targets from teacher
- train_student: Phase 3 - Train student with KD
"""
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import CFG
from .models import TeacherModel, StudentModel
from .losses import TeacherLoss, DistillationLoss, competition_metric


def train_teacher(
    model: TeacherModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CFG,
    fold: int,
    output_dir: Path,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """
    Phase 1: Train teacher model with full information.
    
    Returns:
        best_score: Best validation CV score
        soft_targets: Dict mapping image_id to predictions
    """
    device = cfg.device
    model = model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.teacher_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.teacher_epochs)
    loss_fn = TeacherLoss(cfg)
    scaler = GradScaler()
    
    best_score = -float('inf')
    best_state = None
    
    for epoch in range(cfg.teacher_epochs):
        # === Training ===
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Teacher Fold {fold} Epoch {epoch+1}/{cfg.teacher_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            continuous = batch['continuous'].to(device)
            state = batch['state'].to(device)
            species = batch['species'].to(device)
            month = batch['month'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(images, continuous, state, species, month)
                loss_dict = loss_fn(output, targets, continuous, state, species)
                loss = loss_dict['total']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        # === Validation ===
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                continuous = batch['continuous'].to(device)
                state = batch['state'].to(device)
                species = batch['species'].to(device)
                month = batch['month'].to(device)
                full_targets = batch['full_targets'].to(device)
                
                output = model(images, continuous, state, species, month)
                preds = output['main']['full']  # [Green, Dead, Clover, GDM, Total]
                
                all_preds.append(preds)
                all_targets.append(full_targets)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        cv_score = competition_metric(all_targets, all_preds).item()
        
        print(f"Epoch {epoch+1}/{cfg.teacher_epochs} | "
              f"Train Loss: {train_loss:.4f} | CV: {cv_score:.4f}")
        
        if cv_score > best_score:
            best_score = cv_score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  ✓ New best!")
    
    # Save best model
    save_path = output_dir / f"teacher_fold{fold}.pt"
    torch.save(best_state, save_path)
    print(f"✓ Saved teacher to {save_path}")
    
    # Load best for soft target generation
    model.load_state_dict(best_state)
    
    return best_score, model


def generate_soft_targets(
    model: TeacherModel,
    dataloader: DataLoader,
    cfg: CFG,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Phase 2: Generate soft targets from trained teacher.
    
    Returns:
        dict mapping image_id to:
        - 'pred': [5] tensor (Green, Dead, Clover, GDM, Total)
        - 'independent': [3] tensor (Green, Clover, Dead)
        - 'features': [feat_dim] tensor
    """
    device = cfg.device
    model = model.to(device)
    model.eval()
    
    soft_targets = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating soft targets"):
            images = batch['image'].to(device)
            continuous = batch['continuous'].to(device)
            state = batch['state'].to(device)
            species = batch['species'].to(device)
            month = batch['month'].to(device)
            image_ids = batch['image_id']
            
            output = model(images, continuous, state, species, month)
            
            for i, img_id in enumerate(image_ids):
                soft_targets[img_id] = {
                    'pred': output['main']['full'][i].cpu(),
                    'independent': output['main']['independent'][i].cpu(),
                    'features': output['features'][i].cpu(),
                }
    
    return soft_targets


def train_student(
    model: StudentModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    soft_targets: Dict[str, Dict[str, torch.Tensor]],
    cfg: CFG,
    fold: int,
    output_dir: Path,
) -> float:
    """
    Phase 3: Train student model with knowledge distillation.
    
    Returns:
        best_score: Best validation CV score
    """
    device = cfg.device
    model = model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.student_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.student_epochs)
    loss_fn = DistillationLoss(alpha=cfg.kd_alpha)
    scaler = GradScaler()
    
    best_score = -float('inf')
    best_state = None
    
    for epoch in range(cfg.student_epochs):
        # === Training ===
        model.train()
        train_loss = 0.0
        train_hard = 0.0
        train_soft = 0.0
        
        pbar = tqdm(train_loader, desc=f"Student Fold {fold} Epoch {epoch+1}/{cfg.student_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['targets'].to(device)
            image_ids = batch['image_id']
            
            # Get teacher's soft targets
            teacher_preds = torch.stack([
                soft_targets[img_id]['independent'] for img_id in image_ids
            ]).to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                student_output = model(images)
                
                # Create teacher output dict for loss
                teacher_output = {'independent': teacher_preds}
                
                loss_dict = loss_fn(student_output, teacher_output, targets)
                loss = loss_dict['total']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_hard += loss_dict['hard'].item()
            train_soft += loss_dict['soft'].item()
            pbar.set_postfix({
                'loss': loss.item(),
                'hard': loss_dict['hard'].item(),
                'soft': loss_dict['soft'].item(),
            })
        
        scheduler.step()
        n_batches = len(train_loader)
        train_loss /= n_batches
        train_hard /= n_batches
        train_soft /= n_batches
        
        # === Validation ===
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                full_targets = batch['full_targets'].to(device)
                
                output = model(images)
                preds = output['full']
                
                all_preds.append(preds)
                all_targets.append(full_targets)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        cv_score = competition_metric(all_targets, all_preds).item()
        
        print(f"Epoch {epoch+1}/{cfg.student_epochs} | "
              f"Loss: {train_loss:.4f} (H:{train_hard:.4f} S:{train_soft:.4f}) | "
              f"CV: {cv_score:.4f}")
        
        if cv_score > best_score:
            best_score = cv_score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  ✓ New best!")
    
    # Save best model
    save_path = output_dir / f"student_fold{fold}.pt"
    torch.save(best_state, save_path)
    print(f"✓ Saved student to {save_path}")
    
    return best_score


def inference(
    models: List[StudentModel],
    dataloader: DataLoader,
    cfg: CFG,
    use_tta: bool = True,
) -> torch.Tensor:
    """
    Run ensemble inference with optional TTA.
    
    Returns:
        predictions: [N, 5] tensor (Green, Dead, Clover, GDM, Total)
    """
    from .dataset import get_tta_transforms
    
    device = cfg.device
    all_preds = []
    
    for model in models:
        model = model.to(device)
        model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images = batch['image'].to(device)
            batch_preds = []
            
            for model in models:
                if use_tta:
                    # Simple TTA: original + hflip
                    pred1 = model(images)['full']
                    pred2 = model(torch.flip(images, dims=[3]))['full']  # HFlip
                    pred = (pred1 + pred2) / 2
                else:
                    pred = model(images)['full']
                
                batch_preds.append(pred)
            
            # Average across models
            ensemble_pred = torch.stack(batch_preds).mean(dim=0)
            all_preds.append(ensemble_pred.cpu())
    
    return torch.cat(all_preds, dim=0)
