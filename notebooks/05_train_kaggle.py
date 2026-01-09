#%% [markdown]
# # CSIRO Image2Biomass - Kaggle Training Pipeline
#
# **Kaggle-only version** optimized for:
# - T4 x2 GPU with DataParallel
# - 9-hour session with checkpoint resume
# - Direct submission generation
#
# ### [SOTA Alert]
# DINOv2-Large backbone with bidirectional cross-attention fusion

#%% [markdown]
# ## Section 0: Setup

#%%
# !pip install -q timm albumentations

#%%
import os
import gc
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold

#%%
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

#%% [markdown]
# ## Section 1: Kaggle Configuration

#%%
@dataclass
class CFG:
    # === Kaggle Paths (Fixed) ===
    # Change 'csiro-pasture-biomass' to actual competition dataset name
    DATA_PATH: Path = Path("/kaggle/input/csiro-pasture-biomass")
    OUTPUT_DIR: Path = Path("/kaggle/working")

    # === Model ===
    backbone: str = "vit_large_patch14_dinov2.lvd142m"
    input_size: int = 518
    embed_dim: int = 1024  # DINOv2-large
    num_heads: int = 16
    dropout: float = 0.1

    # === Training ===
    n_folds: int = 5
    train_folds: List[int] = field(default_factory=lambda: [0])  # Train single fold for speed
    epochs: int = 15
    batch_size: int = 4  # Per GPU, effective = 8 with 2 GPUs
    gradient_accumulation: int = 2

    # === Optimizer ===
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 2

    # === Training phases ===
    freeze_backbone_epochs: int = 2

    # === Targets ===
    targets: List[str] = field(default_factory=lambda: [
        'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g'
    ])
    num_targets: int = 5

    # === Hardware ===
    seed: int = 42
    num_workers: int = 4  # Kaggle has good CPU
    mixed_precision: bool = True
    use_multi_gpu: bool = True  # Enable DataParallel for T4x2

    # === Resume ===
    resume_from: Optional[str] = None  # Path to checkpoint for resume

    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_gpus = torch.cuda.device_count()

        # Adjust batch size for multi-GPU
        if self.use_multi_gpu and self.n_gpus > 1:
            self.effective_batch_size = self.batch_size * self.n_gpus * self.gradient_accumulation
        else:
            self.effective_batch_size = self.batch_size * self.gradient_accumulation

cfg = CFG()
seed_everything(cfg.seed)

print(f"PyTorch: {torch.__version__}")
print(f"Device: {cfg.device}")
print(f"GPUs available: {cfg.n_gpus}")
print(f"Effective batch size: {cfg.effective_batch_size}")
print(f"Data path: {cfg.DATA_PATH}")
print(f"Output dir: {cfg.OUTPUT_DIR}")

#%% [markdown]
# ## Section 2: Constants & Metric

#%%
TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5,
}

TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

#%%
def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Globally weighted R² score."""
    weights = np.array([TARGET_WEIGHTS[t] for t in TARGET_ORDER])

    y_weighted_mean = sum(
        y_true[:, i].mean() * weights[i] for i in range(5)
    )

    ss_res = sum(
        ((y_true[:, i] - y_pred[:, i]) ** 2).mean() * weights[i]
        for i in range(5)
    )
    ss_tot = sum(
        ((y_true[:, i] - y_weighted_mean) ** 2).mean() * weights[i]
        for i in range(5)
    )

    return 1 - ss_res / (ss_tot + 1e-8)

#%% [markdown]
# ## Section 3: Data Preparation

#%%
def pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long format to wide format."""
    if 'target' in df.columns:
        df_pt = pd.pivot_table(
            df,
            values='target',
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
            columns='target_name',
            aggfunc='mean'
        ).reset_index()
    else:
        df['target'] = 0
        df_pt = pd.pivot_table(
            df, values='target', index='image_path',
            columns='target_name', aggfunc='mean'
        ).reset_index()
    return df_pt

def melt_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide format to submission format."""
    melted = df.melt(
        id_vars='image_path',
        value_vars=TARGET_ORDER,
        var_name='target_name',
        value_name='target'
    )
    melted['sample_id'] = (
        melted['image_path']
        .str.replace(r'^.*/', '', regex=True)
        .str.replace('.jpg', '', regex=False)
        + '__' + melted['target_name']
    )
    return melted[['sample_id', 'image_path', 'target_name', 'target']]

#%%
def prepare_data(cfg: CFG) -> pd.DataFrame:
    """Load and prepare training data with KFold."""
    train_df = pd.read_csv(cfg.DATA_PATH / "train.csv")
    train_wide = pivot_table(train_df)
    train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)

    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    train_wide['fold'] = -1
    for fold, (_, val_idx) in enumerate(kf.split(train_wide)):
        train_wide.loc[val_idx, 'fold'] = fold

    print(f"Data shape: {train_wide.shape}")
    print(f"Fold distribution:\n{train_wide['fold'].value_counts().sort_index()}")
    return train_wide

#%%
train_df = prepare_data(cfg)
train_df.head()

#%% [markdown]
# ## Section 4: Dataset & Transforms

#%%
def get_transforms(cfg: CFG, mode: str = 'train') -> A.Compose:
    if mode == 'train':
        return A.Compose([
            A.Resize(cfg.input_size, cfg.input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Insight: 색상 변환은 약하게 (녹색/갈색 구분 중요)
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
                               border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(cfg.input_size, cfg.input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

#%%
class BiomassDataset(Dataset):
    """Dataset with left/right image split."""
    def __init__(self, df: pd.DataFrame, cfg: CFG, transforms=None, mode: str = 'train'):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = self.cfg.DATA_PATH / row['image_path']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Split left/right (2000x1000 -> 1000x1000 x2)
        h, w = img.shape[:2]
        mid = w // 2
        left_img = img[:, :mid]
        right_img = img[:, mid:]

        if self.transforms:
            left_aug = self.transforms(image=left_img)['image']
            right_aug = self.transforms(image=right_img)['image']
        else:
            left_aug = torch.from_numpy(left_img).permute(2, 0, 1).float() / 255.0
            right_aug = torch.from_numpy(right_img).permute(2, 0, 1).float() / 255.0

        targets = torch.tensor([row[t] for t in TARGET_ORDER], dtype=torch.float32)

        return {'left': left_aug, 'right': right_aug, 'targets': targets, 'image_id': row['image_id']}

#%% [markdown]
# ## Section 5: Model Architecture

#%%
class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

#%%
class BiDirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention for left/right image fusion."""
    def __init__(self, dim: int = 1024, num_heads: int = 16,
                 num_fuse_tokens: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_fuse_tokens = num_fuse_tokens

        self.fuse_tokens = nn.Parameter(torch.randn(1, num_fuse_tokens, dim) * 0.02)

        self.cross_layers = nn.ModuleList([
            nn.ModuleDict({
                'l2r_attn': nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                'r2l_attn': nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                'l2r_norm': nn.LayerNorm(dim),
                'r2l_norm': nn.LayerNorm(dim),
                'ffn_l': FeedForward(dim, mlp_ratio=4.0, dropout=dropout),
                'ffn_r': FeedForward(dim, mlp_ratio=4.0, dropout=dropout),
                'ffn_norm_l': nn.LayerNorm(dim),
                'ffn_norm_r': nn.LayerNorm(dim),
            })
            for _ in range(num_layers)
        ])

        fusion_input_dim = dim * 2 + dim * num_fuse_tokens
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim * 2)
        )

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        B = feat_left.shape[0]

        feat_left = feat_left.unsqueeze(1)
        feat_right = feat_right.unsqueeze(1)
        fuse = self.fuse_tokens.expand(B, -1, -1)

        for layer in self.cross_layers:
            l_query = torch.cat([feat_left, fuse], dim=1)
            l2r_out, _ = layer['l2r_attn'](layer['l2r_norm'](l_query), feat_right, feat_right, need_weights=False)
            feat_left = feat_left + l2r_out[:, :1]
            fuse_l = l2r_out[:, 1:]

            r_query = torch.cat([feat_right, fuse], dim=1)
            r2l_out, _ = layer['r2l_attn'](layer['r2l_norm'](r_query), feat_left, feat_left, need_weights=False)
            feat_right = feat_right + r2l_out[:, :1]
            fuse_r = r2l_out[:, 1:]

            feat_left = feat_left + layer['ffn_l'](layer['ffn_norm_l'](feat_left))
            feat_right = feat_right + layer['ffn_r'](layer['ffn_norm_r'](feat_right))
            fuse = (fuse_l + fuse_r) / 2

        left_pool = feat_left.squeeze(1)
        right_pool = feat_right.squeeze(1)
        fuse_flat = fuse.flatten(1)

        combined = torch.cat([left_pool, right_pool, fuse_flat], dim=1)
        return self.fusion_proj(combined)

#%%
class BiomassModel(nn.Module):
    """DINOv2-Large + Bidirectional Cross-Attention + Physics-constrained outputs."""
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg

        # ### [SOTA Alert] DINOv2-large for rich visual representations
        self.backbone = timm.create_model(cfg.backbone, pretrained=True, num_classes=0)
        self.embed_dim = self.backbone.embed_dim

        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)

        self.cross_attn = BiDirectionalCrossAttention(
            dim=self.embed_dim, num_heads=cfg.num_heads,
            num_fuse_tokens=4, num_layers=2, dropout=cfg.dropout
        )

        head_dim = self.embed_dim * 2
        hidden_dim = head_dim // 2

        def make_head():
            return nn.Sequential(
                nn.Linear(head_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(hidden_dim, 1)
            )

        self.head_green = make_head()
        self.head_dead = make_head()
        self.head_clover = make_head()
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_left = self.backbone(left)
        feat_right = self.backbone(right)
        fused = self.cross_attn(feat_left, feat_right)

        green = self.softplus(self.head_green(fused))
        dead = self.softplus(self.head_dead(fused))
        clover = self.softplus(self.head_clover(fused))

        # Physics constraints: GDM = Green + Clover, Total = GDM + Dead
        gdm = green + clover
        total = gdm + dead

        all_preds = torch.cat([green, dead, clover, gdm, total], dim=1)
        return {'green': green, 'dead': dead, 'clover': clover, 'gdm': gdm, 'total': total, 'all': all_preds}

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

#%% [markdown]
# ## Section 6: Loss Function

#%%
class WeightedBiomassLoss(nn.Module):
    """Competition-aligned weighted loss with physics regularization."""
    def __init__(self, physics_weight: float = 0.1):
        super().__init__()
        self.register_buffer('weights', torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5]))
        self.physics_weight = physics_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_per_target = self.smooth_l1(pred, target)
        weighted_loss = (loss_per_target * self.weights.unsqueeze(0)).sum(dim=1).mean()

        # Physics constraint regularization
        pred_gdm_check = pred[:, 0] + pred[:, 2]
        pred_total_check = pred[:, 3] + pred[:, 1]
        physics_loss = (
            F.smooth_l1_loss(pred_gdm_check, pred[:, 3]) +
            F.smooth_l1_loss(pred_total_check, pred[:, 4])
        )

        return weighted_loss + self.physics_weight * physics_loss

#%% [markdown]
# ## Section 7: Training Functions

#%%
def train_one_epoch(model, loader, optimizer, scheduler, criterion, cfg, scaler=None, epoch=0):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f'Epoch {epoch+1} Train')

    for step, batch in enumerate(pbar):
        left = batch['left'].to(cfg.device, non_blocking=True)
        right = batch['right'].to(cfg.device, non_blocking=True)
        targets = batch['targets'].to(cfg.device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=cfg.mixed_precision):
            outputs = model(left, right)
            loss = criterion(outputs['all'], targets)
            loss = loss / cfg.gradient_accumulation

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % cfg.gradient_accumulation == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * cfg.gradient_accumulation
        pbar.set_postfix({'loss': f'{loss.item() * cfg.gradient_accumulation:.4f}'})

    return total_loss / len(loader)

#%%
@torch.no_grad()
def validate(model, loader, criterion, cfg):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc='Validation'):
        left = batch['left'].to(cfg.device, non_blocking=True)
        right = batch['right'].to(cfg.device, non_blocking=True)
        targets = batch['targets'].to(cfg.device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=cfg.mixed_precision):
            outputs = model(left, right)
            loss = criterion(outputs['all'], targets)

        total_loss += loss.item()
        all_preds.append(outputs['all'].cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    cv_score = competition_metric(all_targets, all_preds)

    r2_scores = {}
    for i, name in enumerate(TARGET_ORDER):
        ss_res = np.sum((all_targets[:, i] - all_preds[:, i]) ** 2)
        ss_tot = np.sum((all_targets[:, i] - all_targets[:, i].mean()) ** 2)
        r2_scores[name] = 1 - ss_res / (ss_tot + 1e-8)

    return total_loss / len(loader), cv_score, r2_scores, all_preds

#%% [markdown]
# ## Section 8: Post-Processing

#%%
def post_process_biomass(preds: np.ndarray) -> np.ndarray:
    """Project predictions to satisfy physics constraints."""
    C = np.array([
        [1, 0, 1, -1,  0],   # Green + Clover - GDM = 0
        [0, 1, 0,  1, -1]    # Dead + GDM - Total = 0
    ], dtype=np.float64)

    C_T = C.T
    inv_CCt = np.linalg.inv(C @ C_T)
    P = np.eye(5) - C_T @ inv_CCt @ C

    Y = preds.T
    Y_proj = P @ Y
    Y_proj = Y_proj.T
    Y_proj = np.clip(Y_proj, 0, None)

    return Y_proj

#%% [markdown]
# ## Section 9: Main Training Loop with Resume Support

#%%
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_score, fold, cfg):
    """Save checkpoint for resume."""
    # Handle DataParallel
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_score': best_score,
        'fold': fold,
        'config': {
            'backbone': cfg.backbone,
            'embed_dim': cfg.embed_dim,
            'num_heads': cfg.num_heads,
            'dropout': cfg.dropout,
        }
    }

    torch.save(checkpoint, cfg.OUTPUT_DIR / f'checkpoint_fold{fold}.pt')
    print(f"Checkpoint saved: epoch {epoch+1}")

#%%
def train_fold(fold: int, train_df: pd.DataFrame, cfg: CFG):
    """Train one fold with resume support."""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold} | GPUs: {cfg.n_gpus}")
    print(f"{'='*60}")

    # Split
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Datasets
    train_dataset = BiomassDataset(train_data, cfg, transforms=get_transforms(cfg, 'train'), mode='train')
    val_dataset = BiomassDataset(val_data, cfg, transforms=get_transforms(cfg, 'val'), mode='val')

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size * max(1, cfg.n_gpus),
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size * max(1, cfg.n_gpus) * 2,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )

    # Model
    model = BiomassModel(cfg).to(cfg.device)

    # Multi-GPU with DataParallel
    if cfg.use_multi_gpu and cfg.n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel with {cfg.n_gpus} GPUs")

    # Get base model for freeze/unfreeze
    base_model = model.module if hasattr(model, 'module') else model

    # Freeze backbone initially
    if cfg.freeze_backbone_epochs > 0:
        base_model.freeze_backbone()
        print(f"Backbone frozen for first {cfg.freeze_backbone_epochs} epochs")

    # Optimizer
    backbone_params = list(base_model.backbone.parameters())
    other_params = [p for n, p in base_model.named_parameters() if 'backbone' not in n]

    optimizer = AdamW([
        {'params': backbone_params, 'lr': cfg.backbone_lr},
        {'params': other_params, 'lr': cfg.lr}
    ], weight_decay=cfg.weight_decay)

    # Scheduler
    num_training_steps = len(train_loader) * cfg.epochs // cfg.gradient_accumulation
    scheduler = OneCycleLR(
        optimizer, max_lr=[cfg.backbone_lr * 10, cfg.lr],
        total_steps=num_training_steps, pct_start=cfg.warmup_epochs / cfg.epochs, anneal_strategy='cos'
    )

    # Loss & Scaler
    criterion = WeightedBiomassLoss(physics_weight=0.1).to(cfg.device)
    scaler = torch.amp.GradScaler('cuda') if cfg.mixed_precision else None

    # Resume from checkpoint
    start_epoch = 0
    best_score = -float('inf')

    checkpoint_path = cfg.OUTPUT_DIR / f'checkpoint_fold{fold}.pt'
    if cfg.resume_from or checkpoint_path.exists():
        ckpt_path = cfg.resume_from or checkpoint_path
        if Path(ckpt_path).exists():
            print(f"Resuming from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, weights_only=False)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if checkpoint['scaler_state_dict'] and scaler:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint['best_score']
            print(f"Resumed from epoch {start_epoch}, best_score: {best_score:.4f}")

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'cv_score': []}

    for epoch in range(start_epoch, cfg.epochs):
        # Unfreeze backbone
        if epoch == cfg.freeze_backbone_epochs and cfg.freeze_backbone_epochs > 0:
            base_model.unfreeze_backbone()
            print(f"\nBackbone unfrozen at epoch {epoch + 1}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, cfg, scaler, epoch)

        # Validate
        val_loss, cv_score, r2_scores, val_preds = validate(model, val_loader, criterion, cfg)

        # Post-process
        val_preds_pp = post_process_biomass(val_preds)
        val_targets = np.array([val_dataset[i]['targets'].numpy() for i in range(len(val_dataset))])
        cv_score_pp = competition_metric(val_targets, val_preds_pp)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['cv_score'].append(cv_score_pp)

        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"CV Score (raw): {cv_score:.4f} | CV Score (PP): {cv_score_pp:.4f}")
        for name, r2 in r2_scores.items():
            print(f"  {name} (w={TARGET_WEIGHTS[name]}): {r2:.4f}")

        # Save checkpoint every epoch (for 9h session limit)
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_score, fold, cfg)

        # Save best model
        if cv_score_pp > best_score:
            best_score = cv_score_pp
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'best_score': best_score,
                'r2_scores': r2_scores,
                'config': {
                    'backbone': cfg.backbone,
                    'embed_dim': cfg.embed_dim,
                    'num_heads': cfg.num_heads,
                    'dropout': cfg.dropout,
                }
            }, cfg.OUTPUT_DIR / f'best_model_fold{fold}.pt')
            print(f"*** New best model saved (CV: {best_score:.4f}) ***")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss Curve')

    axes[1].plot(history['cv_score'], label='CV Score', color='green')
    axes[1].axhline(y=best_score, color='r', linestyle='--', label=f'Best: {best_score:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Competition Metric')
    axes[1].legend()
    axes[1].set_title('CV Score (Post-processed)')

    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / f'training_history_fold{fold}.png', dpi=150)
    plt.show()

    flush()
    return best_score

#%%
# === Train ===
all_scores = []
for fold in cfg.train_folds:
    best_score = train_fold(fold, train_df, cfg)
    all_scores.append(best_score)

print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"Mean CV Score: {np.mean(all_scores):.4f} +/- {np.std(all_scores):.4f}")
print(f"Scores per fold: {all_scores}")
print(f"{'='*60}")

#%% [markdown]
# ## Section 10: Generate Submission

#%%
@torch.no_grad()
def inference_with_tta(model, loader, cfg, n_tta: int = 4) -> np.ndarray:
    model.eval()
    all_preds = []

    tta_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.flip(x, dims=[2, 3]),
    ][:n_tta]

    for batch in tqdm(loader, desc='Inference'):
        left = batch['left'].to(cfg.device)
        right = batch['right'].to(cfg.device)

        batch_preds = []
        for tta in tta_transforms:
            left_aug = tta(left)
            right_aug = tta(right)
            with torch.amp.autocast('cuda', enabled=cfg.mixed_precision):
                outputs = model(left_aug, right_aug)
                batch_preds.append(outputs['all'].cpu())

        batch_pred = torch.stack(batch_preds).mean(dim=0)
        all_preds.append(batch_pred.numpy())

    return np.concatenate(all_preds, axis=0)

#%%
def create_submission(cfg: CFG, fold: int = 0) -> pd.DataFrame:
    """Create submission from test data."""

    # Load model
    checkpoint = torch.load(cfg.OUTPUT_DIR / f'best_model_fold{fold}.pt', weights_only=False)
    model = BiomassModel(cfg).to(cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model with CV score: {checkpoint['best_score']:.4f}")

    if cfg.use_multi_gpu and cfg.n_gpus > 1:
        model = nn.DataParallel(model)

    # Test data
    test_df = pd.read_csv(cfg.DATA_PATH / 'test.csv')
    test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    test_wide['image_id'] = test_wide['image_path'].apply(lambda x: Path(x).stem)

    for t in TARGET_ORDER:
        if t not in test_wide.columns:
            test_wide[t] = 0.0

    test_dataset = BiomassDataset(test_wide, cfg, transforms=get_transforms(cfg, 'val'), mode='test')
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # Inference with TTA
    preds = inference_with_tta(model, test_loader, cfg, n_tta=4)
    preds = post_process_biomass(preds)

    # Create submission
    test_wide[TARGET_ORDER] = preds
    submission = melt_table(test_wide)

    return submission[['sample_id', 'target']]

#%%
# Generate submission if model exists
if (cfg.OUTPUT_DIR / f'best_model_fold{cfg.train_folds[0]}.pt').exists():
    submission = create_submission(cfg, fold=cfg.train_folds[0])
    submission.to_csv(cfg.OUTPUT_DIR / 'submission.csv', index=False)
    print(f"Submission saved: {len(submission)} rows")
    print(submission.head(10))
else:
    print("No trained model found.")

#%%
print(f"""
{'='*60}
Kaggle Training Complete!
{'='*60}

Output files in {cfg.OUTPUT_DIR}:
- best_model_fold*.pt     (best weights)
- checkpoint_fold*.pt     (resume checkpoint)
- training_history_*.png  (loss curves)
- submission.csv          (final submission)

To resume interrupted training:
  Set cfg.resume_from = '/kaggle/working/checkpoint_fold0.pt'
{'='*60}
""")
