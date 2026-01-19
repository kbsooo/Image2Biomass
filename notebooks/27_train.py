#%% [markdown]
# # 🚀 v27: Breakthrough Strategy
#
# **Goal**: Break LB 0.69 barrier
#
# **Key Changes from v20**:
# 1. **Frozen Backbone**: v22 showed smaller CV-LB gap with frozen backbone
# 2. **Consistency Regularization**: Force augmentation-invariant predictions
# 3. **LOSO CV**: Leave-One-State-Out for honest evaluation
# 4. **Heavy Dropout (0.4)**: Prevent location memorization
# 5. **No Aux Tasks**: Test doesn't have Height/NDVI
# 6. **3-Seed Ensemble**: Variance reduction
#
# **Expected**: CV ~0.65 (honest), LB 0.72+ (gap < 0.05)

#%%
import os
import gc
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

import timm
from torchvision import transforms as T

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

#%% [markdown]
# ## 📊 WandB Setup

#%%
import wandb

wandb.login()

WANDB_ENTITY = "kbsoo0620-"
WANDB_PROJECT = "csiro"

print(f"✓ WandB: {WANDB_ENTITY}/{WANDB_PROJECT}")

#%% [markdown]
# ## 🔐 Setup

#%%
GDRIVE_SAVE_PATH = None

try:
    from google.colab import drive
    drive.mount('/content/drive')
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_v27')
    GDRIVE_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"✓ Drive: {GDRIVE_SAVE_PATH}")
except ImportError:
    print("Not in Colab")

#%%
import kagglehub

IS_KAGGLE = Path("/kaggle/input/csiro-biomass").exists()
if not IS_KAGGLE:
    kagglehub.login()

#%%
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def flush():
    gc.collect()
    torch.cuda.empty_cache()

#%% [markdown]
# ## ⚙️ Configuration

#%%
class CFG:
    """v27 Configuration - Optimized for generalization"""
    
    # Data
    img_size: Tuple[int, int] = (512, 512)
    
    # Model - Simpler than v20 (proven by v22)
    hidden_dim: int = 256      # v20: 512
    num_layers: int = 2        # v20: 3
    dropout: float = 0.4       # v20: 0.1 - Heavy regularization
    use_layernorm: bool = True
    freeze_backbone: bool = True  # KEY: Frozen backbone
    
    # Training
    lr: float = 5e-4           # Higher LR for frozen backbone (only heads train)
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    
    batch_size: int = 8
    epochs: int = 35           # Longer training for frozen backbone
    patience: int = 10
    
    # v27 KEY: Consistency Regularization
    use_consistency: bool = True
    consistency_weight: float = 0.1
    
    # Augmentation
    hue_jitter: float = 0.02   # Conservative (v20 proven)
    
    # Multi-seed ensemble
    seeds: List[int] = [42, 123, 456]

cfg = CFG()

#%%
# Data paths
if IS_KAGGLE:
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    csiro_path = kagglehub.competition_download('csiro-biomass')
    weights_path = kagglehub.dataset_download('kbsooo/pretrained-weights-biomass')
    DATA_PATH = Path(csiro_path)
    WEIGHTS_PATH = Path(weights_path) / "dinov3_large" / "dinov3_large"
    OUTPUT_DIR = Path("/content/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Data: {DATA_PATH}")
print(f"Weights: {WEIGHTS_PATH}")

#%% [markdown]
# ## 📊 Data Loading

#%%
TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5  # Dominant target
}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted R² metric.
    Target order: [Green, Dead, Clover, GDM, Total]
    """
    weighted_r2 = 0.0
    for i, target in enumerate(TARGET_ORDER):
        weight = TARGET_WEIGHTS[target]
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        weighted_r2 += weight * r2
    return weighted_r2

#%%
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long format to wide format (one row per image)."""
    pivot = df.pivot_table(
        index=['image_path', 'State', 'Species', 'Sampling_Date', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()
    pivot.columns.name = None
    return pivot

train_df = pd.read_csv(DATA_PATH / "train.csv")
train_wide = prepare_data(train_df)
train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)

print(f"Train samples: {len(train_wide)}")
print(f"\nState distribution:\n{train_wide['State'].value_counts()}")

#%% [markdown]
# ## 🎯 v27 KEY: Leave-One-State-Out CV
#
# Standard StratifiedGroupKFold gives CV 0.79 but LB 0.69 (0.10 gap!)
# LOSO simulates test condition where location is completely unseen.

#%%
def create_loso_folds(df: pd.DataFrame) -> List[Dict]:
    """
    Leave-One-State-Out Cross-Validation.
    
    Each fold holds out one entire state as validation.
    This gives honest CV estimate closer to LB.
    
    States: NSW, Tas, Vic, WA
    """
    states = sorted(df['State'].unique())
    folds = []
    
    for holdout_state in states:
        train_data = df[df['State'] != holdout_state].reset_index(drop=True)
        val_data = df[df['State'] == holdout_state].reset_index(drop=True)
        
        folds.append({
            'holdout_state': holdout_state,
            'train': train_data,
            'val': val_data,
            'train_size': len(train_data),
            'val_size': len(val_data)
        })
        
        print(f"Fold {holdout_state}: train={len(train_data)}, val={len(val_data)}")
    
    return folds

print("\n=== LOSO Fold Distribution ===")
loso_folds = create_loso_folds(train_wide)

#%% [markdown]
# ## 🎨 Augmentation

#%%
def get_train_transforms(cfg: CFG) -> T.Compose:
    """
    Training augmentations.
    Conservative hue (0.02) to preserve chlorophyll colors.
    """
    return T.Compose([
        T.Resize(cfg.img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(15)], p=0.3),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=cfg.hue_jitter),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(cfg: CFG) -> T.Compose:
    """Validation transforms (no augmentation)."""
    return T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%% [markdown]
# ## 📦 Dataset

#%%
class BiomassDataset(Dataset):
    """
    v27 Dataset.
    
    Returns left/right image crops and main targets only.
    No auxiliary targets (Height/NDVI) - they don't exist at test time.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        data_path: Path,
        transform: T.Compose = None,
        return_raw: bool = False  # For consistency regularization
    ):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.transform = transform
        self.return_raw = return_raw
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_and_split_image(self, img_path: str) -> Tuple[Image.Image, Image.Image]:
        """Load image and split into left/right crops."""
        img = Image.open(self.data_path / img_path).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        left_img = img.crop((0, 0, mid, height))
        right_img = img.crop((mid, 0, width, height))
        
        return left_img, right_img
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        left_img, right_img = self._load_and_split_image(row['image_path'])
        
        if self.return_raw:
            # Return raw PIL images for consistency regularization
            return {
                'left_raw': left_img,
                'right_raw': right_img,
                'targets': torch.tensor([
                    row['Dry_Green_g'],
                    row['Dry_Clover_g'],
                    row['Dry_Dead_g']
                ], dtype=torch.float32)
            }
        
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        # Main targets: [Green, Clover, Dead]
        targets = torch.tensor([
            row['Dry_Green_g'],
            row['Dry_Clover_g'],
            row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        return {
            'left': left_img,
            'right': right_img,
            'targets': targets
        }

#%% [markdown]
# ## 🧠 Model

#%%
class CSIROModelV27(nn.Module):
    """
    v27 Model: Frozen Backbone + Simple Heads
    
    Key design choices:
    1. Frozen DINOv3 backbone - prevents location memorization
    2. Simple head (256 dim, 2 layers) - less overfitting
    3. Heavy dropout (0.4) - regularization
    4. No FiLM fusion - simpler is better for generalization
    5. Physics constraints: GDM = Green + Clover, Total = GDM + Dead
    """
    
    def __init__(self, cfg: CFG, weights_path: Path = None):
        super().__init__()
        self.cfg = cfg
        
        # DINOv3 Large backbone
        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3_qkvb.lvd1689m",
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
        
        # Load pretrained weights
        if weights_path:
            weights_file = weights_path / "dinov3_vitl16_qkvb.pth"
            if weights_file.exists():
                state = torch.load(weights_file, map_location='cpu', weights_only=True)
                self.backbone.load_state_dict(state, strict=False)
                print(f"✓ Loaded backbone weights from {weights_file}")
        
        # Freeze backbone (v27 KEY)
        if cfg.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen")
        
        feat_dim = self.backbone.num_features  # 1024 for ViT-L
        combined_dim = feat_dim * 2  # Left + Right concatenation
        
        # Simple prediction heads
        self.head_green = self._make_head(combined_dim)
        self.head_clover = self._make_head(combined_dim)
        self.head_dead = self._make_head(combined_dim)
        
        self.softplus = nn.Softplus(beta=1.0)
    
    def _make_head(self, in_dim: int) -> nn.Sequential:
        """Create simple prediction head."""
        layers = []
        current_dim = in_dim
        
        for i in range(self.cfg.num_layers):
            layers.append(nn.Linear(current_dim, self.cfg.hidden_dim))
            if i < self.cfg.num_layers - 1:
                if self.cfg.use_layernorm:
                    layers.append(nn.LayerNorm(self.cfg.hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(self.cfg.dropout))
            current_dim = self.cfg.hidden_dim
        
        layers.append(nn.Linear(self.cfg.hidden_dim, 1))
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            left_img: [B, 3, H, W]
            right_img: [B, 3, H, W]
            
        Returns:
            main_output: [B, 5] - [Green, Dead, Clover, GDM, Total]
            independent: [B, 3] - [Green, Clover, Dead]
        """
        # Extract features
        left_feat = self.backbone(left_img)   # [B, 1024]
        right_feat = self.backbone(right_img)  # [B, 1024]
        
        # Simple concatenation (no FiLM)
        combined = torch.cat([left_feat, right_feat], dim=1)  # [B, 2048]
        
        # Independent predictions (non-negative via softplus)
        green = self.softplus(self.head_green(combined))    # [B, 1]
        clover = self.softplus(self.head_clover(combined))  # [B, 1]
        dead = self.softplus(self.head_dead(combined))      # [B, 1]
        
        # Physics constraints
        gdm = green + clover
        total = gdm + dead
        
        # Output in competition order: [Green, Dead, Clover, GDM, Total]
        main_output = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        # Independent targets: [Green, Clover, Dead]
        independent = torch.cat([green, clover, dead], dim=1)
        
        return main_output, independent
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Get only trainable parameters (heads only if backbone frozen)."""
        params = []
        params.extend(self.head_green.parameters())
        params.extend(self.head_clover.parameters())
        params.extend(self.head_dead.parameters())
        return params

#%% [markdown]
# ## 🏋️ Training with Consistency Regularization

#%%
def train_fold(
    fold_data: Dict,
    cfg: CFG,
    seed: int,
    device: str = "cuda"
) -> Tuple[float, Dict]:
    """
    Train one LOSO fold.
    
    v27 KEY: Consistency Regularization
    - Same image with different augmentations should give same prediction
    - Uses dropout as stochastic perturbation for efficiency
    """
    holdout_state = fold_data['holdout_state']
    train_data = fold_data['train']
    val_data = fold_data['val']
    
    print(f"\n{'='*50}")
    print(f"Fold: Holdout {holdout_state} | Seed {seed}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"{'='*50}")
    
    # Datasets
    train_ds = BiomassDataset(train_data, DATA_PATH, get_train_transforms(cfg))
    val_ds = BiomassDataset(val_data, DATA_PATH, get_val_transforms(cfg))
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = CSIROModelV27(cfg, WEIGHTS_PATH).to(device)
    
    # Optimizer (only head parameters)
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler()
    
    best_score = -float('inf')
    best_state = None
    no_improve = 0
    history = {'train_loss': [], 'val_score': []}
    
    for epoch in range(cfg.epochs):
        # === Training ===
        model.train()
        train_loss = 0.0
        train_supervised = 0.0
        train_consistency = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                # Forward pass 1
                _, pred1 = model(left, right)
                
                # Supervised loss
                supervised_loss = F.mse_loss(pred1, targets)
                
                # Consistency loss (v27 KEY)
                # Use dropout as stochastic perturbation
                consistency_loss = torch.tensor(0.0, device=device)
                if cfg.use_consistency:
                    _, pred2 = model(left, right)  # Different dropout mask
                    consistency_loss = F.mse_loss(pred1.detach(), pred2)
                
                loss = supervised_loss + cfg.consistency_weight * consistency_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            train_supervised += supervised_loss.item()
            train_consistency += consistency_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'sup': f'{supervised_loss.item():.4f}',
                'con': f'{consistency_loss.item():.4f}'
            })
        
        n_batches = len(train_loader)
        train_loss /= n_batches
        train_supervised /= n_batches
        train_consistency /= n_batches
        
        # === Validation ===
        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                left = batch['left'].to(device)
                right = batch['right'].to(device)
                batch_targets = batch['targets']
                
                main_output, _ = model(left, right)
                all_preds.append(main_output.cpu().numpy())
                all_targets.append(batch_targets.numpy())
        
        preds = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)
        
        # Expand to 5 targets for metric
        # targets_np: [Green, Clover, Dead]
        # full_targets: [Green, Dead, Clover, GDM, Total]
        full_targets = np.zeros((len(targets_np), 5))
        full_targets[:, 0] = targets_np[:, 0]  # Green
        full_targets[:, 1] = targets_np[:, 2]  # Dead
        full_targets[:, 2] = targets_np[:, 1]  # Clover
        full_targets[:, 3] = targets_np[:, 0] + targets_np[:, 1]  # GDM
        full_targets[:, 4] = full_targets[:, 3] + targets_np[:, 2]  # Total
        
        val_score = competition_metric(full_targets, preds)
        
        # Logging
        history['train_loss'].append(train_loss)
        history['val_score'].append(val_score)
        
        wandb.log({
            f"seed{seed}/{holdout_state}/train_loss": train_loss,
            f"seed{seed}/{holdout_state}/supervised": train_supervised,
            f"seed{seed}/{holdout_state}/consistency": train_consistency,
            f"seed{seed}/{holdout_state}/val_score": val_score,
            f"seed{seed}/{holdout_state}/epoch": epoch + 1,
        })
        
        print(f"  Epoch {epoch+1}: loss={train_loss:.4f} "
              f"(sup={train_supervised:.4f}, con={train_consistency:.4f}), "
              f"CV={val_score:.4f}")
        
        # Early stopping
        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  ✓ New best!")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Save best model
    save_name = f"model_seed{seed}_{holdout_state}.pth"
    torch.save(best_state, OUTPUT_DIR / save_name)
    print(f"✓ Saved: {save_name} (CV={best_score:.4f})")
    
    flush()
    
    return best_score, history

#%% [markdown]
# ## 🚀 Run Training

#%%
run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name=f"v27_frozen_consistency_loso",
    config={
        "version": "v27",
        "freeze_backbone": cfg.freeze_backbone,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "consistency_weight": cfg.consistency_weight,
        "lr": cfg.lr,
        "seeds": cfg.seeds,
        "cv_strategy": "LOSO",
    }
)

#%%
print("\n" + "="*60)
print("🚀 v27 Training: Frozen + Consistency + LOSO")
print("="*60)
print(f"Backbone: Frozen={cfg.freeze_backbone}")
print(f"Head: {cfg.hidden_dim} dim, {cfg.num_layers} layers, dropout={cfg.dropout}")
print(f"Consistency weight: {cfg.consistency_weight}")
print(f"Seeds: {cfg.seeds}")
print(f"CV Strategy: Leave-One-State-Out")

#%%
all_results = {}

for seed in cfg.seeds:
    print(f"\n{'#'*60}")
    print(f"# SEED {seed}")
    print(f"{'#'*60}")
    
    seed_everything(seed)
    
    seed_scores = []
    
    for fold_data in loso_folds:
        score, history = train_fold(fold_data, cfg, seed)
        seed_scores.append({
            'state': fold_data['holdout_state'],
            'score': score
        })
    
    mean_cv = np.mean([s['score'] for s in seed_scores])
    std_cv = np.std([s['score'] for s in seed_scores])
    
    all_results[seed] = {
        'folds': seed_scores,
        'mean_cv': mean_cv,
        'std_cv': std_cv
    }
    
    print(f"\nSeed {seed} Summary:")
    for s in seed_scores:
        print(f"  {s['state']}: {s['score']:.4f}")
    print(f"  Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")
    
    wandb.log({
        f"seed{seed}/mean_cv": mean_cv,
        f"seed{seed}/std_cv": std_cv,
    })

#%% [markdown]
# ## 📊 Final Results

#%%
print("\n" + "="*60)
print("🎉 v27 FINAL RESULTS")
print("="*60)

# Aggregate across seeds
all_mean_cvs = [r['mean_cv'] for r in all_results.values()]
final_mean_cv = np.mean(all_mean_cvs)
final_std_cv = np.std(all_mean_cvs)

print(f"\nPer-seed results:")
for seed, result in all_results.items():
    print(f"  Seed {seed}: CV={result['mean_cv']:.4f} ± {result['std_cv']:.4f}")

print(f"\nFinal CV (across {len(cfg.seeds)} seeds): {final_mean_cv:.4f} ± {final_std_cv:.4f}")
print(f"\nNote: This is LOSO CV (honest estimate). Expected LB ≈ CV ± 0.03")

wandb.log({
    "final/mean_cv": final_mean_cv,
    "final/std_cv": final_std_cv,
    "final/n_models": len(cfg.seeds) * len(loso_folds),
})

#%%
# Save to Google Drive
if GDRIVE_SAVE_PATH:
    # Copy all models
    for f in OUTPUT_DIR.glob("model_seed*.pth"):
        shutil.copy(f, GDRIVE_SAVE_PATH / f.name)
    
    # Save results
    results_summary = {
        'version': 'v27',
        'final_mean_cv': float(final_mean_cv),
        'final_std_cv': float(final_std_cv),
        'config': {
            'freeze_backbone': cfg.freeze_backbone,
            'hidden_dim': cfg.hidden_dim,
            'num_layers': cfg.num_layers,
            'dropout': cfg.dropout,
            'consistency_weight': cfg.consistency_weight,
            'lr': cfg.lr,
            'seeds': cfg.seeds,
        },
        'all_results': {
            str(k): {
                'mean_cv': float(v['mean_cv']),
                'std_cv': float(v['std_cv']),
                'folds': [{kk: float(vv) if isinstance(vv, float) else vv 
                          for kk, vv in f.items()} for f in v['folds']]
            }
            for k, v in all_results.items()
        }
    }
    
    with open(GDRIVE_SAVE_PATH / 'results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✓ Saved {len(list(GDRIVE_SAVE_PATH.glob('*.pth')))} models to: {GDRIVE_SAVE_PATH}")

wandb.finish()

print("\n" + "="*60)
print("✅ Training Complete!")
print(f"Models: {len(cfg.seeds)} seeds × {len(loso_folds)} states = {len(cfg.seeds) * len(loso_folds)} models")
print(f"Expected LB: {final_mean_cv:.4f} (± 0.03)")
print("="*60)
