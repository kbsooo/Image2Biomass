#%% [markdown]
# # 🚀 v22: Backbone Freezing + Aggressive Regularization
#
# **핵심 변경**:
# 1. DINOv3 Backbone 완전 동결 (Feature Extractor로만 사용)
# 2. 강화된 정규화: dropout=0.3, weight_decay=1e-3
# 3. 더 단순한 Head: hidden_dim=256, num_layers=2
#
# **목표**: CV-LB gap 감소 (overfitting 방지)

#%%
import os
import gc
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

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
from sklearn.model_selection import StratifiedGroupKFold

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
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_v22')
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
def seed_everything(seed=42):
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

seed_everything(42)

#%% [markdown]
# ## ⚙️ Configuration

#%%
class CFG:
    img_size = (512, 512)
    
    # === v22 핵심 변경: 더 단순한 Head ===
    hidden_dim = 256   # 512 → 256
    num_layers = 2     # 3 → 2
    dropout = 0.3      # 0.1 → 0.3 (강화)
    use_layernorm = True
    
    # === Backbone 동결 ===
    freeze_backbone = True
    
    # === 강화된 정규화 ===
    lr = 1e-3          # Head만 학습하므로 더 높은 LR
    weight_decay = 1e-3  # 6e-5 → 1e-3 (강화)
    warmup_ratio = 0.1
    
    batch_size = 16    # Backbone 동결로 메모리 여유 → 배치 증가
    epochs = 30        # 더 많은 epoch
    patience = 10
    
    hue_jitter = 0.02
    
cfg = CFG()

print("=== v22 Configuration ===")
print(f"Backbone frozen: {cfg.freeze_backbone}")
print(f"Head: {cfg.hidden_dim}x{cfg.num_layers}, dropout={cfg.dropout}")
print(f"Regularization: lr={cfg.lr}, weight_decay={cfg.weight_decay}")

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

#%% [markdown]
# ## 📊 Data Loading

#%%
TARGET_WEIGHTS = {'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1, 'GDM_g': 0.2, 'Dry_Total_g': 0.5}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true, y_pred):
    weighted_r2 = 0.0
    for i, target in enumerate(TARGET_ORDER):
        weight = TARGET_WEIGHTS[target]
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        weighted_r2 += weight * r2
    return weighted_r2

#%%
def prepare_data(df):
    pivot = df.pivot_table(
        index=['image_path', 'State', 'Species', 'Sampling_Date', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name', values='target', aggfunc='first'
    ).reset_index()
    pivot.columns.name = None
    return pivot

train_df = pd.read_csv(DATA_PATH / "train.csv")
train_wide = prepare_data(train_df)
train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)

# Standard 5-fold split
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_wide['strat_bin'] = pd.qcut(train_wide['Dry_Total_g'], q=5, labels=False, duplicates='drop')

train_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(train_wide, train_wide['strat_bin'], train_wide['image_id'])):
    train_wide.loc[val_idx, 'fold'] = fold

print(f"Train samples: {len(train_wide)}")
print(f"Folds: {train_wide['fold'].value_counts().sort_index().to_dict()}")

#%% [markdown]
# ## 🎨 Augmentation

#%%
def get_train_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=cfg.hue_jitter),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%% [markdown]
# ## 📦 Dataset

#%%
class BiomassDataset(Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.data_path / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        left_img = img.crop((0, 0, mid, height))
        right_img = img.crop((mid, 0, width, height))
        
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Clover_g'], row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        return left_img, right_img, targets

#%% [markdown]
# ## 🧠 Model with Frozen Backbone

#%%
class FiLM(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )
    
    def forward(self, context):
        return torch.chunk(self.mlp(context), 2, dim=1)


def make_head(in_dim, hidden_dim, num_layers, dropout, use_layernorm):
    """v22: 더 단순한 head"""
    layers = []
    current_dim = in_dim
    
    for i in range(num_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        if i < num_layers - 1:
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


class CSIROModelV22(nn.Module):
    """
    v22: Backbone 완전 동결
    
    DINOv3는 pretrained feature extractor로만 사용
    Head만 학습하여 overfitting 방지
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Backbone 로드
        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3_qkvb.lvd1689m", 
            pretrained=False, num_classes=0, global_pool='avg'
        )
        weights_file = WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
        if weights_file.exists():
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        
        # === v22 핵심: Backbone 동결 ===
        if cfg.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen (feature extractor mode)")
        
        feat_dim = self.backbone.num_features  # 1024
        combined_dim = feat_dim * 2
        
        self.film = FiLM(feat_dim)
        
        # 더 단순한 heads
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, 
                                    cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                     cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                   cfg.dropout, cfg.use_layernorm)
        
        self.softplus = nn.Softplus(beta=1.0)
    
    def forward(self, left_img, right_img):
        # Backbone은 동결되어 있으므로 no_grad 사용
        with torch.set_grad_enabled(not self.training or not hasattr(self, '_freeze_backbone') or not self._freeze_backbone):
            left_feat = self.backbone(left_img)
            right_feat = self.backbone(right_img)
        
        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)
        
        left_mod = left_feat * (1 + gamma) + beta
        right_mod = right_feat * (1 + gamma) + beta
        
        combined = torch.cat([left_mod, right_mod], dim=1)
        
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))
        
        gdm = green + clover
        total = gdm + dead
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## 🏋️ Training

#%%
def train_fold(fold, train_df, cfg, device="cuda"):
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    train_ds = BiomassDataset(train_data, DATA_PATH, get_train_transforms(cfg))
    val_ds = BiomassDataset(val_data, DATA_PATH, get_val_transforms(cfg))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    model = CSIROModelV22(cfg).to(device)
    
    # === v22: Head 파라미터만 학습 ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler()
    
    best_score = -float('inf')
    no_improve = 0
    
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        
        for left, right, targets in train_loader:
            left = left.to(device)
            right = right.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(left, right)
                pred = outputs[:, [0, 2, 1]]  # Green, Clover, Dead
                loss = F.mse_loss(pred, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for left, right, targets in val_loader:
                left, right = left.to(device), right.to(device)
                outputs = model(left, right)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        # 5개 타겟으로 확장
        full_targets = np.zeros((len(targets), 5))
        full_targets[:, 0] = targets[:, 0]  # Green
        full_targets[:, 1] = targets[:, 2]  # Dead
        full_targets[:, 2] = targets[:, 1]  # Clover
        full_targets[:, 3] = targets[:, 0] + targets[:, 1]  # GDM
        full_targets[:, 4] = full_targets[:, 3] + targets[:, 2]  # Total
        
        val_score = competition_metric(full_targets, preds)
        
        wandb.log({
            f"fold{fold}/train_loss": train_loss,
            f"fold{fold}/val_score": val_score,
            f"fold{fold}/epoch": epoch + 1,
        })
        
        print(f"  Epoch {epoch+1}: loss={train_loss:.4f}, CV={val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f'model_fold{fold}.pth')
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    wandb.log({f"fold{fold}/best_score": best_score})
    
    flush()
    return best_score

#%% [markdown]
# ## 🚀 Run Training

#%%
run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name=f"v22_frozen_h{cfg.hidden_dim}_d{cfg.dropout}",
    config={
        "version": "v22",
        "freeze_backbone": cfg.freeze_backbone,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
    }
)

#%%
print("\n" + "="*60)
print("🚀 v22: Backbone Freezing + Aggressive Regularization")
print("="*60)

fold_scores = []

for fold in range(5):
    print(f"\n--- Fold {fold} ---")
    score = train_fold(fold, train_wide, cfg)
    fold_scores.append(score)
    
    wandb.log({
        "current_fold": fold,
        "running_mean": np.mean(fold_scores),
        "running_std": np.std(fold_scores) if len(fold_scores) > 1 else 0,
    })

#%%
mean_cv = np.mean(fold_scores)
std_cv = np.std(fold_scores)

wandb.log({
    "final/mean_cv": mean_cv,
    "final/std_cv": std_cv,
    "final/min_fold": np.min(fold_scores),
    "final/max_fold": np.max(fold_scores),
})

print("\n" + "="*60)
print("🎉 v22 RESULTS")
print("="*60)
print(f"Folds: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")

#%%
# Save to Drive
if GDRIVE_SAVE_PATH:
    for f in OUTPUT_DIR.glob("model_fold*.pth"):
        shutil.copy(f, GDRIVE_SAVE_PATH / f.name)
    
    with open(GDRIVE_SAVE_PATH / 'results.json', 'w') as f:
        json.dump({
            'fold_scores': fold_scores,
            'mean_cv': float(mean_cv),
            'std_cv': float(std_cv),
            'config': {
                'freeze_backbone': cfg.freeze_backbone,
                'hidden_dim': cfg.hidden_dim,
                'num_layers': cfg.num_layers,
                'dropout': cfg.dropout,
                'lr': cfg.lr,
                'weight_decay': cfg.weight_decay,
            }
        }, f, indent=2)
    print(f"\n✓ Saved to: {GDRIVE_SAVE_PATH}")

wandb.finish()
