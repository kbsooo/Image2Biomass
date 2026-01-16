#%% [markdown]
# # 🚀 v21: Leave-One-Group-Out Cross-Validation
#
# **전략**: 처음 보는 계절/지역에 대한 일반화 능력 테스트
# - Leave-One-Season-Out: 3계절로 학습 → 1계절 예측
# - Leave-One-State-Out: 3지역으로 학습 → 1지역 예측

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
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_v21')
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
    
    # Model (Trial 12 기반)
    hidden_dim = 512
    num_layers = 3
    dropout = 0.1
    use_layernorm = True
    
    # Training
    lr = 2.33e-4
    backbone_lr_mult = 0.084
    warmup_ratio = 0.078
    weight_decay = 6.37e-5
    
    batch_size = 8
    epochs = 25
    patience = 7
    
    hue_jitter = 0.02  # v20에서 적용한 hue 축소
    
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

#%% [markdown]
# ## 📊 Data Loading & Group Split

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

# 월 추출
train_wide['Month'] = pd.to_datetime(train_wide['Sampling_Date']).dt.month

# 호주 남반구 계절 매핑
def get_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:  # 9, 10, 11
        return 'Spring'

train_wide['Season'] = train_wide['Month'].apply(get_season)

print(f"Train samples: {len(train_wide)}")
print(f"\nState 분포: {train_wide['State'].value_counts().to_dict()}")
print(f"Season 분포: {train_wide['Season'].value_counts().to_dict()}")

#%% [markdown]
# ## 🎯 Leave-One-Group-Out Fold 생성

#%%
# === Option 1: Leave-One-Season-Out ===
# 4 folds: 각 fold가 하나의 계절을 val로 사용

SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']
STATES = ['Tas', 'Vic', 'NSW', 'WA']

# 실행할 전략 선택 (한번에 하나만!)
SPLIT_MODE = "season"  # "season" or "state"

if SPLIT_MODE == "season":
    GROUPS = SEASONS
    GROUP_COL = "Season"
else:
    GROUPS = STATES
    GROUP_COL = "State"

print(f"\n=== Leave-One-{GROUP_COL}-Out ===")
for i, group in enumerate(GROUPS):
    count = len(train_wide[train_wide[GROUP_COL] == group])
    print(f"Fold {i}: val={group} ({count} samples)")

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
# ## 🧠 Model

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


class CSIROModelV21(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3_qkvb.lvd1689m", 
            pretrained=False, num_classes=0, global_pool='avg'
        )
        weights_file = WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
        if weights_file.exists():
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        
        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        
        self.film = FiLM(feat_dim)
        
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, 
                                    cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                     cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                   cfg.dropout, cfg.use_layernorm)
        
        self.softplus = nn.Softplus(beta=1.0)
    
    def forward(self, left_img, right_img):
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
def train_fold(fold_idx, val_group, train_df, cfg, device="cuda"):
    """
    Leave-One-Group-Out: val_group에 해당하는 데이터를 validation으로 사용
    """
    train_data = train_df[train_df[GROUP_COL] != val_group].reset_index(drop=True)
    val_data = train_df[train_df[GROUP_COL] == val_group].reset_index(drop=True)
    
    print(f"  Train: {len(train_data)} (excluding {val_group})")
    print(f"  Val: {len(val_data)} ({val_group} only)")
    
    if len(val_data) == 0:
        print(f"  ⚠️ No validation data for {val_group}, skipping...")
        return None
    
    train_ds = BiomassDataset(train_data, DATA_PATH, get_train_transforms(cfg))
    val_ds = BiomassDataset(val_data, DATA_PATH, get_val_transforms(cfg))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    model = CSIROModelV21(cfg).to(device)
    
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.head_green.parameters()) + 
                   list(model.head_clover.parameters()) +
                   list(model.head_dead.parameters()) + 
                   list(model.film.parameters()))
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': cfg.lr * cfg.backbone_lr_mult},
        {'params': head_params, 'lr': cfg.lr}
    ], weight_decay=cfg.weight_decay)
    
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
            f"fold{fold_idx}/train_loss": train_loss,
            f"fold{fold_idx}/val_score": val_score,
            f"fold{fold_idx}/epoch": epoch + 1,
        })
        
        print(f"  Epoch {epoch+1}: loss={train_loss:.4f}, CV={val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f'model_fold{fold_idx}.pth')
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    wandb.log({f"fold{fold_idx}/best_score": best_score})
    
    flush()
    return best_score

#%% [markdown]
# ## 🚀 Run Training

#%%
run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name=f"v21_LOGO_{SPLIT_MODE}",
    config={
        "version": "v21",
        "split_mode": SPLIT_MODE,
        "groups": GROUPS,
        "hidden_dim": cfg.hidden_dim,
        "lr": cfg.lr,
    }
)

#%%
print("\n" + "="*60)
print(f"🚀 v21: Leave-One-{GROUP_COL}-Out Training")
print("="*60)
print(f"Groups: {GROUPS}")

fold_scores = []
valid_folds = 0

for fold_idx, val_group in enumerate(GROUPS):
    print(f"\n--- Fold {fold_idx}: val={val_group} ---")
    score = train_fold(fold_idx, val_group, train_wide, cfg)
    
    if score is not None:
        fold_scores.append(score)
        valid_folds += 1
        
        wandb.log({
            "current_fold": fold_idx,
            "val_group": val_group,
            "running_mean": np.mean(fold_scores),
        })

#%%
if fold_scores:
    mean_cv = np.mean(fold_scores)
    std_cv = np.std(fold_scores)

    wandb.log({
        "final/mean_cv": mean_cv,
        "final/std_cv": std_cv,
        "final/min_fold": np.min(fold_scores),
        "final/max_fold": np.max(fold_scores),
        "final/valid_folds": valid_folds,
    })

    print("\n" + "="*60)
    print(f"🎉 v21 RESULTS (Leave-One-{GROUP_COL}-Out)")
    print("="*60)
    print(f"Folds: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")
else:
    print("No valid folds!")

#%%
# Save to Drive
if GDRIVE_SAVE_PATH and fold_scores:
    for f in OUTPUT_DIR.glob("model_fold*.pth"):
        shutil.copy(f, GDRIVE_SAVE_PATH / f.name)
    
    with open(GDRIVE_SAVE_PATH / 'results.json', 'w') as f:
        json.dump({
            'split_mode': SPLIT_MODE,
            'groups': GROUPS,
            'fold_scores': fold_scores,
            'mean_cv': float(mean_cv),
            'std_cv': float(std_cv),
        }, f, indent=2)
    print(f"\n✓ Saved to: {GDRIVE_SAVE_PATH}")

wandb.finish()
