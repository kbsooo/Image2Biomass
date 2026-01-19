#%% [markdown]
# # 🚀 v23: Consistency Regularization
#
# **v20 대비 개선사항**:
# - Consistency Loss: 같은 이미지 + 다른 augmentation → 같은 예측 강제
# - 목표: CV-LB gap 감소 및 일반화 성능 향상
#
# **기반**: v20 (Hue 축소 + Auxiliary Tasks + Multi-Stratified Folds)

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
from sklearn.preprocessing import LabelEncoder

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
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_v23')
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
    # Data
    img_size = (512, 512)
    
    # Model (v20과 동일)
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
    
    # v20 설정
    hue_jitter = 0.02
    aux_weight = 0.2
    
    # === NEW: Consistency Regularization ===
    consistency_weight = 0.1  # consistency loss 가중치
    
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
train_wide['Month'] = pd.to_datetime(train_wide['Sampling_Date']).dt.month

print(f"Train samples: {len(train_wide)}")
print(f"\nState 분포:\n{train_wide['State'].value_counts()}")
print(f"\nMonth 분포:\n{train_wide['Month'].value_counts().sort_index()}")

#%% [markdown]
# ## 🎯 Multi-Stratified Fold Split

#%%
def create_stratified_folds(df, n_splits=5):
    df = df.copy()
    df['strat_key'] = df['State'] + '_' + df['Month'].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(df, df['strat_key'], groups=df['image_id'])):
        df.loc[val_idx, 'fold'] = fold
    
    return df

train_wide = create_stratified_folds(train_wide)

print("\n=== Fold 분포 검증 ===")
for fold in range(5):
    fold_data = train_wide[train_wide['fold'] == fold]
    print(f"\nFold {fold}: {len(fold_data)} samples")
    print(f"  State: {fold_data['State'].value_counts().to_dict()}")

#%% [markdown]
# ## 🎨 Augmentation

#%%
def get_train_transforms(cfg):
    """Train augmentation"""
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
# ## 📦 Dataset with Dual Augmentation (NEW)

#%%
class BiomassDatasetV23(Dataset):
    """
    v23 Dataset: 학습 시 같은 이미지에 두 번의 다른 augmentation 적용
    
    Consistency Regularization을 위해:
    - Training: (left1, right1, left2, right2, main_targets, aux_targets) 반환
    - Validation: (left, right, main_targets, aux_targets) 반환
    """
    def __init__(self, df, data_path, transform=None, is_train=True,
                 height_mean=None, height_std=None, ndvi_mean=None, ndvi_std=None):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.transform = transform
        self.is_train = is_train
        
        self.height_mean = height_mean if height_mean else df['Height_Ave_cm'].mean()
        self.height_std = height_std if height_std else df['Height_Ave_cm'].std()
        self.ndvi_mean = ndvi_mean if ndvi_mean else df['Pre_GSHH_NDVI'].mean()
        self.ndvi_std = ndvi_std if ndvi_std else df['Pre_GSHH_NDVI'].std()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.data_path / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        left_pil = img.crop((0, 0, mid, height))
        right_pil = img.crop((mid, 0, width, height))
        
        # 메인 타겟
        main_targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Clover_g'], row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        # Auxiliary 타겟
        height_norm = (row['Height_Ave_cm'] - self.height_mean) / (self.height_std + 1e-8)
        ndvi_norm = (row['Pre_GSHH_NDVI'] - self.ndvi_mean) / (self.ndvi_std + 1e-8)
        aux_targets = torch.tensor([height_norm, ndvi_norm], dtype=torch.float32)
        
        if self.is_train and self.transform:
            # === 핵심: 같은 이미지에 두 번의 다른 augmentation ===
            left1 = self.transform(left_pil)
            right1 = self.transform(right_pil)
            left2 = self.transform(left_pil)
            right2 = self.transform(right_pil)
            return left1, right1, left2, right2, main_targets, aux_targets
        else:
            left = self.transform(left_pil) if self.transform else left_pil
            right = self.transform(right_pil) if self.transform else right_pil
            return left, right, main_targets, aux_targets
    
    def get_stats(self):
        return {
            'height_mean': self.height_mean,
            'height_std': self.height_std,
            'ndvi_mean': self.ndvi_mean,
            'ndvi_std': self.ndvi_std
        }

#%% [markdown]
# ## 🧠 Model (v20과 동일)

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


class CSIROModelV23(nn.Module):
    """v23 Model (구조는 v20과 동일)"""
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
        
        self.head_height = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self.head_ndvi = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
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
        
        main_output = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        height_pred = self.head_height(combined)
        ndvi_pred = self.head_ndvi(combined)
        aux_output = torch.cat([height_pred, ndvi_pred], dim=1)
        
        return main_output, aux_output

#%% [markdown]
# ## 🏋️ Training with Consistency Regularization (NEW)

#%%
def train_fold(fold, train_df, cfg, device="cuda"):
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    # Dataset
    train_ds = BiomassDatasetV23(train_data, DATA_PATH, get_train_transforms(cfg), is_train=True)
    stats = train_ds.get_stats()
    val_ds = BiomassDatasetV23(val_data, DATA_PATH, get_val_transforms(cfg), is_train=False, **stats)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model
    model = CSIROModelV23(cfg).to(device)
    
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.head_green.parameters()) + 
                   list(model.head_clover.parameters()) +
                   list(model.head_dead.parameters()) + 
                   list(model.head_height.parameters()) +
                   list(model.head_ndvi.parameters()) +
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
        # ===== Train with Consistency Regularization =====
        model.train()
        train_loss = 0
        train_main_loss = 0
        train_aux_loss = 0
        train_cons_loss = 0
        
        for left1, right1, left2, right2, main_targets, aux_targets in train_loader:
            left1, right1 = left1.to(device), right1.to(device)
            left2, right2 = left2.to(device), right2.to(device)
            main_targets = main_targets.to(device)
            aux_targets = aux_targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                # 첫 번째 augmentation 예측
                main_output1, aux_output1 = model(left1, right1)
                
                # 두 번째 augmentation 예측
                main_output2, aux_output2 = model(left2, right2)
                
                # Supervised Loss (첫 번째 예측 기준)
                pred = main_output1[:, [0, 2, 1]]  # Green, Clover, Dead
                main_loss = F.mse_loss(pred, main_targets)
                aux_loss = F.mse_loss(aux_output1, aux_targets)
                
                # === NEW: Consistency Loss ===
                # 두 예측이 일치해야 함 (augmentation에 불변)
                consistency_loss = F.mse_loss(main_output1, main_output2)
                
                # Total Loss
                loss = main_loss + cfg.aux_weight * aux_loss + cfg.consistency_weight * consistency_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            train_main_loss += main_loss.item()
            train_aux_loss += aux_loss.item()
            train_cons_loss += consistency_loss.item()
        
        train_loss /= len(train_loader)
        train_main_loss /= len(train_loader)
        train_aux_loss /= len(train_loader)
        train_cons_loss /= len(train_loader)
        
        # ===== Validate =====
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for left, right, main_targets, _ in val_loader:
                left, right = left.to(device), right.to(device)
                main_output, _ = model(left, right)
                all_preds.append(main_output.cpu().numpy())
                all_targets.append(main_targets.numpy())
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        full_targets = np.zeros((len(targets), 5))
        full_targets[:, 0] = targets[:, 0]  # Green
        full_targets[:, 1] = targets[:, 2]  # Dead
        full_targets[:, 2] = targets[:, 1]  # Clover
        full_targets[:, 3] = targets[:, 0] + targets[:, 1]  # GDM
        full_targets[:, 4] = full_targets[:, 3] + targets[:, 2]  # Total
        
        val_score = competition_metric(full_targets, preds)
        
        # WandB logging
        wandb.log({
            f"fold{fold}/train_loss": train_loss,
            f"fold{fold}/main_loss": train_main_loss,
            f"fold{fold}/aux_loss": train_aux_loss,
            f"fold{fold}/consistency_loss": train_cons_loss,
            f"fold{fold}/val_score": val_score,
            f"fold{fold}/epoch": epoch + 1,
        })
        
        print(f"  Epoch {epoch+1}: loss={train_loss:.4f} (main={train_main_loss:.4f}, aux={train_aux_loss:.4f}, cons={train_cons_loss:.4f}), CV={val_score:.4f}")
        
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
    name=f"v23_consistency{cfg.consistency_weight}",
    config={
        "version": "v23",
        "consistency_weight": cfg.consistency_weight,
        "hue_jitter": cfg.hue_jitter,
        "aux_weight": cfg.aux_weight,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "lr": cfg.lr,
    }
)

#%%
print("\n" + "="*60)
print("🚀 v23 Training: Consistency Regularization")
print("="*60)
print(f"NEW: Consistency weight: {cfg.consistency_weight}")
print(f"     Hue jitter: {cfg.hue_jitter}")
print(f"     Auxiliary weight: {cfg.aux_weight}")

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
print("🎉 v23 RESULTS")
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
                'consistency_weight': cfg.consistency_weight,
                'hue_jitter': cfg.hue_jitter,
                'aux_weight': cfg.aux_weight,
                'hidden_dim': cfg.hidden_dim,
                'num_layers': cfg.num_layers,
            }
        }, f, indent=2)
    print(f"\n✓ Saved to: {GDRIVE_SAVE_PATH}")

wandb.finish()
