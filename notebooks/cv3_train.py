#%% [markdown]
# # 🚀 CV3: LoRA + Progressive Head + Huber Loss
#
# **2026 SOTA 접근법**:
# 1. LoRA (r=8): Backbone 미세조정 (적은 파라미터)
# 2. Progressive Head: 2048 → 512 → 256 → 1 (점진적 축소)
# 3. LayerNorm: 수렴 안정성 향상
# 4. Huber Loss: Outlier 강건성
# 5. Date-based CV + 해상도 560

#%%
import os
import gc
import json
import random
import shutil
import zipfile
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
# ## 📦 LoRA Setup

#%%
# peft 설치 (Colab에서)
try:
    from peft import LoraConfig, get_peft_model
    print("✓ PEFT available")
except ImportError:
    print("Installing peft...")
    import subprocess
    subprocess.run(["pip", "install", "peft", "-q"])
    from peft import LoraConfig, get_peft_model
    print("✓ PEFT installed")

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
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_cv3')
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
    """
    CV3: LoRA + Progressive Head + Huber Loss
    
    2026 SOTA 접근법:
    - Backbone: Frozen + LoRA (r=8)
    - Head: 2048 → 512 → 256 → 1 (점진적 축소)
    - LayerNorm + Dropout 0.1
    - Huber Loss (delta=1.0)
    """
    # === LoRA 설정 ===
    use_lora = True
    lora_r = 8              # LoRA rank
    lora_alpha = 16         # LoRA alpha (일반적으로 2*r)
    lora_dropout = 0.1
    
    # === Head 설정 (Progressive) ===
    # DINOv2 Large: 1024 dim × 2 (left+right) = 2048
    head_dims = [512, 256]  # 2048 → 512 → 256 → 1
    head_dropout = 0.1
    
    # === 해상도 ===
    img_size = (560, 560)
    
    # === Training ===
    lr = 1e-4               # LoRA + Head 동시 학습에 안정적
    warmup_ratio = 0.1
    weight_decay = 1e-4
    
    batch_size = 8
    epochs = 25
    patience = 7
    hue_jitter = 0.02
    
    # === Loss ===
    use_huber_loss = True   # True: Huber, False: MSE
    huber_delta = 1.0
    use_weighted_loss = True
    aux_weight = 0.2

cfg = CFG()

EXP_NAME = f"cv3_lora_r{cfg.lora_r}_progressive"
if cfg.use_huber_loss:
    EXP_NAME += "_huber"
if cfg.use_weighted_loss:
    EXP_NAME += "_wloss"

print("="*60)
print(f"🔧 Experiment: {EXP_NAME}")
print("="*60)
print(f"  LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}")
print(f"  Head: 2048 → {' → '.join(map(str, cfg.head_dims))} → 1")
print(f"  Dropout: {cfg.head_dropout}")
print(f"  Loss: {'Huber' if cfg.use_huber_loss else 'MSE'} (delta={cfg.huber_delta})")
print(f"  LR: {cfg.lr}")

#%%
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

#%%
def create_proper_folds(df, n_splits=5):
    """Sampling_Date 기반 CV split"""
    df = df.copy()
    df['date_group'] = pd.to_datetime(df['Sampling_Date']).dt.strftime('%Y-%m-%d')
    df['strat_key'] = df['State'] + '_' + df['Month'].astype(str)
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(
        df, df['strat_key'], groups=df['date_group']
    )):
        df.loc[val_idx, 'fold'] = fold
    
    print("✓ Date-based CV folds created")
    return df

train_wide = create_proper_folds(train_wide)

#%% [markdown]
# ## 🎨 Augmentation & Dataset

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

#%%
class BiomassDataset(Dataset):
    def __init__(self, df, data_path, transform=None, 
                 height_mean=None, height_std=None,
                 ndvi_mean=None, ndvi_std=None,
                 return_idx=False):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.transform = transform
        self.return_idx = return_idx
        
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
        
        left_img = img.crop((0, 0, mid, height))
        right_img = img.crop((mid, 0, width, height))
        
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        full_targets = torch.tensor([
            row['Dry_Green_g'],
            row['Dry_Dead_g'],
            row['Dry_Clover_g'],
            row['Dry_Green_g'] + row['Dry_Clover_g'],
            row['Dry_Green_g'] + row['Dry_Clover_g'] + row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        height_norm = (row['Height_Ave_cm'] - self.height_mean) / (self.height_std + 1e-8)
        ndvi_norm = (row['Pre_GSHH_NDVI'] - self.ndvi_mean) / (self.ndvi_std + 1e-8)
        aux_targets = torch.tensor([height_norm, ndvi_norm], dtype=torch.float32)
        
        if self.return_idx:
            return left_img, right_img, full_targets, aux_targets, idx
        return left_img, right_img, full_targets, aux_targets
    
    def get_stats(self):
        return {
            'height_mean': self.height_mean,
            'height_std': self.height_std,
            'ndvi_mean': self.ndvi_mean,
            'ndvi_std': self.ndvi_std
        }

#%% [markdown]
# ## 📉 Weighted Huber Loss

#%%
class WeightedHuberLoss(nn.Module):
    """
    Weighted Huber Loss
    - Outlier에 강건 (MSE보다 안정적)
    - 대회 가중치 적용
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
        self.register_buffer('weights', torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5]))
    
    def forward(self, pred, target):
        # Huber loss per element
        diff = torch.abs(pred - target)
        huber = torch.where(
            diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        # Weighted sum
        weighted_huber = (huber * self.weights).sum(dim=1).mean()
        return weighted_huber

#%% [markdown]
# ## 🧠 Model with LoRA

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


class ProgressiveHead(nn.Module):
    """
    Progressive Head: 점진적 차원 축소
    2048 → 512 → 256 → 1
    LayerNorm + Dropout 적용
    """
    def __init__(self, in_dim, hidden_dims, dropout=0.1):
        super().__init__()
        
        layers = []
        current_dim = in_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 2026 SOTA: LayerNorm
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class CSIROModelCV3(nn.Module):
    """
    CV3: LoRA + Progressive Head
    
    - Backbone: DINOv2 Large + LoRA
    - Head: Progressive (2048 → 512 → 256 → 1)
    - FiLM fusion
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Backbone
        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3_qkvb.lvd1689m", 
            pretrained=False, num_classes=0, global_pool='avg'
        )
        weights_file = WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
        if weights_file.exists():
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        
        # LoRA 적용
        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["qkv"],  # ViT의 attention에 적용
                bias="none"
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()
        
        feat_dim = 1024  # DINOv2 Large
        combined_dim = feat_dim * 2
        
        self.film = FiLM(feat_dim)
        
        # Progressive Heads
        self.head_green = ProgressiveHead(combined_dim, cfg.head_dims, cfg.head_dropout)
        self.head_clover = ProgressiveHead(combined_dim, cfg.head_dims, cfg.head_dropout)
        self.head_dead = ProgressiveHead(combined_dim, cfg.head_dims, cfg.head_dropout)
        
        # Aux heads
        self.head_height = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.head_ndvi = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
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
# ## 🏋️ Training

#%%
def train_fold(fold, train_df, cfg, device="cuda"):
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    print(f"\n  Train: {len(train_data)} | Val: {len(val_data)}")
    
    # Dataset
    train_ds = BiomassDataset(train_data, DATA_PATH, get_train_transforms(cfg))
    stats = train_ds.get_stats()
    val_ds = BiomassDataset(val_data, DATA_PATH, get_val_transforms(cfg), 
                            return_idx=True, **stats)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model
    model = CSIROModelCV3(cfg).to(device)
    
    # Loss
    if cfg.use_huber_loss and cfg.use_weighted_loss:
        main_criterion = WeightedHuberLoss(delta=cfg.huber_delta).to(device)
        print("  Using Weighted Huber Loss")
    elif cfg.use_huber_loss:
        main_criterion = nn.HuberLoss(delta=cfg.huber_delta)
        print("  Using Huber Loss")
    else:
        main_criterion = nn.MSELoss()
        print("  Using MSE Loss")
    
    # Optimizer (LoRA + Head 함께)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler()
    
    best_score = -float('inf')
    no_improve = 0
    best_oof = None
    
    for epoch in range(cfg.epochs):
        # Train
        model.train()
        train_loss = 0
        
        for left, right, full_targets, aux_targets in train_loader:
            left = left.to(device)
            right = right.to(device)
            full_targets = full_targets.to(device)
            aux_targets = aux_targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                main_output, aux_output = model(left, right)
                main_loss = main_criterion(main_output, full_targets)
                aux_loss = F.mse_loss(aux_output, aux_targets)
                loss = main_loss + cfg.aux_weight * aux_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        all_preds, all_targets, all_indices = [], [], []
        
        with torch.no_grad():
            for left, right, full_targets, _, indices in val_loader:
                left, right = left.to(device), right.to(device)
                main_output, _ = model(left, right)
                all_preds.append(main_output.cpu().numpy())
                all_targets.append(full_targets.numpy())
                all_indices.extend(indices.numpy().tolist())
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        val_score = competition_metric(targets, preds)
        
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
            
            best_oof = {
                'predictions': preds.copy(),
                'targets': targets.copy(),
                'indices': np.array(all_indices),
                'fold': fold,
                'val_score': val_score
            }
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    np.save(OUTPUT_DIR / f'oof_fold{fold}.npy', best_oof)
    print(f"  ✓ Best score: {best_score:.4f}")
    
    wandb.log({f"fold{fold}/best_score": best_score})
    
    flush()
    return best_score, best_oof

#%% [markdown]
# ## 🚀 Run Training

#%%
run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name=EXP_NAME,
    config={
        "version": "cv3",
        "exp_name": EXP_NAME,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "head_dims": cfg.head_dims,
        "head_dropout": cfg.head_dropout,
        "use_huber_loss": cfg.use_huber_loss,
        "huber_delta": cfg.huber_delta,
        "lr": cfg.lr,
    }
)

#%%
print("\n" + "="*60)
print(f"🚀 {EXP_NAME}")
print("="*60)

fold_scores = []

for fold in range(5):
    print(f"\n--- Fold {fold} ---")
    score, _ = train_fold(fold, train_wide, cfg)
    fold_scores.append(score)

#%%
mean_cv = np.mean(fold_scores)
std_cv = np.std(fold_scores)

print("\n" + "="*60)
print(f"🎉 {EXP_NAME} RESULTS")
print("="*60)
print(f"Folds: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")

#%% [markdown]
# ## 📊 OOF Score

#%%
all_predictions = []
all_targets = []

for fold in range(5):
    oof = np.load(OUTPUT_DIR / f'oof_fold{fold}.npy', allow_pickle=True).item()
    all_predictions.append(oof['predictions'])
    all_targets.append(oof['targets'])

oof_predictions = np.concatenate(all_predictions)
oof_targets = np.concatenate(all_targets)

total_oof_score = competition_metric(oof_targets, oof_predictions)
print(f"\n✓ Total OOF Score: {total_oof_score:.4f}")

#%% [markdown]
# ## 💾 Save to Google Drive

#%%
if GDRIVE_SAVE_PATH:
    for f in OUTPUT_DIR.glob("model_fold*.pth"):
        shutil.copy(f, GDRIVE_SAVE_PATH / f.name)
    for f in OUTPUT_DIR.glob("oof_fold*.npy"):
        shutil.copy(f, GDRIVE_SAVE_PATH / f.name)
    
    zip_path = GDRIVE_SAVE_PATH / f'{EXP_NAME}_models.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in OUTPUT_DIR.glob("model_fold*.pth"):
            zf.write(f, f.name)
        for f in OUTPUT_DIR.glob("oof_fold*.npy"):
            zf.write(f, f.name)
    print(f"✓ ZIP saved: {zip_path}")
    
    with open(GDRIVE_SAVE_PATH / 'results.json', 'w') as f:
        json.dump({
            'exp_name': EXP_NAME,
            'fold_scores': fold_scores,
            'mean_cv': float(mean_cv),
            'std_cv': float(std_cv),
            'total_oof_score': float(total_oof_score),
            'config': {
                'lora_r': cfg.lora_r,
                'lora_alpha': cfg.lora_alpha,
                'head_dims': cfg.head_dims,
                'head_dropout': cfg.head_dropout,
                'use_huber_loss': cfg.use_huber_loss,
                'lr': cfg.lr,
            }
        }, f, indent=2)
    
    print(f"✓ All saved to: {GDRIVE_SAVE_PATH}")

wandb.log({
    "final/mean_cv": mean_cv,
    "final/std_cv": std_cv,
    "final/oof_score": total_oof_score,
})

wandb.finish()

print(f"\n" + "="*60)
print(f"✅ {EXP_NAME} 완료!")
print(f"   Mean CV: {mean_cv:.4f}")
print(f"   OOF Score: {total_oof_score:.4f}")
print("="*60)
