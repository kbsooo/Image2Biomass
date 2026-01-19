#%% [markdown]
# # 🔍 CV2: Optuna Hyperparameter Optimization
#
# **핵심 전략 (Breakthrough Strategy 기반)**:
# 1. Date-based CV (cv1에서 구현됨)
# 2. 해상도 560x560
# 3. ⭐ **작은 Head 탐색**: 357개 이미지에 맞게!
#    - hidden_dim: 32, 64, 128, 256, 512
#    - num_layers: 1, 2, 3, 4
#    - dropout: 0.1~0.5
#
# **빠른 탐색을 위해 2-fold만 사용**

#%%
import os
import gc
import random
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

import optuna
from optuna.pruners import MedianPruner

import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

#%% [markdown]
# ## 🔐 Setup

#%%
import kagglehub

IS_KAGGLE = Path("/kaggle/input/csiro-biomass").exists()
if not IS_KAGGLE:
    kagglehub.login()

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

#%%
if IS_KAGGLE:
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large")
else:
    csiro_path = kagglehub.competition_download('csiro-biomass')
    weights_path = kagglehub.dataset_download('kbsooo/pretrained-weights-biomass')
    DATA_PATH = Path(csiro_path)
    WEIGHTS_PATH = Path(weights_path) / "dinov3_large" / "dinov3_large"

print(f"Data: {DATA_PATH}")

#%% [markdown]
# ## 📊 Data & Metrics

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
    
    return df

train_wide = create_proper_folds(train_wide)
print("✓ Date-based CV folds created")

#%% [markdown]
# ## 🎨 Augmentation & Dataset

#%%
IMG_SIZE = (560, 560)

def get_train_transforms():
    return T.Compose([
        T.Resize(IMG_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%%
class BiomassDataset(Dataset):
    def __init__(self, df, data_path, transform=None, 
                 height_mean=None, height_std=None,
                 ndvi_mean=None, ndvi_std=None):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.transform = transform
        
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
        
        main_targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Clover_g'], row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        height_norm = (row['Height_Ave_cm'] - self.height_mean) / (self.height_std + 1e-8)
        ndvi_norm = (row['Pre_GSHH_NDVI'] - self.ndvi_mean) / (self.ndvi_std + 1e-8)
        aux_targets = torch.tensor([height_norm, ndvi_norm], dtype=torch.float32)
        
        return left_img, right_img, main_targets, aux_targets
    
    def get_stats(self):
        return {
            'height_mean': self.height_mean,
            'height_std': self.height_std,
            'ndvi_mean': self.ndvi_mean,
            'ndvi_std': self.ndvi_std
        }

#%% [markdown]
# ## 🧠 Flexible Model

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


def make_head(in_dim, hidden_dim, num_layers, dropout, use_layernorm=True):
    """유연한 Head 생성"""
    if num_layers == 1:
        # 단일 레이어: in_dim → hidden_dim → 1
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    else:
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


class CSIROModelOptuna(nn.Module):
    """Optuna 탐색용 유연한 모델"""
    def __init__(self, hidden_dim, num_layers, dropout, use_aux=True):
        super().__init__()
        
        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3_qkvb.lvd1689m", 
            pretrained=False, num_classes=0, global_pool='avg'
        )
        weights_file = WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
        if weights_file.exists():
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        
        feat_dim = self.backbone.num_features  # 1024
        combined_dim = feat_dim * 2  # 2048
        
        self.film = FiLM(feat_dim)
        
        self.head_green = make_head(combined_dim, hidden_dim, num_layers, dropout)
        self.head_clover = make_head(combined_dim, hidden_dim, num_layers, dropout)
        self.head_dead = make_head(combined_dim, hidden_dim, num_layers, dropout)
        
        self.use_aux = use_aux
        if use_aux:
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
        
        if self.use_aux:
            height_pred = self.head_height(combined)
            ndvi_pred = self.head_ndvi(combined)
            aux_output = torch.cat([height_pred, ndvi_pred], dim=1)
            return main_output, aux_output
        
        return main_output

#%% [markdown]
# ## 🏋️ Training Function

#%%
def train_single_fold(fold, train_df, params, device="cuda", epochs=15, patience=5):
    """단일 Fold 학습 (빠른 Optuna 탐색용)"""
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    # Dataset
    train_ds = BiomassDataset(train_data, DATA_PATH, get_train_transforms())
    stats = train_ds.get_stats()
    val_ds = BiomassDataset(val_data, DATA_PATH, get_val_transforms(), **stats)
    
    batch_size = params.get('batch_size', 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model
    model = CSIROModelOptuna(
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        use_aux=params.get('use_aux', True)
    ).to(device)
    
    # Optimizer
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    
    backbone_lr_mult = params.get('backbone_lr_mult', 0.1)
    optimizer = AdamW([
        {'params': backbone_params, 'lr': params['lr'] * backbone_lr_mult},
        {'params': head_params, 'lr': params['lr']}
    ], weight_decay=params['weight_decay'])
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * params.get('warmup_ratio', 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler()
    aux_weight = params.get('aux_weight', 0.2)
    
    best_score = -float('inf')
    no_improve = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for left, right, main_targets, aux_targets in train_loader:
            left = left.to(device)
            right = right.to(device)
            main_targets = main_targets.to(device)
            aux_targets = aux_targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                if params.get('use_aux', True):
                    main_output, aux_output = model(left, right)
                    pred = main_output[:, [0, 2, 1]]
                    main_loss = F.mse_loss(pred, main_targets)
                    aux_loss = F.mse_loss(aux_output, aux_targets)
                    loss = main_loss + aux_weight * aux_loss
                else:
                    main_output = model(left, right)
                    pred = main_output[:, [0, 2, 1]]
                    loss = F.mse_loss(pred, main_targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
        # Validate
        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for left, right, main_targets, _ in val_loader:
                left, right = left.to(device), right.to(device)
                if params.get('use_aux', True):
                    main_output, _ = model(left, right)
                else:
                    main_output = model(left, right)
                all_preds.append(main_output.cpu().numpy())
                all_targets.append(main_targets.numpy())
        
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
        
        if val_score > best_score:
            best_score = val_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    flush()
    return best_score

#%% [markdown]
# ## 🔍 Optuna Objective

#%%
def objective(trial):
    """Optuna 목적 함수"""
    seed_everything(42)
    
    params = {
        # ⭐ 작은 값부터 탐색! (32 추가)
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512]),
        'num_layers': trial.suggest_int('num_layers', 1, 4),  # 4까지 확장
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        
        # Learning rate & regularization
        'lr': trial.suggest_float('lr', 1e-5, 5e-4, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'backbone_lr_mult': trial.suggest_float('backbone_lr_mult', 0.01, 0.2),
        
        # Training
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]),
        'warmup_ratio': trial.suggest_float('warmup_ratio', 0.05, 0.2),
        'aux_weight': trial.suggest_float('aux_weight', 0.0, 0.5),
        'use_aux': True,
    }
    
    # 2-fold만 빠르게 검증 (시간 절약)
    scores = []
    for fold in [0, 1]:
        try:
            score = train_single_fold(fold, train_wide, params, epochs=10, patience=3)
            scores.append(score)
            
            # Pruning: 첫 fold가 너무 나쁘면 조기 종료
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        except Exception as e:
            print(f"Error in fold {fold}: {e}")
            return -1.0
    
    mean_score = np.mean(scores)
    print(f"  Trial {trial.number}: {mean_score:.4f} (hidden={params['hidden_dim']}, layers={params['num_layers']})")
    
    return mean_score

#%% [markdown]
# ## 🚀 Run Optuna

#%%
print("\n" + "="*60)
print("🔍 Optuna Hyperparameter Optimization")
print("="*60)
print("Search space:")
print("  hidden_dim: [32, 64, 128, 256, 512]")
print("  num_layers: [1, 2, 3, 4]")
print("  dropout: [0.1, 0.5]")
print("  lr: [1e-5, 5e-4]")
print("  Using 2-fold fast validation")

#%%
# Optuna Study 생성
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
study = optuna.create_study(
    direction='maximize',
    pruner=pruner,
    study_name='csiro_cv2_optuna'
)

#%%
# 최적화 실행
N_TRIALS = 50  # 시간에 따라 조정

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

#%% [markdown]
# ## 📊 Results

#%%
print("\n" + "="*60)
print("🎉 Optuna Results")
print("="*60)

print(f"\nBest Trial:")
print(f"  Value: {study.best_value:.4f}")
print(f"  Params:")
for key, val in study.best_params.items():
    print(f"    {key}: {val}")

#%%
# 상위 10개 trials
print("\nTop 10 Trials:")
trials_df = study.trials_dataframe()
trials_df = trials_df.sort_values('value', ascending=False)
print(trials_df[['number', 'value', 'params_hidden_dim', 'params_num_layers', 'params_dropout']].head(10))

#%%
# Best params 저장
import json

best_params = study.best_params
best_params['best_value'] = study.best_value

with open('optuna_best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print(f"\n✓ Best params saved to optuna_best_params.json")

#%% [markdown]
# ## 💡 Next Steps
#
# 1. 최적 파라미터로 5-fold 전체 학습
# 2. `cv2_train.py` 에 best params 적용
# 3. Kaggle 제출 및 LB 확인
