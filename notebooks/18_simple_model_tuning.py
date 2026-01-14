#%% [markdown]
# # 🔬 v18: Simple Model Tuning
#
# **전략**: 단순한 모델 + 더 강한 regularization
#
# **핵심 가설**: 357개 데이터에서는 복잡한 모델이 overfitting
#
# **튜닝 범위** (v16보다 단순):
# - hidden_dim: [32, 64, 96, 128]
# - num_layers: [1, 2]
# - dropout: [0.2, 0.3, 0.4] (더 높은 dropout)

#%%
import os
import gc
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

import timm
from torchvision import transforms as T

import optuna
from optuna.trial import Trial

from sklearn.model_selection import StratifiedGroupKFold

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

print(f"PyTorch: {torch.__version__}")
print(f"Optuna: {optuna.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

#%% [markdown]
# ## 🔐 Setup

#%%
GDRIVE_SAVE_PATH = None

try:
    from google.colab import drive
    drive.mount('/content/drive')
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_v18')
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

def flush():
    gc.collect()
    torch.cuda.empty_cache()

seed_everything(42)

#%% [markdown]
# ## 📊 Data

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

#%%
TARGET_WEIGHTS = {'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1, 'GDM_g': 0.2, 'Dry_Total_g': 0.5}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(train_wide, train_wide['State'], groups=train_wide['image_id'])):
    train_wide.loc[val_idx, 'fold'] = fold

print(f"Train samples: {len(train_wide)}")

#%% [markdown]
# ## 🎨 Augmentation

#%%
def get_augmentation_strategy(strategy: str, img_size=(512, 512)):
    if strategy == "minimal":
        return T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif strategy == "moderate":
        return T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # color_focus
        return T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_val_transforms(img_size=(512, 512)):
    return T.Compose([
        T.Resize(img_size),
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
        gamma_beta = self.mlp(context)
        return torch.chunk(gamma_beta, 2, dim=1)

def make_head(in_dim: int, hidden_dim: int, num_layers: int, dropout: float, use_layernorm: bool):
    """동적 head 생성"""
    layers = []
    current_dim = in_dim
    
    for i in range(num_layers):
        out_dim = hidden_dim if i < num_layers - 1 else 1
        layers.append(nn.Linear(current_dim, out_dim if i < num_layers - 1 else hidden_dim))
        
        if i < num_layers - 1:
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


class CSIROModel(nn.Module):
    def __init__(self, model_name, weights_path, hidden_dim=64, num_layers=1, 
                 dropout=0.3, use_layernorm=False):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='avg')
        if weights_path.exists():
            state = torch.load(weights_path / "dinov3_vitl16_qkvb.pth", map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        
        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        
        self.film = FiLM(feat_dim)
        
        self.head_green = make_head(combined_dim, hidden_dim, num_layers, dropout, use_layernorm)
        self.head_clover = make_head(combined_dim, hidden_dim, num_layers, dropout, use_layernorm)
        self.head_dead = make_head(combined_dim, hidden_dim, num_layers, dropout, use_layernorm)
        
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
def train_one_fold(fold, train_df, config, device="cuda"):
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    train_transform = get_augmentation_strategy(config['aug_strategy'])
    val_transform = get_val_transforms()
    
    train_ds = BiomassDataset(train_data, DATA_PATH, train_transform)
    val_ds = BiomassDataset(val_data, DATA_PATH, val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size']*2, shuffle=False, num_workers=4, pin_memory=True)
    
    model = CSIROModel(
        model_name="vit_large_patch16_dinov3_qkvb.lvd1689m",
        weights_path=WEIGHTS_PATH,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_layernorm=config['use_layernorm']
    ).to(device)
    
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head_green.parameters()) + list(model.head_clover.parameters()) + \
                  list(model.head_dead.parameters()) + list(model.film.parameters())
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': config['lr'] * config['backbone_lr_mult']},
        {'params': head_params, 'lr': config['lr']}
    ], weight_decay=config['weight_decay'])
    
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler()
    
    best_score = -float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        for left, right, targets in train_loader:
            left, right, targets = left.to(device), right.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(left, right)
                pred = outputs[:, [0, 2, 1]]
                loss = F.mse_loss(pred, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
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
        
        full_targets = np.zeros((len(targets), 5))
        full_targets[:, 0] = targets[:, 0]
        full_targets[:, 1] = targets[:, 2]
        full_targets[:, 2] = targets[:, 1]
        full_targets[:, 3] = targets[:, 0] + targets[:, 1]
        full_targets[:, 4] = full_targets[:, 3] + targets[:, 2]
        
        score = competition_metric(full_targets, preds)
        
        if score > best_score:
            best_score = score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break
    
    flush()
    return best_score

#%% [markdown]
# ## 🔍 Optuna Objective (Simple Model Range)

#%%
def objective(trial: Trial) -> float:
    """단순한 모델 범위로 튜닝"""
    
    config = {
        # === 단순한 Head 구조 ===
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 96, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 2),
        'use_layernorm': trial.suggest_categorical('use_layernorm', [True, False]),
        
        # === 더 높은 Dropout (regularization) ===
        'dropout': trial.suggest_categorical('dropout', [0.2, 0.3, 0.4]),
        
        # === Learning rate ===
        'lr': trial.suggest_float('lr', 5e-5, 5e-4, log=True),
        'backbone_lr_mult': trial.suggest_float('backbone_lr_mult', 0.05, 0.15),
        'warmup_ratio': trial.suggest_float('warmup_ratio', 0.05, 0.15),
        
        # === Regularization (더 강하게) ===
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
        
        # === Augmentation ===
        'aug_strategy': trial.suggest_categorical('aug_strategy', ['minimal', 'moderate', 'color_focus']),
        
        # === Training ===
        'batch_size': trial.suggest_categorical('batch_size', [8, 16]),
        'epochs': 15,
        'patience': 5,
    }
    
    score = train_one_fold(fold=0, train_df=train_wide, config=config)
    return score

#%% [markdown]
# ## 🚀 Run Optuna

#%%
study = optuna.create_study(
    direction='maximize',
    study_name='biomass_simple_model_search',
    sampler=optuna.samplers.TPESampler(seed=42)
)

print("🔍 Simple Model Hyperparameter Search")
print("Range: hidden_dim=[32,64,96,128], layers=[1,2], dropout=[0.2,0.3,0.4]")

N_TRIALS = 25

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

#%%
print("\n" + "="*60)
print("🎉 OPTIMIZATION COMPLETE")
print("="*60)
print(f"\nBest CV: {study.best_trial.value:.4f}")
print(f"\nBest Parameters:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")

#%%
results = {
    'best_score': study.best_trial.value,
    'best_params': study.best_trial.params,
    'all_trials': [
        {'value': t.value, 'params': t.params} 
        for t in study.trials if t.value is not None
    ]
}

import json
results_path = OUTPUT_DIR / f"optuna_simple_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {results_path}")

if GDRIVE_SAVE_PATH:
    import shutil
    shutil.copy(results_path, GDRIVE_SAVE_PATH / results_path.name)
    print("Backed up to Drive")

#%%
print("\n📋 Best config for v19:")
print(json.dumps(study.best_trial.params, indent=2))
