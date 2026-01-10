#%% [markdown]
# # 🔬 Exp B: Pseudo-Labeling
#
# **아이디어**: 
# 1. Image → NDVI/Height 예측 모델 학습 (Train 데이터로)
# 2. Test 이미지에 대해 NDVI/Height 예측
# 3. 예측된 pseudo tabular로 Multi-Modal 학습
#
# **장점**: Test에서도 tabular 정보 활용 가능

#%%
import os
import gc
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")

#%%
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

seed_everything(42)

#%% [markdown]
# ## Configuration

#%%
class CFG:
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    OUTPUT_DIR = Path("/kaggle/working")
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass")
    
    backbone = "efficientnet_b4"
    input_size = 384
    
    # Phase 1: Tabular 예측 모델
    tabular_epochs = 10
    tabular_lr = 2e-4
    
    # Phase 2: Biomass 예측 모델
    n_folds = 5
    train_folds = 1  # 싹수 확인
    epochs = 10
    batch_size = 16
    lr = 2e-4
    weight_decay = 1e-4
    
    tabular_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
    
    seed = 42
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    independent_targets = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g']
    all_targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

cfg = CFG()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {cfg.device}")

#%% [markdown]
# ## Phase 1: Image → Tabular 예측 모델

#%%
class TabularPredictorDataset(Dataset):
    """Image → NDVI/Height 예측을 위한 데이터셋"""
    def __init__(self, df, cfg, transforms=None, scaler=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
        
        tabular_data = df[cfg.tabular_cols].values.astype(np.float32)
        if scaler is not None:
            if mode == 'train':
                self.targets = scaler.fit_transform(tabular_data)
            else:
                self.targets = scaler.transform(tabular_data)
        else:
            self.targets = tabular_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = self.cfg.DATA_PATH / row['image_path']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img, targets

#%%
class TabularPredictor(nn.Module):
    """Image → NDVI/Height 예측 모델"""
    def __init__(self, backbone_name, n_outputs=2, pretrained=True, weights_path=None):
        super().__init__()
        
        if pretrained and weights_path and Path(weights_path).exists():
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
            weights = torch.load(weights_path, weights_only=True)
            weights = {k: v for k, v in weights.items() if not k.startswith('classifier')}
            self.backbone.load_state_dict(weights, strict=False)
            print(f"✓ Loaded pretrained weights")
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs)
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

#%%
def get_transforms(mode: str = 'train', size: int = 384) -> A.Compose:
    if mode == 'train':
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05, p=0.7),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

#%%
def prepare_data(df, is_train=True):
    if 'target' in df.columns:
        df_wide = pd.pivot_table(
            df, values='target',
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
            columns='target_name', aggfunc='mean'
        ).reset_index()
    else:
        df['target'] = 0
        cols = ['image_path']
        for col in ['Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']:
            if col in df.columns:
                cols.append(col)
        df_wide = df.drop_duplicates(subset=['image_path'])[cols].reset_index(drop=True)
    return df_wide

#%%
# Load data
train_df = pd.read_csv(cfg.DATA_PATH / "train.csv")
train_wide = prepare_data(train_df, is_train=True)

# KFold
kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
train_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(kf.split(train_wide)):
    train_wide.loc[val_idx, 'fold'] = fold

print(f"Train data: {len(train_wide)}")

#%% [markdown]
# ## Phase 1: Train Tabular Predictor

#%%
def train_tabular_predictor(train_df, cfg):
    print(f"\n{'='*60}")
    print(f"📊 Phase 1: Training Tabular Predictor (Image → NDVI/Height)")
    print(f"{'='*60}")
    
    # Use all data for tabular predictor (or split if needed)
    train_data = train_df[train_df['fold'] != 0].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == 0].reset_index(drop=True)
    
    scaler = StandardScaler()
    
    train_dataset = TabularPredictorDataset(
        train_data, cfg, get_transforms('train', cfg.input_size), scaler, 'train'
    )
    val_dataset = TabularPredictorDataset(
        val_data, cfg, get_transforms('val', cfg.input_size), scaler, 'val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)
    
    weights_path = None
    if cfg.WEIGHTS_PATH.exists():
        weights_path = str(cfg.WEIGHTS_PATH / cfg.backbone / f"{cfg.backbone}.pth")
    
    model = TabularPredictor(cfg.backbone, n_outputs=2, pretrained=True, weights_path=weights_path)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(cfg.device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.tabular_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.tabular_epochs)
    
    best_loss = float('inf')
    
    for epoch in range(cfg.tabular_epochs):
        # Train
        model.train()
        train_loss = 0
        for imgs, targets in tqdm(train_loader, desc='Train', leave=False):
            imgs, targets = imgs.to(cfg.device), targets.to(cfg.device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(cfg.device), targets.to(cfg.device)
                preds = model(imgs)
                loss = F.mse_loss(preds, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{cfg.tabular_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'scaler': scaler,
            }, cfg.OUTPUT_DIR / 'tabular_predictor.pt')
            print(f"  ✓ Saved!")
    
    flush()
    return scaler

#%%
tabular_scaler = train_tabular_predictor(train_wide, cfg)

#%% [markdown]
# ## Phase 1.5: Generate Pseudo Labels for Test

#%%
@torch.no_grad()
def generate_pseudo_labels(model, loader, device, scaler):
    model.eval()
    all_preds = []
    
    for batch in tqdm(loader, desc='Generating Pseudo Labels'):
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch
        imgs = imgs.to(device)
        preds = model(imgs)
        all_preds.append(preds.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    # Inverse transform
    all_preds = scaler.inverse_transform(all_preds)
    return all_preds

#%%
# Load test data
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_wide = prepare_data(test_df, is_train=False)
print(f"Test data: {len(test_wide)} images")

# Create test dataset for pseudo labeling
class SimpleImageDataset(Dataset):
    def __init__(self, df, cfg, transforms):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.cfg.DATA_PATH / row['image_path']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img

test_dataset = SimpleImageDataset(test_wide, cfg, get_transforms('val', cfg.input_size))
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

# Load tabular predictor
ckpt = torch.load(cfg.OUTPUT_DIR / 'tabular_predictor.pt', weights_only=False)
tabular_model = TabularPredictor(cfg.backbone, n_outputs=2, pretrained=False).to(cfg.device)
tabular_model.load_state_dict(ckpt['model_state_dict'])
scaler = ckpt['scaler']

# Generate pseudo labels
pseudo_tabular = generate_pseudo_labels(tabular_model, test_loader, cfg.device, scaler)
print(f"Pseudo labels shape: {pseudo_tabular.shape}")
print(f"  NDVI: min={pseudo_tabular[:, 0].min():.2f}, max={pseudo_tabular[:, 0].max():.2f}")
print(f"  Height: min={pseudo_tabular[:, 1].min():.2f}, max={pseudo_tabular[:, 1].max():.2f}")

# Add to test_wide
test_wide['Pre_GSHH_NDVI'] = pseudo_tabular[:, 0]
test_wide['Height_Ave_cm'] = pseudo_tabular[:, 1]

#%% [markdown]
# ## Phase 2: Train Biomass Model with Tabular (using pseudo labels for test)

#%%
# Competition metric
TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1,
    'GDM_g': 0.2, 'Dry_Total_g': 0.5,
}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true, y_pred):
    weights = np.array([TARGET_WEIGHTS[t] for t in TARGET_ORDER])
    y_weighted_mean = sum(y_true[:, i].mean() * weights[i] for i in range(5))
    ss_res = sum(((y_true[:, i] - y_pred[:, i]) ** 2).mean() * weights[i] for i in range(5))
    ss_tot = sum(((y_true[:, i] - y_weighted_mean) ** 2).mean() * weights[i] for i in range(5))
    return 1 - ss_res / (ss_tot + 1e-8)

#%%
class BiomassDatasetWithTabular(Dataset):
    """Image + Tabular → Biomass"""
    def __init__(self, df, cfg, transforms=None, mode='train', tabular_scaler=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode
        
        # Tabular features
        if all(col in df.columns for col in cfg.tabular_cols):
            tabular_data = df[cfg.tabular_cols].values.astype(np.float32)
            if tabular_scaler is not None:
                if mode == 'train':
                    self.tabular = tabular_scaler.fit_transform(tabular_data)
                else:
                    self.tabular = tabular_scaler.transform(tabular_data)
            else:
                self.tabular = tabular_data
            self.has_tabular = True
        else:
            self.tabular = None
            self.has_tabular = False
        
        # Add dummy targets for test
        for t in TARGET_ORDER:
            if t not in df.columns:
                df[t] = 0.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = self.cfg.DATA_PATH / row['image_path']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Clover_g'], row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        if self.has_tabular:
            tabular = torch.tensor(self.tabular[idx], dtype=torch.float32)
            return img, tabular, targets
        return img, targets

#%%
class PhysicsConstrainedHead(nn.Module):
    def __init__(self, in_features, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        raw = self.head(x)
        independent = self.softplus(raw)
        green, clover, dead = independent[:, 0:1], independent[:, 1:2], independent[:, 2:3]
        gdm = green + clover
        total = gdm + dead
        full = torch.cat([green, dead, clover, gdm, total], dim=1)
        return independent, full

#%%
class BiomassModelWithTabular(nn.Module):
    """FiLM Conditioning: Image + Tabular → Biomass"""
    def __init__(self, backbone_name, n_tabular=2, pretrained=True, weights_path=None):
        super().__init__()
        
        if pretrained and weights_path and Path(weights_path).exists():
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
            weights = torch.load(weights_path, weights_only=True)
            weights = {k: v for k, v in weights.items() if not k.startswith('classifier')}
            self.backbone.load_state_dict(weights, strict=False)
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        feat_dim = self.backbone.num_features
        
        # Tabular encoder + FiLM
        self.tabular_encoder = nn.Sequential(
            nn.Linear(n_tabular, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.film_gamma = nn.Linear(128, feat_dim)
        self.film_beta = nn.Linear(128, feat_dim)
        
        self.head = PhysicsConstrainedHead(feat_dim)
    
    def forward(self, image, tabular=None):
        img_feat = self.backbone(image)
        
        if tabular is not None:
            tab_feat = self.tabular_encoder(tabular)
            gamma = self.film_gamma(tab_feat)
            beta = self.film_beta(tab_feat)
            img_feat = img_feat * (1 + gamma) + beta
        
        return self.head(img_feat)

#%%
def train_biomass_with_tabular(train_df, cfg):
    print(f"\n{'='*60}")
    print(f"🌿 Phase 2: Training Biomass Model with Tabular")
    print(f"{'='*60}")
    
    train_data = train_df[train_df['fold'] != 0].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == 0].reset_index(drop=True)
    
    tabular_scaler = StandardScaler()
    
    train_dataset = BiomassDatasetWithTabular(
        train_data, cfg, get_transforms('train', cfg.input_size), 'train', tabular_scaler
    )
    val_dataset = BiomassDatasetWithTabular(
        val_data, cfg, get_transforms('val', cfg.input_size), 'val', tabular_scaler
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)
    
    weights_path = None
    if cfg.WEIGHTS_PATH.exists():
        weights_path = str(cfg.WEIGHTS_PATH / cfg.backbone / f"{cfg.backbone}.pth")
    
    model = BiomassModelWithTabular(cfg.backbone, n_tabular=2, pretrained=True, weights_path=weights_path)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(cfg.device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    best_score = -float('inf')
    
    for epoch in range(cfg.epochs):
        # Train
        model.train()
        train_loss = 0
        for imgs, tabular, targets in tqdm(train_loader, desc='Train', leave=False):
            imgs = imgs.to(cfg.device)
            tabular = tabular.to(cfg.device)
            targets = targets.to(cfg.device)
            
            optimizer.zero_grad()
            independent, _ = model(imgs, tabular)
            loss = F.mse_loss(independent, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Val
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, tabular, targets in val_loader:
                imgs = imgs.to(cfg.device)
                tabular = tabular.to(cfg.device)
                _, full_pred = model(imgs, tabular)
                all_preds.append(full_pred.cpu().numpy())
                
                green, clover, dead = targets[:, 0:1], targets[:, 1:2], targets[:, 2:3]
                gdm = green + clover
                total = gdm + dead
                full_targets = torch.cat([green, dead, clover, gdm, total], dim=1)
                all_targets.append(full_targets.numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        cv_score = competition_metric(all_targets, all_preds)
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{cfg.epochs} | Train: {train_loss:.2f} | CV: {cv_score:.4f}")
        
        if cv_score > best_score:
            best_score = cv_score
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'score': best_score,
                'tabular_scaler': tabular_scaler,
            }, cfg.OUTPUT_DIR / 'exp_b_biomass.pt')
            print(f"  ✓ Saved!")
    
    flush()
    return best_score, tabular_scaler

#%%
best_cv, biomass_tabular_scaler = train_biomass_with_tabular(train_wide, cfg)

#%% [markdown]
# ## Phase 3: Inference with Pseudo Labels

#%%
# Test dataset with pseudo tabular
test_dataset = BiomassDatasetWithTabular(
    test_wide, cfg, get_transforms('val', cfg.input_size), 'test', biomass_tabular_scaler
)
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

# Load model
ckpt = torch.load(cfg.OUTPUT_DIR / 'exp_b_biomass.pt', weights_only=False)
model = BiomassModelWithTabular(cfg.backbone, n_tabular=2, pretrained=False).to(cfg.device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f"✓ Loaded model (CV: {ckpt['score']:.4f})")

#%%
# Inference
all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Inference'):
        if len(batch) == 3:
            imgs, tabular, _ = batch
            imgs = imgs.to(cfg.device)
            tabular = tabular.to(cfg.device)
            _, full_pred = model(imgs, tabular)
        else:
            imgs, _ = batch
            imgs = imgs.to(cfg.device)
            _, full_pred = model(imgs, None)
        all_preds.append(full_pred.cpu().numpy())

preds = np.concatenate(all_preds)
print(f"Predictions shape: {preds.shape}")

#%%
# Submission
def melt_table(df):
    melted = df.melt(
        id_vars='image_path', value_vars=TARGET_ORDER,
        var_name='target_name', value_name='target'
    )
    melted['sample_id'] = (
        melted['image_path']
        .str.replace(r'^.*/', '', regex=True)
        .str.replace('.jpg', '', regex=False)
        + '__' + melted['target_name']
    )
    return melted[['sample_id', 'target']]

test_wide[TARGET_ORDER] = preds
test_wide[TARGET_ORDER] = test_wide[TARGET_ORDER].clip(lower=0)

submission = melt_table(test_wide)
submission.to_csv(cfg.OUTPUT_DIR / 'submission_exp_b.csv', index=False)

print(f"\n📄 Submission saved: {len(submission)} rows")
print(submission.head(10))

#%%
print(f"""
{'='*60}
🔬 Exp B: Pseudo-Labeling 완료
{'='*60}

CV Score: {best_cv:.4f} (1-fold quick test)

Pipeline:
1. Image → NDVI/Height 예측 모델 학습
2. Test 이미지에 pseudo NDVI/Height 생성
3. Image + Pseudo Tabular로 Biomass 예측

Output: {cfg.OUTPUT_DIR / 'submission_exp_b.csv'}
{'='*60}
""")
