#%% [markdown]
# # 🧠 Hybrid Approach: DINOv2 + Knowledge Distillation
#
# **전략**:
# 1. Phase 1: Teacher Model 학습 (Image + Tabular → Biomass)
# 2. Phase 2: Soft Targets 생성
# 3. Phase 3: Student Model 학습 (Image only, KD Loss)
# 4. Phase 4: Inference & Submission
#
# **Key Features**:
# - DINOv2 ViT-B/14 backbone (self-supervised, 142M images)
# - FiLM (Feature-wise Linear Modulation) for fusion
# - Zero-Inflated Regression for Clover
# - Physics Constraints (GDM = Green + Clover, Total = GDM + Dead)

#%%
import os
import gc
import math
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

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
    # === Paths (Kaggle) ===
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    OUTPUT_DIR = Path("/kaggle/working")
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass")
    
    # === Model ===
    backbone = "dinov2_vitb14"  # DINOv2 ViT-B/14
    backbone_dim = 768  # ViT-B/14 output dim
    input_size = 518    # DINOv2 optimal (divisible by 14)
    freeze_backbone = False  # Unfreeze DINOv2 for domain adaptation
    
    # === Teacher ===
    teacher_epochs = 15
    teacher_lr = 2e-4
    
    # === Student ===
    student_epochs = 20
    student_lr = 1e-4
    kd_alpha = 0.5  # Balance hard/soft loss
    
    # === Training ===
    n_folds = 5
    batch_size = 8  # DINOv2 is larger
    weight_decay = 1e-4
    
    # === Tabular ===
    tabular_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
    
    # === Misc ===
    seed = 42
    num_workers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {cfg.device}")
print(f"Backbone: {cfg.backbone}")
print(f"Folds: {cfg.n_folds}, Teacher epochs: {cfg.teacher_epochs}, Student epochs: {cfg.student_epochs}")

#%% [markdown]
# ## Competition Metric

#%%
TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1,
    'GDM_g': 0.2, 'Dry_Total_g': 0.5,
}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted R² score."""
    weights = np.array([TARGET_WEIGHTS[t] for t in TARGET_ORDER])
    total_r2 = 0.0
    for i in range(5):
        ss_res = ((y_true[:, i] - y_pred[:, i]) ** 2).sum()
        ss_tot = ((y_true[:, i] - y_true[:, i].mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        total_r2 += weights[i] * r2
    return total_r2

#%% [markdown]
# ## Data Preparation

#%%
def prepare_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Pivot long format to wide format."""
    if 'target' in df.columns:
        df_wide = pd.pivot_table(
            df, values='target',
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
            columns='target_name', aggfunc='mean'
        ).reset_index()
    else:
        df = df.copy()
        cols = ['image_path']
        for col in ['Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']:
            if col in df.columns:
                cols.append(col)
        df_wide = df.drop_duplicates(subset=['image_path'])[cols].reset_index(drop=True)
    return df_wide

def get_transforms(mode: str = 'train', size: int = 518) -> A.Compose:
    if mode == 'train':
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05, p=0.7),
            A.RandomBrightnessContrast(p=0.5),
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
# Load data
train_df = pd.read_csv(cfg.DATA_PATH / "train.csv")
train_wide = prepare_data(train_df, is_train=True)
train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)

# Geographic Stratified KFold (by State)
skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
train_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_wide, train_wide['State'])):
    train_wide.loc[val_idx, 'fold'] = fold

# Encode categoricals
state_encoder = LabelEncoder()
species_encoder = LabelEncoder()
train_wide['state_encoded'] = state_encoder.fit_transform(train_wide['State'])
train_wide['species_encoded'] = species_encoder.fit_transform(train_wide['Species'])

# Month from date
train_wide['month'] = pd.to_datetime(train_wide['Sampling_Date']).dt.month

print(f"Train data: {len(train_wide)} images")
print(f"States: {train_wide['State'].nunique()}")
print(f"Species: {train_wide['Species'].nunique()}")
print(f"Fold distribution:\n{train_wide['fold'].value_counts().sort_index()}")

#%% [markdown]
# ## Model Components

#%%
class DINOv2Backbone(nn.Module):
    """DINOv2 ViT-B/14 backbone with offline weight loading."""
    def __init__(self, model_name: str = "dinov2_vitb14", freeze: bool = True, weights_path: Path = None):
        super().__init__()
        self.feat_dim = 768  # ViT-B/14
        
        # Try loading from local weights (Kaggle offline)
        # Path format: {WEIGHTS_PATH}/dinov2/dinov2_vitb14.pth
        weight_file = None
        if weights_path and weights_path.exists():
            weight_file = weights_path / "dinov2" / f"{model_name}.pth"
        
        if weight_file and weight_file.exists():
            # Load DINOv2 architecture from timm
            self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0)
            state_dict = torch.load(weight_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"✓ DINOv2 loaded from: {weight_file}")
        else:
            # Fallback: won't work offline!
            print(f"⚠ Local weights not found at {weight_file}, trying online...")
            self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        
        print(f"✓ DINOv2 backbone: feat_dim={self.feat_dim}")
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ DINOv2 backbone frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TabularEncoder(nn.Module):
    """Encode tabular features: NDVI, Height, State, Species, Month."""
    def __init__(self, n_states: int = 4, n_species: int = 30, embed_dim: int = 16, output_dim: int = 128):
        super().__init__()
        self.cont_bn = nn.BatchNorm1d(2)
        self.cont_linear = nn.Linear(2, 64)
        self.state_embed = nn.Embedding(n_states, embed_dim)
        self.species_embed = nn.Embedding(n_species, embed_dim)
        self.month_linear = nn.Linear(2, embed_dim)  # sin/cos
        
        self.fusion = nn.Sequential(
            nn.Linear(64 + embed_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, continuous, state, species, month):
        cont = self.cont_linear(self.cont_bn(continuous))
        state_emb = self.state_embed(state)
        species_emb = self.species_embed(species)
        
        month_rad = (month.float() - 1) / 12 * 2 * math.pi
        month_enc = self.month_linear(torch.stack([torch.sin(month_rad), torch.cos(month_rad)], dim=-1))
        
        all_feat = torch.cat([cont, state_emb, species_emb, month_enc], dim=-1)
        return self.fusion(all_feat)


class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation."""
    def __init__(self, img_dim: int, tab_dim: int):
        super().__init__()
        self.gamma_net = nn.Sequential(nn.Linear(tab_dim, img_dim), nn.Tanh())
        self.beta_net = nn.Linear(tab_dim, img_dim)
    
    def forward(self, img_feat, tab_feat):
        gamma = self.gamma_net(tab_feat)
        beta = self.beta_net(tab_feat)
        return img_feat * (1 + gamma) + beta


class ZeroInflatedHead(nn.Module):
    """Two-stage prediction for zero-inflated Clover."""
    def __init__(self, in_features: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Softplus()
        )
    
    def forward(self, x):
        p = self.classifier(x)
        amount = self.regressor(x)
        return p, amount, p * amount


class PhysicsHead(nn.Module):
    """Physics-constrained prediction head."""
    def __init__(self, in_features: int, use_zero_inflated: bool = True):
        super().__init__()
        self.use_zero_inflated = use_zero_inflated
        
        self.shared = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.green_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        self.dead_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        
        if use_zero_inflated:
            self.clover_head = ZeroInflatedHead(128)
        else:
            self.clover_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
    
    def forward(self, x):
        feat = self.shared(x)
        green = self.green_head(feat)
        dead = self.dead_head(feat)
        
        if self.use_zero_inflated:
            clover_p, clover_amt, clover = self.clover_head(feat)
        else:
            clover = self.clover_head(feat)
            clover_p, clover_amt = None, None
        
        gdm = green + clover
        total = gdm + dead
        
        # [Green, Dead, Clover, GDM, Total]
        full = torch.cat([green, dead, clover, gdm, total], dim=1)
        independent = torch.cat([green, clover, dead], dim=1)
        
        return {'full': full, 'independent': independent, 'clover_p': clover_p, 'clover_amt': clover_amt}


class TeacherModel(nn.Module):
    """Teacher: Image + Tabular → Biomass"""
    def __init__(self, cfg, n_states, n_species):
        super().__init__()
        self.backbone = DINOv2Backbone(cfg.backbone, cfg.freeze_backbone, cfg.WEIGHTS_PATH)
        self.tabular_encoder = TabularEncoder(n_states, n_species)
        self.fusion = FiLMFusion(cfg.backbone_dim, 128)
        self.head = PhysicsHead(cfg.backbone_dim)
    
    def forward(self, image, continuous, state, species, month):
        img_feat = self.backbone(image)
        tab_feat = self.tabular_encoder(continuous, state, species, month)
        fused = self.fusion(img_feat, tab_feat)
        return self.head(fused), fused


class StudentModel(nn.Module):
    """Student: Image only → Biomass"""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = DINOv2Backbone(cfg.backbone, cfg.freeze_backbone, cfg.WEIGHTS_PATH)
        self.head = PhysicsHead(cfg.backbone_dim)
    
    def forward(self, image):
        img_feat = self.backbone(image)
        return self.head(img_feat), img_feat

#%% [markdown]
# ## Datasets

#%%
class TeacherDataset(Dataset):
    """Dataset for Teacher (Image + Tabular → Biomass)."""
    def __init__(self, df, cfg, transforms=None, tabular_scaler=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
        
        # Scale tabular
        tabular = df[cfg.tabular_cols].values.astype(np.float32)
        if tabular_scaler is not None:
            if mode == 'train':
                self.tabular = tabular_scaler.fit_transform(tabular)
            else:
                self.tabular = tabular_scaler.transform(tabular)
        else:
            self.tabular = tabular
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img = cv2.imread(str(self.cfg.DATA_PATH / row['image_path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        continuous = torch.tensor(self.tabular[idx], dtype=torch.float32)
        state = torch.tensor(row['state_encoded'], dtype=torch.long)
        species = torch.tensor(row['species_encoded'], dtype=torch.long)
        month = torch.tensor(row['month'], dtype=torch.long)
        
        targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Clover_g'], row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        full_targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Dead_g'], row['Dry_Clover_g'],
            row['GDM_g'], row['Dry_Total_g']
        ], dtype=torch.float32)
        
        return {
            'image': img,
            'continuous': continuous,
            'state': state,
            'species': species,
            'month': month,
            'targets': targets,
            'full_targets': full_targets,
            'image_id': row['image_id']
        }


class StudentDataset(Dataset):
    """Dataset for Student (Image only)."""
    def __init__(self, df, cfg, transforms=None, soft_targets=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
        self.soft_targets = soft_targets
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img = cv2.imread(str(self.cfg.DATA_PATH / row['image_path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Clover_g'], row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        full_targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Dead_g'], row['Dry_Clover_g'],
            row['GDM_g'], row['Dry_Total_g']
        ], dtype=torch.float32)
        
        result = {
            'image': img,
            'targets': targets,
            'full_targets': full_targets,
            'image_id': row['image_id']
        }
        
        if self.soft_targets and row['image_id'] in self.soft_targets:
            result['soft_target'] = self.soft_targets[row['image_id']]
        
        return result

#%% [markdown]
# ## Training Functions

#%%
def train_teacher_fold(fold: int, train_df: pd.DataFrame, cfg, n_states, n_species) -> Tuple[float, Dict]:
    print(f"\n{'='*60}")
    print(f"🎓 PHASE 1: Teacher Model - Fold {fold}")
    print(f"{'='*60}")
    
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    tabular_scaler = StandardScaler()
    
    train_ds = TeacherDataset(train_data, cfg, get_transforms('train', cfg.input_size), tabular_scaler, 'train')
    val_ds = TeacherDataset(val_data, cfg, get_transforms('val', cfg.input_size), tabular_scaler, 'val')
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    
    model = TeacherModel(cfg, n_states, n_species)
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(cfg.device)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.teacher_lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.teacher_epochs)
    scaler = GradScaler()
    
    best_score = -float('inf')
    best_state = None
    
    for epoch in range(cfg.teacher_epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Teacher E{epoch+1}', leave=False):
            imgs = batch['image'].to(cfg.device)
            cont = batch['continuous'].to(cfg.device)
            state = batch['state'].to(cfg.device)
            species = batch['species'].to(cfg.device)
            month = batch['month'].to(cfg.device)
            targets = batch['targets'].to(cfg.device)
            
            optimizer.zero_grad()
            with autocast():
                output, _ = model(imgs, cont, state, species, month)
                loss = F.mse_loss(output['independent'], targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(cfg.device)
                cont = batch['continuous'].to(cfg.device)
                state = batch['state'].to(cfg.device)
                species = batch['species'].to(cfg.device)
                month = batch['month'].to(cfg.device)
                full_targets = batch['full_targets']
                
                output, _ = model(imgs, cont, state, species, month)
                all_preds.append(output['full'].cpu().numpy())
                all_targets.append(full_targets.numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        cv_score = competition_metric(all_targets, all_preds)
        
        print(f"Epoch {epoch+1}/{cfg.teacher_epochs} | Loss: {train_loss:.4f} | CV: {cv_score:.4f}")
        
        if cv_score > best_score:
            best_score = cv_score
            model_to_save = model.module if hasattr(model, 'module') else model
            best_state = {k: v.cpu() for k, v in model_to_save.state_dict().items()}
            torch.save({
                'model_state_dict': best_state,
                'score': best_score,
                'tabular_scaler': tabular_scaler,
            }, cfg.OUTPUT_DIR / f'teacher_fold{fold}.pt')
            print(f"  ✓ New best!")
    
    flush()
    return best_score, tabular_scaler


@torch.no_grad()
def generate_soft_targets(model, loader, cfg):
    """Generate soft targets from teacher."""
    model.eval()
    soft_targets = {}
    
    for batch in tqdm(loader, desc='Generating Soft Targets'):
        imgs = batch['image'].to(cfg.device)
        cont = batch['continuous'].to(cfg.device)
        state = batch['state'].to(cfg.device)
        species = batch['species'].to(cfg.device)
        month = batch['month'].to(cfg.device)
        image_ids = batch['image_id']
        
        output, _ = model(imgs, cont, state, species, month)
        
        for i, img_id in enumerate(image_ids):
            soft_targets[img_id] = output['independent'][i].cpu()
    
    return soft_targets


def train_student_fold(fold: int, train_df: pd.DataFrame, soft_targets: Dict, cfg) -> float:
    print(f"\n{'='*60}")
    print(f"🎯 PHASE 2: Student Model (KD) - Fold {fold}")
    print(f"{'='*60}")
    
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    train_ds = StudentDataset(train_data, cfg, get_transforms('train', cfg.input_size), soft_targets)
    val_ds = StudentDataset(val_data, cfg, get_transforms('val', cfg.input_size))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    
    model = StudentModel(cfg)
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(cfg.device)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.student_lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.student_epochs)
    scaler = GradScaler()
    
    best_score = -float('inf')
    
    for epoch in range(cfg.student_epochs):
        model.train()
        train_loss, hard_loss_sum, soft_loss_sum = 0, 0, 0
        
        for batch in tqdm(train_loader, desc=f'Student E{epoch+1}', leave=False):
            imgs = batch['image'].to(cfg.device)
            targets = batch['targets'].to(cfg.device)
            
            # Get soft targets
            soft = torch.stack([batch['soft_target'][i] for i in range(len(batch['image_id']))]).to(cfg.device)
            
            optimizer.zero_grad()
            with autocast():
                output, _ = model(imgs)
                pred = output['independent']
                
                hard_loss = F.mse_loss(pred, targets)
                soft_loss = F.mse_loss(pred, soft)
                loss = cfg.kd_alpha * hard_loss + (1 - cfg.kd_alpha) * soft_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            hard_loss_sum += hard_loss.item()
            soft_loss_sum += soft_loss.item()
        
        n = len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(cfg.device)
                full_targets = batch['full_targets']
                
                output, _ = model(imgs)
                all_preds.append(output['full'].cpu().numpy())
                all_targets.append(full_targets.numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        cv_score = competition_metric(all_targets, all_preds)
        
        print(f"Epoch {epoch+1}/{cfg.student_epochs} | Loss: {train_loss/n:.4f} (H:{hard_loss_sum/n:.4f} S:{soft_loss_sum/n:.4f}) | CV: {cv_score:.4f}")
        
        if cv_score > best_score:
            best_score = cv_score
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'model_state_dict': {k: v.cpu() for k, v in model_to_save.state_dict().items()},
                'score': best_score,
            }, cfg.OUTPUT_DIR / f'student_fold{fold}.pt')
            print(f"  ✓ New best!")
    
    flush()
    return best_score

#%% [markdown]
# ## Training Pipeline

#%%
n_states = train_wide['state_encoded'].nunique()
n_species = train_wide['species_encoded'].nunique()
print(f"n_states: {n_states}, n_species: {n_species}")

teacher_scores = []
student_scores = []
all_soft_targets = {}

for fold in range(cfg.n_folds):
    # Phase 1: Train Teacher
    teacher_score, tabular_scaler = train_teacher_fold(fold, train_wide, cfg, n_states, n_species)
    teacher_scores.append(teacher_score)
    
    # Load teacher for soft target generation
    ckpt = torch.load(cfg.OUTPUT_DIR / f'teacher_fold{fold}.pt', weights_only=False)
    teacher = TeacherModel(cfg, n_states, n_species).to(cfg.device)
    teacher.load_state_dict(ckpt['model_state_dict'])
    
    # Generate soft targets for training data
    train_data = train_wide[train_wide['fold'] != fold].reset_index(drop=True)
    train_ds = TeacherDataset(train_data, cfg, get_transforms('val', cfg.input_size), tabular_scaler, 'val')
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)
    
    fold_soft_targets = generate_soft_targets(teacher, train_loader, cfg)
    all_soft_targets.update(fold_soft_targets)
    print(f"Generated {len(fold_soft_targets)} soft targets")
    
    del teacher
    flush()
    
    # Phase 2: Train Student with KD
    student_score = train_student_fold(fold, train_wide, all_soft_targets, cfg)
    student_scores.append(student_score)
    
    print(f"Fold {fold} | Teacher CV: {teacher_score:.4f} | Student CV: {student_score:.4f}")

print(f"\n{'='*60}")
print(f"📊 Overall Results")
print(f"{'='*60}")
print(f"Teacher CV: {np.mean(teacher_scores):.4f} ± {np.std(teacher_scores):.4f}")
print(f"Student CV: {np.mean(student_scores):.4f} ± {np.std(student_scores):.4f}")

#%% [markdown]
# ## Inference

#%%
# Load test data
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_wide = prepare_data(test_df, is_train=False)
test_wide['image_id'] = test_wide['image_path'].apply(lambda x: Path(x).stem)
print(f"\nTest data: {len(test_wide)} images")

# Create test dataset (image only for student)
class TestDataset(Dataset):
    def __init__(self, df, cfg, transforms):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.cfg.DATA_PATH / row['image_path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, row['image_id']

test_ds = TestDataset(test_wide, cfg, get_transforms('val', cfg.input_size))
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

#%%
# Load student models and inference
models = []
for fold in range(cfg.n_folds):
    ckpt = torch.load(cfg.OUTPUT_DIR / f'student_fold{fold}.pt', weights_only=False)
    model = StudentModel(cfg).to(cfg.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    models.append(model)
    print(f"✓ Loaded student fold {fold} (CV: {ckpt['score']:.4f})")

#%%
@torch.no_grad()
def inference(models, loader, device):
    all_preds = []
    all_ids = []
    
    for imgs, img_ids in tqdm(loader, desc='Inference'):
        imgs = imgs.to(device)
        batch_preds = []
        
        for model in models:
            output, _ = model(imgs)
            batch_preds.append(output['full'].cpu().numpy())
        
        avg_pred = np.mean(batch_preds, axis=0)
        all_preds.append(avg_pred)
        all_ids.extend(img_ids)
    
    return np.concatenate(all_preds), all_ids

preds, image_ids = inference(models, test_loader, cfg.device)
print(f"Predictions shape: {preds.shape}")

#%%
# Create submission
submission_rows = []
for i, img_id in enumerate(image_ids):
    for j, target in enumerate(TARGET_ORDER):
        submission_rows.append({
            'sample_id': f"{img_id}__{target}",
            'target': max(0, preds[i, j])  # Ensure non-negative
        })

submission = pd.DataFrame(submission_rows)
submission.to_csv(cfg.OUTPUT_DIR / 'submission.csv', index=False)

# Verify physics constraints
print(f"\n✓ Physics constraint check:")
for i, img_id in enumerate(image_ids):
    green, dead, clover, gdm, total = preds[i]
    print(f"  {img_id}: GDM=G+C: {np.isclose(gdm, green+clover)}, Total=GDM+D: {np.isclose(total, gdm+dead)}")

print(f"\n📄 Submission saved: {len(submission)} rows")
print(submission.head(10))

#%%
print(f"""
{'='*60}
🏆 Hybrid Approach Complete!
{'='*60}

Pipeline:
1. Teacher (Image + Tabular) trained with FiLM fusion
2. Soft targets generated for Knowledge Distillation
3. Student (Image only) trained with KD loss

Results:
  Teacher CV: {np.mean(teacher_scores):.4f} ± {np.std(teacher_scores):.4f}
  Student CV: {np.mean(student_scores):.4f} ± {np.std(student_scores):.4f}

Output: {cfg.OUTPUT_DIR / 'submission.csv'}
{'='*60}
""")
