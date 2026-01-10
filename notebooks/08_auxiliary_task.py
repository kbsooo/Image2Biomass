#%% [markdown]
# # 🔬 Exp A: Auxiliary Task Learning
#
# **아이디어**: 이미지에서 NDVI/Height도 예측하도록 학습
# → 모델이 "푸름", "높이" 개념을 이해하게 됨
#
# **결과**: Test에서는 이미지만 넣어도 이 지식이 반영됨

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
    # === Kaggle Paths ===
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    OUTPUT_DIR = Path("/kaggle/working")
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass")
    
    # === Model ===
    backbone = "efficientnet_b4"
    input_size = 384
    
    # === Training ===
    n_folds = 5
    train_folds = 1  # 싹수 확인용: 1-fold만 학습
    epochs = 10       # 빠른 테스트
    batch_size = 16   # GPU 2개 기준
    lr = 2e-4
    weight_decay = 1e-4
    
    # === Auxiliary Task ===
    aux_weight = 0.3  # aux_loss 가중치
    tabular_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
    
    # === Misc ===
    seed = 42
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === Targets ===
    independent_targets = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g']
    all_targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

cfg = CFG()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {cfg.device}")
print(f"Training {cfg.train_folds} fold(s) for quick test")

#%% [markdown]
# ## Competition Metric

#%%
TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5,
}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    weights = np.array([TARGET_WEIGHTS[t] for t in TARGET_ORDER])
    y_weighted_mean = sum(y_true[:, i].mean() * weights[i] for i in range(5))
    ss_res = sum(((y_true[:, i] - y_pred[:, i]) ** 2).mean() * weights[i] for i in range(5))
    ss_tot = sum(((y_true[:, i] - y_weighted_mean) ** 2).mean() * weights[i] for i in range(5))
    return 1 - ss_res / (ss_tot + 1e-8)

#%% [markdown]
# ## Data Preparation

#%%
def prepare_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
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
        for t in TARGET_ORDER:
            df_wide[t] = 0.0
    return df_wide

#%%
train_df = pd.read_csv(cfg.DATA_PATH / "train.csv")
train_wide = prepare_data(train_df, is_train=True)
train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)

kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
train_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(kf.split(train_wide)):
    train_wide.loc[val_idx, 'fold'] = fold

print(f"Train data shape: {train_wide.shape}")

# Tabular 통계
for col in cfg.tabular_cols:
    print(f"  {col}: min={train_wide[col].min():.2f}, max={train_wide[col].max():.2f}, mean={train_wide[col].mean():.2f}")

#%% [markdown]
# ## Augmentations

#%%
def get_transforms(mode: str = 'train', size: int = 384) -> A.Compose:
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

#%% [markdown]
# ## Dataset (Auxiliary Target 포함)

#%%
class BiomassDatasetWithAux(Dataset):
    """Biomass + Auxiliary targets (NDVI, Height)"""
    def __init__(
        self, 
        df: pd.DataFrame, 
        cfg, 
        transforms=None, 
        mode: str = 'train',
        tabular_scaler: Optional[StandardScaler] = None
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode
        
        # Auxiliary targets (NDVI, Height) - 학습에만 사용
        if mode in ['train', 'val'] and all(col in df.columns for col in cfg.tabular_cols):
            aux_data = df[cfg.tabular_cols].values.astype(np.float32)
            if tabular_scaler is not None:
                if mode == 'train':
                    self.aux_targets = tabular_scaler.fit_transform(aux_data)
                else:
                    self.aux_targets = tabular_scaler.transform(aux_data)
            else:
                self.aux_targets = aux_data
            self.has_aux = True
        else:
            self.aux_targets = None
            self.has_aux = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Image
        img_path = self.cfg.DATA_PATH / row['image_path']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        # Main targets (Green, Clover, Dead)
        main_targets = torch.tensor([
            row['Dry_Green_g'],
            row['Dry_Clover_g'],
            row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        # Auxiliary targets (NDVI, Height)
        if self.has_aux:
            aux_targets = torch.tensor(self.aux_targets[idx], dtype=torch.float32)
            return img, main_targets, aux_targets
        else:
            return img, main_targets

#%% [markdown]
# ## 🔑 Model with Auxiliary Head

#%%
class PhysicsConstrainedHead(nn.Module):
    """물리적 제약 조건을 만족하는 예측 헤드"""
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.3):
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.head(x)
        independent = self.softplus(raw)
        
        green = independent[:, 0:1]
        clover = independent[:, 1:2]
        dead = independent[:, 2:3]
        
        gdm = green + clover
        total = gdm + dead
        
        full = torch.cat([green, dead, clover, gdm, total], dim=1)
        return independent, full

#%%
class BiomassModelWithAux(nn.Module):
    """
    Auxiliary Task Learning 모델
    
    - Main Head: 바이오매스 예측
    - Aux Head: NDVI/Height 예측 (학습 시에만 사용)
    """
    def __init__(
        self, 
        backbone_name: str = "efficientnet_b4",
        n_aux: int = 2,  # NDVI, Height
        dropout: float = 0.3,
        pretrained: bool = True,
        weights_path: Optional[str] = None
    ):
        super().__init__()
        
        # Backbone
        if pretrained and weights_path and Path(weights_path).exists():
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
            weights = torch.load(weights_path, weights_only=True)
            weights = {k: v for k, v in weights.items() if not k.startswith('classifier')}
            self.backbone.load_state_dict(weights, strict=False)
            print(f"✓ Loaded pretrained weights from {weights_path}")
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        self.feat_dim = self.backbone.num_features
        
        # Main Head (바이오매스)
        self.main_head = PhysicsConstrainedHead(
            in_features=self.feat_dim, 
            hidden_dim=256, 
            dropout=dropout
        )
        
        # Auxiliary Head (NDVI, Height)
        self.aux_head = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_aux)
        )
    
    def forward(self, image: torch.Tensor, return_aux: bool = False):
        """
        Args:
            image: [B, C, H, W]
            return_aux: True면 aux 예측도 반환
        
        Returns:
            independent: [B, 3] - Green, Clover, Dead
            full: [B, 5] - All 5 targets
            aux: [B, 2] - NDVI, Height (optional)
        """
        feat = self.backbone(image)
        independent, full = self.main_head(feat)
        
        if return_aux:
            aux = self.aux_head(feat)
            return independent, full, aux
        return independent, full

#%% [markdown]
# ## Training

#%%
def train_one_epoch_with_aux(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: str,
    aux_weight: float = 0.3
) -> Tuple[float, float, float]:
    model.train()
    total_main_loss = 0
    total_aux_loss = 0
    
    for batch in tqdm(loader, desc='Train', leave=False):
        imgs, main_targets, aux_targets = batch
        imgs = imgs.to(device)
        main_targets = main_targets.to(device)
        aux_targets = aux_targets.to(device)
        
        optimizer.zero_grad()
        independent, _, aux_pred = model(imgs, return_aux=True)
        
        # Main loss (바이오매스)
        main_loss = F.mse_loss(independent, main_targets)
        
        # Aux loss (NDVI, Height)
        aux_loss = F.mse_loss(aux_pred, aux_targets)
        
        # Total loss
        loss = main_loss + aux_weight * aux_loss
        loss.backward()
        optimizer.step()
        
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item()
    
    n = len(loader)
    return total_main_loss / n, total_aux_loss / n, (total_main_loss + aux_weight * total_aux_loss) / n

@torch.no_grad()
def validate_with_aux(
    model: nn.Module, 
    loader: DataLoader, 
    device: str
) -> Tuple[float, float]:
    model.eval()
    all_preds, all_targets = [], []
    
    for batch in tqdm(loader, desc='Valid', leave=False):
        imgs, main_targets, _ = batch  # aux_targets는 여기선 무시
        imgs = imgs.to(device)
        
        _, full_pred = model(imgs, return_aux=False)
        all_preds.append(full_pred.cpu().numpy())
        
        # Full targets 재구성
        green = main_targets[:, 0:1]
        clover = main_targets[:, 1:2]
        dead = main_targets[:, 2:3]
        gdm = green + clover
        total = gdm + dead
        full_targets = torch.cat([green, dead, clover, gdm, total], dim=1)
        all_targets.append(full_targets.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    cv_score = competition_metric(all_targets, all_preds)
    mse = np.mean((all_preds - all_targets) ** 2)
    
    return mse, cv_score

#%%
def train_fold_with_aux(fold: int, train_df: pd.DataFrame, cfg) -> float:
    print(f"\n{'='*60}")
    print(f"🔬 Exp A: Auxiliary Task - Fold {fold}")
    print(f"{'='*60}")
    
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Tabular scaler (aux targets용)
    tabular_scaler = StandardScaler()
    
    train_dataset = BiomassDatasetWithAux(
        train_data, cfg, get_transforms('train', cfg.input_size), 'train', tabular_scaler
    )
    val_dataset = BiomassDatasetWithAux(
        val_data, cfg, get_transforms('val', cfg.input_size), 'val', tabular_scaler
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    
    # Model
    weights_path = None
    if cfg.WEIGHTS_PATH.exists():
        weights_path = str(cfg.WEIGHTS_PATH / cfg.backbone / f"{cfg.backbone}.pth")
    
    model = BiomassModelWithAux(
        backbone_name=cfg.backbone,
        n_aux=len(cfg.tabular_cols),
        dropout=0.3,
        pretrained=True,
        weights_path=weights_path
    )
    
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(cfg.device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    best_score = -float('inf')
    
    for epoch in range(cfg.epochs):
        main_loss, aux_loss, total_loss = train_one_epoch_with_aux(
            model, train_loader, optimizer, cfg.device, cfg.aux_weight
        )
        val_mse, cv_score = validate_with_aux(model, val_loader, cfg.device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{cfg.epochs} | LR: {lr:.6f} | "
              f"Main: {main_loss:.2f} | Aux: {aux_loss:.4f} | "
              f"Val MSE: {val_mse:.2f} | CV: {cv_score:.4f}")
        
        if cv_score > best_score:
            best_score = cv_score
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'fold': fold,
                'score': best_score,
                'tabular_scaler': tabular_scaler,
            }, cfg.OUTPUT_DIR / f'exp_a_fold{fold}.pt')
            print(f"  ✓ New best!")
    
    flush()
    return best_score

#%%
# Train (빠른 테스트: 1-fold만)
fold_scores = []
for fold in range(cfg.train_folds):
    score = train_fold_with_aux(fold, train_wide, cfg)
    fold_scores.append(score)
    print(f"Fold {fold} Best CV: {score:.4f}")

print(f"\n{'='*60}")
print(f"📊 Exp A Results: CV = {np.mean(fold_scores):.4f}")
print(f"{'='*60}")

#%% [markdown]
# ## Inference (Test)

#%%
@torch.no_grad()
def inference_exp_a(models: list, loader: DataLoader, device: str) -> np.ndarray:
    all_preds = []
    
    for batch in tqdm(loader, desc='Inference'):
        if len(batch) == 3:
            imgs, _, _ = batch
        else:
            imgs, _ = batch
        imgs = imgs.to(device)
        
        batch_preds = []
        for model in models:
            model.eval()
            _, full_pred = model(imgs, return_aux=False)
            batch_preds.append(full_pred.cpu().numpy())
        
        avg_pred = np.mean(batch_preds, axis=0)
        all_preds.append(avg_pred)
    
    return np.concatenate(all_preds)

#%%
# Load test data
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_wide = prepare_data(test_df, is_train=False)
print(f"Test data: {len(test_wide)} images")

#%%
# Load models
models = []
for fold in range(cfg.train_folds):
    ckpt_path = cfg.OUTPUT_DIR / f'exp_a_fold{fold}.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        
        model = BiomassModelWithAux(
            backbone_name=cfg.backbone,
            n_aux=len(cfg.tabular_cols),
            pretrained=False
        ).to(cfg.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"✓ Loaded fold {fold} (CV: {ckpt['score']:.4f})")

print(f"\nLoaded {len(models)} models")

#%%
# Test dataset (aux 없이)
test_dataset = BiomassDatasetWithAux(
    test_wide, cfg, get_transforms('val', cfg.input_size), 'test', None
)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=True
)

# Inference
preds = inference_exp_a(models, test_loader, cfg.device)
print(f"Predictions shape: {preds.shape}")

#%%
# Submission
def melt_table(df: pd.DataFrame) -> pd.DataFrame:
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
    return melted[['sample_id', 'image_path', 'target_name', 'target']]

test_wide[TARGET_ORDER] = preds
test_wide[TARGET_ORDER] = test_wide[TARGET_ORDER].clip(lower=0)

submission = melt_table(test_wide)
submission = submission[['sample_id', 'target']]
submission.to_csv(cfg.OUTPUT_DIR / 'submission_exp_a.csv', index=False)

print(f"\n📄 Submission saved: {len(submission)} rows")
print(submission.head(10))

#%%
print(f"""
{'='*60}
🔬 Exp A: Auxiliary Task 완료
{'='*60}

CV Score: {np.mean(fold_scores):.4f} (1-fold quick test)

핵심 확인 사항:
1. Aux loss가 감소하는가? (모델이 NDVI/Height를 학습하고 있는가?)
2. Main CV가 baseline보다 좋은가?

Output: {cfg.OUTPUT_DIR / 'submission_exp_a.csv'}
{'='*60}
""")
