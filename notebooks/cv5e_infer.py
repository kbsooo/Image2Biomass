#%% [markdown]
# # 🚀 CV5E: DINOv3 + ConvNeXt Multi-Backbone Ensemble
#
# **Phase A 전략**: Transformer + CNN 혼합 앙상블
#
# **구성**:
# - v27 (DINOv3): v20, v22, v23, v25, v26 모델들
# - cv5 (ConvNeXt-Base): 5-fold
#
# **가중치**: v27 70% + cv5 30%
# **Expected**: LB 0.71~0.73

#%%
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import timm

tqdm.pandas()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

#%%
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything(42)

#%% [markdown]
# ## ⚙️ Configuration

#%%
class CFG:
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    
    # DINOv3 backbone weights
    BACKBONE_WEIGHTS = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large/dinov3_vitl16_qkvb.pth")
    
    # v27 모델들 (DINOv3 기반, 5개 버전)
    V27_MODELS = {
        'v20': Path("/kaggle/input/csiro-v20-models"),
        'v22': Path("/kaggle/input/csiro-v22-models"),
        'v23': Path("/kaggle/input/csiro-v23-models"),
        'v25': Path("/kaggle/input/csiro-v25-models"),
        'v26': Path("/kaggle/input/csiro-v26-models"),
    }
    
    # cv5 모델 (ConvNeXt 기반)
    CV5_MODELS_DIR = Path("/kaggle/input/csiro-cv5-models")
    
    # ⭐ 앙상블 가중치 (핵심!)
    V27_WEIGHT = 0.7   # DINOv3가 성능 좋으므로 높게
    CV5_WEIGHT = 0.3   # ConvNeXt는 다양성 목적
    
    # DINOv3 모델 설정 (v27용)
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size_v27 = (512, 512)
    
    # ConvNeXt 모델 설정 (cv5용)
    backbone_name_cv5 = "convnext_base.fb_in22k_ft_in1k"
    img_size_cv5 = (560, 560)
    
    # 공통 Head 설정
    hidden_dim = 512
    num_layers = 3
    dropout = 0.1
    use_layernorm = True
    veg_feat_dim = 128  # v25용
    
    # Inference
    batch_size = 8
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TTA 비활성화 (효과 없음 확인됨)
    use_tta = False

cfg = CFG()

TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

print(f"Device: {cfg.device}")
print(f"v27 weight: {cfg.V27_WEIGHT}, cv5 weight: {cfg.CV5_WEIGHT}")

#%% [markdown]
# ## 📊 Dataset

#%%
class TestDataset(Dataset):
    """기본 Dataset (v20, v22, v23, v26, cv5용)"""
    def __init__(self, df, cfg, transform=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.cfg.DATA_PATH / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        left_img = img.crop((0, 0, mid, height))
        right_img = img.crop((mid, 0, width, height))
        
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        return left_img, right_img, row['sample_id_prefix']


def compute_vegetation_indices(img_array):
    """RGB 이미지에서 Vegetation Index 계산"""
    img = img_array.astype(np.float32) / 255.0
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    exg = 2*g - r - b
    exg = (exg + 2) / 4
    
    gr_ratio = g / (r + 1e-8)
    gr_ratio = np.clip(gr_ratio, 0, 3) / 3
    
    return np.stack([exg, gr_ratio], axis=-1)


class TestDatasetV25(Dataset):
    """v25용 Dataset (Vegetation Index 이미지 포함)"""
    def __init__(self, df, cfg, transform=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
        self.img_size = cfg.img_size_v27
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.cfg.DATA_PATH / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        left_pil = img.crop((0, 0, mid, height))
        right_pil = img.crop((mid, 0, width, height))
        
        # Resize & numpy
        left_pil = left_pil.resize(self.img_size)
        right_pil = right_pil.resize(self.img_size)
        
        left_np = np.array(left_pil)
        right_np = np.array(right_pil)
        
        # Vegetation Index
        left_veg = compute_vegetation_indices(left_np)
        right_veg = compute_vegetation_indices(right_np)
        
        # RGB Transform
        if self.transform:
            left_rgb = self.transform(left_pil)
            right_rgb = self.transform(right_pil)
        
        # Veg to Tensor
        left_veg = torch.from_numpy(left_veg).permute(2, 0, 1).float()
        right_veg = torch.from_numpy(right_veg).permute(2, 0, 1).float()
        
        return left_rgb, right_rgb, left_veg, right_veg, row['sample_id_prefix']


def get_transform_v27(cfg):
    return T.Compose([
        T.Resize(cfg.img_size_v27),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_transform_cv5(cfg):
    return T.Compose([
        T.Resize(cfg.img_size_cv5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_transform_v25():
    """v25용 Transform (Resize 없음 - Dataset에서 이미 resize)"""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%% [markdown]
# ## 🧠 Models

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


class CSIROModelBase(nn.Module):
    """v20/v23/v26 모델 (DINOv3 + FiLM)"""
    def __init__(self, cfg, backbone_weights_path=None):
        super().__init__()
        
        if backbone_weights_path and Path(backbone_weights_path).exists():
            self.backbone = timm.create_model(cfg.model_name, pretrained=False, 
                                               num_classes=0, global_pool='avg')
            state = torch.load(backbone_weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        else:
            self.backbone = timm.create_model(cfg.model_name, pretrained=True, 
                                               num_classes=0, global_pool='avg')
        
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
            nn.Linear(combined_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.head_ndvi = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, 1)
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
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)


class CSIROModelV22(nn.Module):
    """v22 모델 (hidden_dim=256, num_layers=2, no aux)"""
    def __init__(self, cfg, backbone_weights_path=None):
        super().__init__()
        
        if backbone_weights_path and Path(backbone_weights_path).exists():
            self.backbone = timm.create_model(cfg.model_name, pretrained=False, 
                                               num_classes=0, global_pool='avg')
            state = torch.load(backbone_weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        else:
            self.backbone = timm.create_model(cfg.model_name, pretrained=True, 
                                               num_classes=0, global_pool='avg')
        
        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        
        self.film = FiLM(feat_dim)
        
        # v22 specific: hidden_dim=256, num_layers=2
        self.head_green = make_head(combined_dim, 256, 2, cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, 256, 2, cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, 256, 2, cfg.dropout, cfg.use_layernorm)
        
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


class VegetationEncoder(nn.Module):
    """v25 Vegetation Encoder (Conv2d 기반)"""
    def __init__(self, in_channels=2, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.encoder(x)


class CSIROModelV25(nn.Module):
    """v25 모델 (Vegetation Index Late Fusion)"""
    def __init__(self, cfg, backbone_weights_path=None):
        super().__init__()
        
        if backbone_weights_path and Path(backbone_weights_path).exists():
            self.backbone = timm.create_model(cfg.model_name, pretrained=False, 
                                               num_classes=0, global_pool='avg')
            state = torch.load(backbone_weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        else:
            self.backbone = timm.create_model(cfg.model_name, pretrained=True, 
                                               num_classes=0, global_pool='avg')
        
        feat_dim = self.backbone.num_features
        
        self.veg_encoder = VegetationEncoder(in_channels=2, out_dim=cfg.veg_feat_dim)
        self.film = FiLM(feat_dim)
        
        combined_dim = feat_dim * 2 + cfg.veg_feat_dim * 2
        
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                    cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                     cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                   cfg.dropout, cfg.use_layernorm)
        
        self.head_height = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.head_ndvi = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, 1)
        )
        
        self.softplus = nn.Softplus(beta=1.0)
    
    def forward(self, left_rgb, right_rgb, left_veg, right_veg):
        left_feat = self.backbone(left_rgb)
        right_feat = self.backbone(right_rgb)
        
        left_veg_feat = self.veg_encoder(left_veg)
        right_veg_feat = self.veg_encoder(right_veg)
        
        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)
        
        left_mod = left_feat * (1 + gamma) + beta
        right_mod = right_feat * (1 + gamma) + beta
        
        combined = torch.cat([left_mod, right_mod, left_veg_feat, right_veg_feat], dim=1)
        
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))
        
        gdm = green + clover
        total = gdm + dead
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)


class CSIROModelCV5(nn.Module):
    """cv5 모델: ConvNeXt-Base (no FiLM)"""
    def __init__(self, cfg):
        super().__init__()
        
        self.backbone = timm.create_model(
            cfg.backbone_name_cv5,
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
        
        feat_dim = self.backbone.num_features  # 1024
        combined_dim = feat_dim * 2
        
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, 
                                    cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                     cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                   cfg.dropout, cfg.use_layernorm)
        
        self.head_height = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.head_ndvi = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, 1)
        )
        
        self.softplus = nn.Softplus(beta=1.0)
    
    def forward(self, left_img, right_img):
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)
        
        # Simple concatenation (no FiLM for CNN)
        combined = torch.cat([left_feat, right_feat], dim=1)
        
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))
        
        gdm = green + clover
        total = gdm + dead
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## 🔮 Prediction Functions

#%%
@torch.no_grad()
def predict_base(model, loader, device):
    """기본 모델 예측 (v20, v22, v23, v26)"""
    model.eval()
    all_outputs, all_ids = [], []
    
    for left, right, ids in tqdm(loader, desc="Predicting"):
        left, right = left.to(device), right.to(device)
        pred = model(left, right)
        all_outputs.append(pred.cpu().numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids


@torch.no_grad()
def predict_v25(model, loader, device):
    """v25 모델 예측 (4-input)"""
    model.eval()
    all_outputs, all_ids = [], []
    
    for left_rgb, right_rgb, left_veg, right_veg, ids in tqdm(loader, desc="Predicting v25"):
        left_rgb = left_rgb.to(device)
        right_rgb = right_rgb.to(device)
        left_veg = left_veg.to(device)
        right_veg = right_veg.to(device)
        pred = model(left_rgb, right_rgb, left_veg, right_veg)
        all_outputs.append(pred.cpu().numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids


def predict_v27_all(test_wide, cfg, device):
    """v27의 모든 버전 예측 (v20, v22, v23, v25, v26)"""
    all_preds = []
    n_models = 0
    
    # v27용 기본 DataLoader (512x512)
    transform_v27 = get_transform_v27(cfg)
    dataset_v27 = TestDataset(test_wide, cfg, transform_v27)
    loader_v27 = DataLoader(dataset_v27, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    
    # v25용 DataLoader (Vegetation Index)
    transform_v25 = get_transform_v25()
    dataset_v25 = TestDatasetV25(test_wide, cfg, transform_v25)
    loader_v25 = DataLoader(dataset_v25, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    
    for version, model_dir in cfg.V27_MODELS.items():
        if not model_dir.exists():
            print(f"⚠️ {version} not found: {model_dir}")
            continue
        
        model_files = sorted(model_dir.glob("model_fold*.pth"))
        if not model_files:
            print(f"⚠️ No models in {version}")
            continue
        
        print(f"\n=== {version}: {len(model_files)} models ===")
        
        for model_file in model_files:
            print(f"  Loading {model_file.name}...")
            
            # 버전별 모델 클래스 선택
            if version == 'v22':
                model = CSIROModelV22(cfg, cfg.BACKBONE_WEIGHTS).to(device)
                model.load_state_dict(torch.load(model_file, map_location=device))
                preds, _ = predict_base(model, loader_v27, device)
            elif version == 'v25':
                model = CSIROModelV25(cfg, cfg.BACKBONE_WEIGHTS).to(device)
                model.load_state_dict(torch.load(model_file, map_location=device))
                preds, _ = predict_v25(model, loader_v25, device)
            else:  # v20, v23, v26
                model = CSIROModelBase(cfg, cfg.BACKBONE_WEIGHTS).to(device)
                model.load_state_dict(torch.load(model_file, map_location=device))
                preds, _ = predict_base(model, loader_v27, device)
            
            all_preds.append(preds)
            n_models += 1
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
    
    print(f"\n✓ v27 total: {n_models} models")
    return np.mean(all_preds, axis=0) if all_preds else None


def predict_cv5_all(test_wide, cfg, device):
    """cv5 모델 예측 (ConvNeXt 5-fold)"""
    all_preds = []
    
    if not cfg.CV5_MODELS_DIR.exists():
        print(f"⚠️ cv5 not found: {cfg.CV5_MODELS_DIR}")
        return None
    
    model_files = sorted(cfg.CV5_MODELS_DIR.glob("model_fold*.pth"))
    print(f"\n=== cv5: {len(model_files)} models ===")
    
    # cv5용 DataLoader (560x560)
    transform_cv5 = get_transform_cv5(cfg)
    dataset_cv5 = TestDataset(test_wide, cfg, transform_cv5)
    loader_cv5 = DataLoader(dataset_cv5, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    
    for model_file in model_files:
        print(f"  Loading {model_file.name}...")
        
        model = CSIROModelCV5(cfg).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        
        preds, ids = predict_base(model, loader_cv5, device)
        all_preds.append(preds)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"✓ cv5 total: {len(model_files)} models")
    return np.mean(all_preds, axis=0), ids

#%% [markdown]
# ## 📋 Main

#%%
# 데이터 로드
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
print(f"Test samples: {len(test_wide)}")

#%%
print("\n" + "="*60)
print("🚀 CV5E: DINOv3 + ConvNeXt Multi-Backbone Ensemble")
print("="*60)
print(f"v27 weight: {cfg.V27_WEIGHT}")
print(f"cv5 weight: {cfg.CV5_WEIGHT}")

#%%
# v27 예측 (DINOv3)
print("\n" + "="*60)
print("📊 Predicting v27 (DINOv3)")
print("="*60)
v27_pred = predict_v27_all(test_wide, cfg, cfg.device)

#%%
# cv5 예측 (ConvNeXt)
print("\n" + "="*60)
print("📊 Predicting cv5 (ConvNeXt)")
print("="*60)
result = predict_cv5_all(test_wide, cfg, cfg.device)
if result is not None:
    cv5_pred, sample_ids = result
else:
    cv5_pred = None
    sample_ids = test_wide['sample_id_prefix'].tolist()

#%%
# 앙상블
print("\n" + "="*60)
print("⭐ Ensemble")
print("="*60)

if v27_pred is not None and cv5_pred is not None:
    final_pred = cfg.V27_WEIGHT * v27_pred + cfg.CV5_WEIGHT * cv5_pred
    print(f"✓ Weighted average: v27({cfg.V27_WEIGHT}) + cv5({cfg.CV5_WEIGHT})")
elif v27_pred is not None:
    final_pred = v27_pred
    print("⚠️ cv5 not available, using v27 only")
elif cv5_pred is not None:
    final_pred = cv5_pred
    print("⚠️ v27 not available, using cv5 only")
else:
    raise ValueError("No models available!")

print(f"Final shape: {final_pred.shape}")

#%%
# 예측 통계
print("\n=== Prediction Statistics ===")
print(f"{'Target':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
for idx, target in enumerate(TARGET_ORDER):
    vals = final_pred[:, idx]
    print(f"{target:<15} {vals.mean():>10.2f} {vals.std():>10.2f} {vals.min():>10.2f} {vals.max():>10.2f}")

#%%
# Submission 생성
pred_df = pd.DataFrame(final_pred, columns=TARGET_ORDER)
pred_df['sample_id_prefix'] = sample_ids

sub_df = pred_df.melt(
    id_vars=['sample_id_prefix'],
    value_vars=TARGET_ORDER,
    var_name='target_name',
    value_name='target'
)
sub_df['sample_id'] = sub_df['sample_id_prefix'] + '__' + sub_df['target_name']

submission = sub_df[['sample_id', 'target']]
submission.to_csv('submission.csv', index=False)

# 검증
sample_sub = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")
assert len(submission) == len(sample_sub), "Format mismatch!"

print(f"\n✅ submission.csv saved")
print(f"   {len(submission)} rows")
print(f"   Ensemble: v27({cfg.V27_WEIGHT}) + cv5({cfg.CV5_WEIGHT})")
print(f"   Expected LB: 0.71~0.73")
