#%% [markdown]
# # 🚀 v27 Breakthrough Inference
#
# **기존 모델(v20) 그대로 사용 + Inference-time 개선만으로 LB 돌파**
#
# ## Breakthrough Strategies:
# 1. **RGB Vegetation Index Calibration**: 이미지에서 직접 계산 가능한 식생지수로 보정
# 2. **Prediction Uncertainty Weighting**: 불확실한 예측은 평균으로 shrink
# 3. **Physics-Aware Post-Processing**: GDM = Green + Clover 관계 강제 보정
# 4. **Multi-Scale TTA**: 다양한 해상도에서 예측 후 앙상블
# 5. **Monte Carlo Dropout**: Inference에서 dropout으로 uncertainty 추정

#%%
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import timm

tqdm.pandas()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

#%%
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything(42)

#%%
class CFG:
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    BACKBONE_WEIGHTS = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large/dinov3_vitl16_qkvb.pth")
    MODELS_DIR = Path("/kaggle/input/csiro-v20-models")
    
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (512, 512)
    
    hidden_dim = 512
    num_layers = 3
    dropout = 0.1
    use_layernorm = True
    
    batch_size = 16
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    mc_dropout_iterations = 10
    multi_scale_sizes = [(448, 448), (512, 512), (576, 576)]

cfg = CFG()

TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
TARGET_WEIGHTS = {'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1, 'GDM_g': 0.2, 'Dry_Total_g': 0.5}

#%% [markdown]
# ## 🌿 Strategy 1: RGB Vegetation Index
#
# Train과 Test 모두에서 **이미지만으로** 계산 가능한 식생지수
# - ExG (Excess Green) = 2G - R - B
# - VARI = (G - R) / (G + R - B)
# - Green Ratio = G / (R + G + B)

#%%
def compute_vegetation_indices(img: Image.Image) -> Dict[str, float]:
    arr = np.array(img).astype(np.float32) / 255.0
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    exg = (2 * G - R - B).mean()
    green_ratio = G.mean() / (R.mean() + G.mean() + B.mean() + 1e-8)
    vari = ((G - R) / (G + R - B + 1e-8)).mean()
    vari = np.clip(vari, -1, 1)
    brightness = arr.mean()
    greenness = G.mean()
    
    return {
        'exg': exg,
        'green_ratio': green_ratio,
        'vari': vari,
        'brightness': brightness,
        'greenness': greenness
    }

#%%
class TestDatasetWithVegIndex(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: CFG, transform: T.Compose = None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.cfg.DATA_PATH / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        left_img = img.crop((0, 0, mid, height))
        right_img = img.crop((mid, 0, width, height))
        
        veg_indices = compute_vegetation_indices(img)
        
        if self.transform:
            left_t = self.transform(left_img)
            right_t = self.transform(right_img)
        else:
            left_t = left_img
            right_t = right_img
        
        return left_t, right_t, row['sample_id_prefix'], veg_indices

#%%
def get_multi_scale_transforms(cfg: CFG) -> List[Tuple[Tuple[int, int], T.Compose]]:
    transforms_list = []
    
    for size in cfg.multi_scale_sizes:
        base = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transforms_list.append((size, base))
        
        hflip = T.Compose([
            T.Resize(size),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transforms_list.append((size, hflip))
        
        vflip = T.Compose([
            T.Resize(size),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transforms_list.append((size, vflip))
    
    return transforms_list

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

class CSIROModelV20(nn.Module):
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

#%% [markdown]
# ## 🎲 Strategy 2: Monte Carlo Dropout
#
# Inference 시 dropout을 켜서 여러번 예측 → uncertainty 추정

#%%
def enable_dropout(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def mc_dropout_predict(
    model: nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    n_iter: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    enable_dropout(model)
    
    predictions = []
    for _ in range(n_iter):
        with torch.no_grad():
            pred = model(left, right)
        predictions.append(pred.cpu().numpy())
    
    predictions = np.stack(predictions, axis=0)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    
    return mean_pred, std_pred

#%% [markdown]
# ## 🔧 Strategy 3: Physics-Aware Post-Processing
#
# 물리적 제약을 재적용하여 일관성 확보:
# - GDM = Green + Clover
# - Total = GDM + Dead = Green + Clover + Dead
#
# 모델이 어긋난 예측을 했다면 보정

#%%
def physics_aware_correction(preds: np.ndarray) -> np.ndarray:
    corrected = preds.copy()
    
    green = corrected[:, 0]
    dead = corrected[:, 1]
    clover = corrected[:, 2]
    gdm_pred = corrected[:, 3]
    total_pred = corrected[:, 4]
    
    gdm_true = green + clover
    total_true = gdm_true + dead
    
    gdm_error = np.abs(gdm_pred - gdm_true)
    total_error = np.abs(total_pred - total_true)
    
    corrected[:, 3] = 0.7 * gdm_pred + 0.3 * gdm_true
    corrected[:, 4] = 0.7 * total_pred + 0.3 * total_true
    
    corrected = np.maximum(corrected, 0)
    
    return corrected

#%% [markdown]
# ## 📊 Strategy 4: Vegetation Index Based Scaling
#
# Train 통계 기반으로 ExG와 biomass의 관계를 활용

#%%
TRAIN_STATS = {
    'exg_mean': 0.15,
    'exg_std': 0.08,
    'total_mean': 45.0,
    'total_std': 25.0,
    'green_mean': 25.0,
    'green_std': 18.0,
}

def vegetation_index_scaling(
    preds: np.ndarray,
    veg_indices: List[Dict[str, float]],
    alpha: float = 0.15
) -> np.ndarray:
    scaled = preds.copy()
    
    for i, veg in enumerate(veg_indices):
        exg = veg['exg']
        greenness = veg['greenness']
        
        exg_z = (exg - TRAIN_STATS['exg_mean']) / (TRAIN_STATS['exg_std'] + 1e-8)
        
        if exg_z > 1.5:
            scale_factor = 1 + alpha * min(exg_z - 1, 1)
            scaled[i, 0] *= scale_factor
            scaled[i, 3] *= scale_factor
            scaled[i, 4] *= scale_factor
        elif exg_z < -1.5:
            scale_factor = 1 - alpha * min(abs(exg_z) - 1, 0.5)
            scaled[i, 0] *= scale_factor
            scaled[i, 3] *= scale_factor
            scaled[i, 4] *= scale_factor
    
    return scaled

#%% [markdown]
# ## 🎯 Strategy 5: Uncertainty-Aware Shrinkage
#
# MC Dropout으로 얻은 uncertainty가 높으면 → 평균으로 shrink

#%%
def uncertainty_shrinkage(
    preds: np.ndarray,
    stds: np.ndarray,
    global_means: np.ndarray,
    shrink_threshold: float = 0.3
) -> np.ndarray:
    shrunk = preds.copy()
    
    for j in range(5):
        target_mean = global_means[j]
        target_std = stds[:, j]
        
        normalized_std = target_std / (np.abs(preds[:, j]) + 1e-8)
        
        high_uncertainty_mask = normalized_std > shrink_threshold
        
        if high_uncertainty_mask.any():
            shrink_factor = np.clip(1 - (normalized_std - shrink_threshold), 0.5, 1.0)
            shrunk[high_uncertainty_mask, j] = (
                shrink_factor[high_uncertainty_mask] * preds[high_uncertainty_mask, j] +
                (1 - shrink_factor[high_uncertainty_mask]) * target_mean
            )
    
    return shrunk

#%% [markdown]
# ## 🚀 Main Inference Pipeline

#%%
def load_model(cfg: CFG, model_path: Path) -> nn.Module:
    model = CSIROModelV20(cfg, cfg.BACKBONE_WEIGHTS).to(cfg.device)
    state_dict = torch.load(model_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(state_dict)
    return model

#%%
@torch.no_grad()
def predict_single_transform(
    model: nn.Module,
    df: pd.DataFrame,
    cfg: CFG,
    transform: T.Compose,
    use_mc_dropout: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    dataset = TestDatasetWithVegIndex(df, cfg, transform)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)
    
    all_preds, all_stds, all_ids, all_veg = [], [], [], []
    
    for left, right, ids, veg_indices in tqdm(loader, desc="Predicting"):
        left = left.to(cfg.device)
        right = right.to(cfg.device)
        
        if use_mc_dropout:
            mean_pred, std_pred = mc_dropout_predict(model, left, right, cfg.mc_dropout_iterations)
            all_preds.append(mean_pred)
            all_stds.append(std_pred)
        else:
            model.eval()
            pred = model(left, right).cpu().numpy()
            all_preds.append(pred)
            all_stds.append(np.zeros_like(pred))
        
        all_ids.extend(ids)
        for key in veg_indices:
            if isinstance(veg_indices[key], torch.Tensor):
                veg_indices[key] = veg_indices[key].numpy()
        
        batch_veg = [{k: veg_indices[k][i].item() if hasattr(veg_indices[k][i], 'item') 
                     else veg_indices[k][i] for k in veg_indices} 
                    for i in range(len(ids))]
        all_veg.extend(batch_veg)
    
    return np.concatenate(all_preds), np.concatenate(all_stds), all_ids, all_veg

#%%
def full_inference_pipeline(cfg: CFG, test_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    model_files = sorted(cfg.MODELS_DIR.glob("model_fold*.pth"))
    print(f"Found {len(model_files)} models")
    
    transforms_list = get_multi_scale_transforms(cfg)
    print(f"Using {len(transforms_list)} transform variants (multi-scale + TTA)")
    
    all_ensemble_preds = []
    all_ensemble_stds = []
    final_ids = None
    final_veg = None
    
    for model_path in tqdm(model_files, desc="Models"):
        model = load_model(cfg, model_path)
        
        model_preds = []
        model_stds = []
        
        for size, transform in transforms_list:
            preds, stds, ids, veg = predict_single_transform(
                model, test_df, cfg, transform, use_mc_dropout=True
            )
            model_preds.append(preds)
            model_stds.append(stds)
            
            if final_ids is None:
                final_ids = ids
                final_veg = veg
        
        model_mean = np.mean(model_preds, axis=0)
        model_std = np.mean(model_stds, axis=0)
        
        all_ensemble_preds.append(model_mean)
        all_ensemble_stds.append(model_std)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    raw_preds = np.mean(all_ensemble_preds, axis=0)
    uncertainty = np.mean(all_ensemble_stds, axis=0)
    
    print("\n=== Applying Post-Processing ===")
    
    print("1. Vegetation Index Scaling...")
    scaled_preds = vegetation_index_scaling(raw_preds, final_veg, alpha=0.1)
    
    print("2. Uncertainty Shrinkage...")
    global_means = raw_preds.mean(axis=0)
    shrunk_preds = uncertainty_shrinkage(scaled_preds, uncertainty, global_means, shrink_threshold=0.25)
    
    print("3. Physics-Aware Correction...")
    final_preds = physics_aware_correction(shrunk_preds)
    
    final_preds = np.maximum(final_preds, 0)
    
    return final_preds, final_ids

#%%
print("\n" + "="*60)
print("🚀 v27 Breakthrough Inference")
print("="*60)

test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
print(f"Test samples: {len(test_wide)}")

#%%
final_preds, sample_ids = full_inference_pipeline(cfg, test_wide)
print(f"\nFinal predictions shape: {final_preds.shape}")

#%%
print("\n=== Prediction Statistics ===")
for idx, target in enumerate(TARGET_ORDER):
    vals = final_preds[:, idx]
    print(f"{target:15s}: mean={vals.mean():7.2f}, std={vals.std():7.2f}, "
          f"min={vals.min():7.2f}, max={vals.max():7.2f}")

#%%
pred_df = pd.DataFrame(final_preds, columns=TARGET_ORDER)
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

print(f"\n✅ Saved submission.csv: {len(submission)} rows")

#%%
sample_sub = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")
assert len(submission) == len(sample_sub), f"Length mismatch"
assert set(submission['sample_id']) == set(sample_sub['sample_id']), "ID mismatch"
print("✓ Format verified!")

print("\n" + "="*60)
print("🎉 Breakthrough Inference Complete!")
print("="*60)
print("Applied techniques:")
print("  1. Multi-scale TTA (3 sizes × 3 augments = 9x)")
print("  2. Monte Carlo Dropout (10 iterations)")
print("  3. RGB Vegetation Index Scaling")
print("  4. Uncertainty-Aware Shrinkage")
print("  5. Physics-Aware Correction")
