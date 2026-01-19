#%% [markdown]
# # v27 Inference
# 
# 12-model ensemble (3 seeds × 4 states) with 8x TTA

#%%
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
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
    MODELS_DIR = Path("/kaggle/input/csiro-v27-models")
    
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (512, 512)
    
    hidden_dim = 256
    num_layers = 2
    dropout = 0.4
    use_layernorm = True
    freeze_backbone = True
    
    batch_size = 32
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

#%%
class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: CFG, transform: T.Compose = None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
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

#%%
def get_tta_transforms(cfg: CFG) -> List[T.Compose]:
    """8x TTA: original + flips + rotations + brightness"""
    base_normalize = [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    return [
        T.Compose([T.Resize(cfg.img_size)] + base_normalize),
        T.Compose([T.Resize(cfg.img_size), T.RandomHorizontalFlip(p=1.0)] + base_normalize),
        T.Compose([T.Resize(cfg.img_size), T.RandomVerticalFlip(p=1.0)] + base_normalize),
        T.Compose([T.Resize(cfg.img_size), T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0)] + base_normalize),
        T.Compose([T.Resize(cfg.img_size), T.RandomRotation([90, 90])] + base_normalize),
        T.Compose([T.Resize(cfg.img_size), T.RandomRotation([180, 180])] + base_normalize),
        T.Compose([T.Resize(cfg.img_size), T.RandomRotation([270, 270])] + base_normalize),
        T.Compose([T.Resize(cfg.img_size), T.ColorJitter(brightness=0.1)] + base_normalize),
    ]

def get_tta_dataloaders(df: pd.DataFrame, cfg: CFG) -> List[DataLoader]:
    transforms_list = get_tta_transforms(cfg)
    loaders = []
    
    for transform in transforms_list:
        dataset = TestDataset(df, cfg, transform)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        loaders.append(loader)
    
    return loaders

#%%
class CSIROModelV27(nn.Module):
    def __init__(self, cfg: CFG, backbone_weights_path: Path = None):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
        
        if backbone_weights_path and backbone_weights_path.exists():
            state = torch.load(backbone_weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        
        self.head_green = self._make_head(combined_dim)
        self.head_clover = self._make_head(combined_dim)
        self.head_dead = self._make_head(combined_dim)
        
        self.softplus = nn.Softplus(beta=1.0)
    
    def _make_head(self, in_dim: int) -> nn.Sequential:
        layers = []
        current_dim = in_dim
        
        for i in range(self.cfg.num_layers):
            layers.append(nn.Linear(current_dim, self.cfg.hidden_dim))
            if i < self.cfg.num_layers - 1:
                if self.cfg.use_layernorm:
                    layers.append(nn.LayerNorm(self.cfg.hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(self.cfg.dropout))
            current_dim = self.cfg.hidden_dim
        
        layers.append(nn.Linear(self.cfg.hidden_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)
        
        combined = torch.cat([left_feat, right_feat], dim=1)
        
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))
        
        gdm = green + clover
        total = gdm + dead
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%%
@torch.no_grad()
def predict_single_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    all_outputs, all_ids = [], []
    
    for left, right, ids in loader:
        left = left.to(device)
        right = right.to(device)
        outputs = model(left, right)
        all_outputs.append(outputs.cpu().numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids

def predict_with_tta(
    model: nn.Module,
    tta_loaders: List[DataLoader],
    device: str
) -> Tuple[np.ndarray, List[str]]:
    all_tta_preds = []
    final_ids = None
    
    for i, loader in enumerate(tta_loaders):
        preds, ids = predict_single_loader(model, loader, device)
        all_tta_preds.append(preds)
        if final_ids is None:
            final_ids = ids
    
    return np.mean(all_tta_preds, axis=0), final_ids

def predict_ensemble(
    cfg: CFG,
    tta_loaders: List[DataLoader]
) -> Tuple[np.ndarray, List[str]]:
    model_files = sorted(cfg.MODELS_DIR.glob("model_seed*.pth"))
    print(f"Found {len(model_files)} models")
    
    all_model_preds = []
    final_ids = None
    
    for model_file in tqdm(model_files, desc="Ensemble"):
        model = CSIROModelV27(cfg, cfg.BACKBONE_WEIGHTS).to(cfg.device)
        state_dict = torch.load(model_file, map_location=cfg.device, weights_only=True)
        model.load_state_dict(state_dict)
        
        preds, ids = predict_with_tta(model, tta_loaders, cfg.device)
        all_model_preds.append(preds)
        
        if final_ids is None:
            final_ids = ids
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    return np.mean(all_model_preds, axis=0), final_ids

#%%
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
print(f"Test samples: {len(test_wide)}")

#%%
tta_loaders = get_tta_dataloaders(test_wide, cfg)
print(f"TTA transforms: {len(tta_loaders)}x")

preds, sample_ids = predict_ensemble(cfg, tta_loaders)
print(f"Predictions shape: {preds.shape}")

#%%
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

pred_df = pd.DataFrame(preds, columns=TARGET_ORDER)
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
assert len(submission) == len(sample_sub), f"Length mismatch: {len(submission)} vs {len(sample_sub)}"
assert set(submission['sample_id']) == set(sample_sub['sample_id']), "Sample ID mismatch"
print("✓ Format verified!")

#%%
print("\n" + "="*50)
print("Prediction Statistics:")
print("="*50)
for col in TARGET_ORDER:
    vals = pred_df[col]
    print(f"{col:15s}: mean={vals.mean():7.2f}, std={vals.std():7.2f}, min={vals.min():7.2f}, max={vals.max():7.2f}")
