#%% [markdown]
# # CV3 Inference with TTA
#
# **특징**:
# 1. LoRA + Progressive Head 모델
# 2. 해상도 560 (학습과 동일)
# 3. TTA: 4-fold flip
# 4. 5-fold 앙상블

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

# PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "peft", "-q"])
    from peft import LoraConfig, get_peft_model

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
    BACKBONE_WEIGHTS = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large/dinov3_vitl16_qkvb.pth")
    
    # ⚠️ CV3 모델 경로
    MODELS_DIR = Path("/kaggle/input/csiro-cv3-models")
    
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (560, 560)
    
    # === LoRA 설정 (학습과 동일!) ===
    use_lora = True
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    
    # === Head 설정 (학습과 동일!) ===
    head_dims = [512, 256]
    head_dropout = 0.1
    
    # Inference
    batch_size = 16
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TTA
    use_tta = True
    n_tta = 4

cfg = CFG()

print(f"Device: {cfg.device}")
print(f"LoRA: r={cfg.lora_r}")
print(f"Head: 2048 → {' → '.join(map(str, cfg.head_dims))} → 1")
print(f"TTA: {cfg.use_tta} ({cfg.n_tta} augmentations)")

TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

#%% [markdown]
# ## 📊 Dataset

#%%
class TestDataset(Dataset):
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


def get_test_transform(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
        return torch.chunk(self.mlp(context), 2, dim=1)


class ProgressiveHead(nn.Module):
    """Progressive Head: 점진적 차원 축소"""
    def __init__(self, in_dim, hidden_dims, dropout=0.1):
        super().__init__()
        
        layers = []
        current_dim = in_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class CSIROModelCV3(nn.Module):
    """CV3: LoRA + Progressive Head"""
    def __init__(self, cfg, backbone_weights_path=None):
        super().__init__()
        
        # Backbone
        if backbone_weights_path and Path(backbone_weights_path).exists():
            self.backbone = timm.create_model(cfg.model_name, pretrained=False, 
                                               num_classes=0, global_pool='avg')
            state = torch.load(backbone_weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        else:
            self.backbone = timm.create_model(cfg.model_name, pretrained=True, 
                                               num_classes=0, global_pool='avg')
        
        # LoRA
        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["qkv"],
                bias="none"
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        
        feat_dim = 1024
        combined_dim = feat_dim * 2
        
        self.film = FiLM(feat_dim)
        
        self.head_green = ProgressiveHead(combined_dim, cfg.head_dims, cfg.head_dropout)
        self.head_clover = ProgressiveHead(combined_dim, cfg.head_dims, cfg.head_dropout)
        self.head_dead = ProgressiveHead(combined_dim, cfg.head_dims, cfg.head_dropout)
        
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
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## 🔮 Inference with TTA

#%%
@torch.no_grad()
def predict_single(model, left, right, device):
    return model(left.to(device), right.to(device)).cpu()


@torch.no_grad()
def predict_with_tta(model, left, right, device, n_tta=4):
    """TTA: HFlip x VFlip = 4가지"""
    preds = []
    
    for hflip in [False, True]:
        for vflip in [False, True]:
            l = torch.flip(left, [3]) if hflip else left
            r = torch.flip(right, [3]) if hflip else right
            l = torch.flip(l, [2]) if vflip else l
            r = torch.flip(r, [2]) if vflip else r
            
            pred = model(l.to(device), r.to(device))
            preds.append(pred.cpu())
            
            if len(preds) >= n_tta:
                break
        if len(preds) >= n_tta:
            break
    
    return torch.stack(preds).mean(0)


def predict_batch(model, loader, cfg):
    model.eval()
    device = cfg.device
    all_outputs, all_ids = [], []
    
    for left, right, ids in tqdm(loader, desc="Predicting"):
        if cfg.use_tta:
            outputs = predict_with_tta(model, left, right, device, cfg.n_tta)
        else:
            outputs = predict_single(model, left, right, device)
        
        all_outputs.append(outputs.numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids


def predict_ensemble(cfg, loader):
    model_files = sorted(cfg.MODELS_DIR.glob("model_fold*.pth"))
    print(f"\nFound {len(model_files)} models")
    
    all_fold_preds = []
    final_ids = None
    
    for model_file in model_files:
        print(f"\nLoading {model_file.name}...")
        
        model = CSIROModelCV3(cfg, cfg.BACKBONE_WEIGHTS).to(cfg.device)
        model.load_state_dict(torch.load(model_file, map_location=cfg.device))
        print("✓ Loaded")
        
        preds, ids = predict_batch(model, loader, cfg)
        all_fold_preds.append(preds)
        
        if final_ids is None:
            final_ids = ids
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    ensemble_pred = np.mean(all_fold_preds, axis=0)
    
    return ensemble_pred, final_ids

#%% [markdown]
# ## 📋 Main

#%%
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
print(f"Test samples: {len(test_wide)}")

#%%
print("\n" + "="*60)
print("🚀 CV3 Inference with TTA (LoRA + Progressive Head)")
print("="*60)
print(f"LoRA: r={cfg.lora_r}")
print(f"Head: Progressive")
print(f"TTA: {cfg.n_tta}-fold")

transform = get_test_transform(cfg)
dataset = TestDataset(test_wide, cfg, transform)
loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                   num_workers=cfg.num_workers, pin_memory=True)

predictions, sample_ids = predict_ensemble(cfg, loader)
print(f"\nPredictions: {predictions.shape}")

#%%
print("\n=== Prediction Statistics ===")
print(f"{'Target':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
for idx, target in enumerate(TARGET_ORDER):
    vals = predictions[:, idx]
    print(f"{target:<15} {vals.mean():>10.2f} {vals.std():>10.2f} {vals.min():>10.2f} {vals.max():>10.2f}")

#%%
pred_df = pd.DataFrame(predictions, columns=TARGET_ORDER)
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

sample_sub = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")
assert len(submission) == len(sample_sub), "Format mismatch!"

print(f"\n✅ submission.csv saved")
print(f"   {len(submission)} rows")
