#%% [markdown]
# # 🔧 DINOv3 Inference v15
#
# **모델**: v15 (15_back_to_basics.py로 학습된 모델)
# **환경**: Kaggle (학습된 모델을 Dataset으로 업로드 후 사용)

#%%
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torchvision import transforms as T

import timm

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

#%% [markdown]
# ## ⚙️ Configuration

#%%
class CFG:
    # === 경로 (Kaggle) ===
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    
    # ⚠️ 이 경로를 업로드한 모델 Dataset 경로로 변경하세요
    MODELS_DIR = Path("/kaggle/input/csiro-v15-models")  # 예시
    
    # === Model (v15와 동일해야 함) ===
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (512, 512)
    dropout = 0.1
    
    # === TTA ===
    use_tta = True  # Test Time Augmentation
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()
print(f"Device: {cfg.device}")
print(f"Models: {cfg.MODELS_DIR}")

#%% [markdown]
# ## 🧠 Model Definition (v15와 동일)

#%%
class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )
    
    def forward(self, context):
        gamma_beta = self.mlp(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma, beta

class CSIROModel(nn.Module):
    """v15 Model (v12 기반)"""
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        
        # Backbone (pretrained=False로 로드 - 가중치는 나중에 load_state_dict)
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='avg')
        
        feat_dim = self.backbone.num_features
        
        # FiLM
        self.film = FiLM(feat_dim)
        
        # Heads (v12와 동일: 256 hidden units)
        def make_head():
            return nn.Sequential(
                nn.Linear(feat_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 1)
            )
        
        self.head_green = make_head()
        self.head_clover = make_head()
        self.head_dead = make_head()
        
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
        
        # [Green, Dead, Clover, GDM, Total]
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## 🎨 Transforms

#%%
def get_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_tta_transforms(cfg):
    """TTA: original + hflip + vflip"""
    base = get_transforms(cfg)
    
    hflip = T.Compose([
        T.Resize(cfg.img_size),
        T.functional.hflip,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    vflip = T.Compose([
        T.Resize(cfg.img_size),
        T.functional.vflip,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return [base, hflip, vflip]

#%% [markdown]
# ## 🔮 Inference Functions

#%%
@torch.no_grad()
def predict_single(model, img_path, transforms, cfg):
    """Single image prediction with optional TTA"""
    img = Image.open(cfg.DATA_PATH / img_path).convert('RGB')
    width, height = img.size
    mid = width // 2
    
    left_img = img.crop((0, 0, mid, height))
    right_img = img.crop((mid, 0, width, height))
    
    if cfg.use_tta:
        # TTA: average over transforms
        all_preds = []
        for transform in transforms:
            left_t = transform(left_img).unsqueeze(0).to(cfg.device)
            right_t = transform(right_img).unsqueeze(0).to(cfg.device)
            outputs = model(left_t, right_t)
            all_preds.append(outputs.cpu().numpy())
        return np.mean(all_preds, axis=0)
    else:
        transform = transforms[0]
        left_t = transform(left_img).unsqueeze(0).to(cfg.device)
        right_t = transform(right_img).unsqueeze(0).to(cfg.device)
        outputs = model(left_t, right_t)
        return outputs.cpu().numpy()

def predict_all_folds(test_df, cfg):
    """Ensemble prediction across all folds"""
    transforms = get_tta_transforms(cfg) if cfg.use_tta else [get_transforms(cfg)]
    
    all_fold_preds = []
    model_files = sorted(cfg.MODELS_DIR.glob("model_fold*.pth"))
    
    print(f"Found {len(model_files)} model files")
    
    for model_path in model_files:
        print(f"Loading {model_path.name}...")
        
        model = CSIROModel(cfg.model_name, cfg.dropout)
        state_dict = torch.load(model_path, map_location=cfg.device)
        model.load_state_dict(state_dict)
        model = model.to(cfg.device)
        model.eval()
        
        fold_preds = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Fold"):
            pred = predict_single(model, row['image_path'], transforms, cfg)
            fold_preds.append(pred)
        
        all_fold_preds.append(np.concatenate(fold_preds))
        
        del model
        torch.cuda.empty_cache()
    
    # Average across folds
    return np.mean(all_fold_preds, axis=0)

#%% [markdown]
# ## 📋 Main Inference

#%%
if __name__ == "__main__":
    # Load test data
    test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
    test_df['target'] = 0.0
    test_df[['sample_id_prefix', 'sample_id_suffix']] = test_df.sample_id.str.split('__', expand=True)
    
    # Get unique images
    test_data = test_df.groupby(['sample_id_prefix', 'image_path']).apply(
        lambda df: df.set_index('target_name').target
    ).reset_index()
    test_data.columns.name = None
    
    print(f"Test images: {len(test_data)}")
    
    # Predict
    preds = predict_all_folds(test_data, cfg)
    
    # Assign predictions
    # Output order: [Green, Dead, Clover, GDM, Total]
    test_data['Dry_Green_g'] = preds[:, 0]
    test_data['Dry_Dead_g'] = preds[:, 1]
    test_data['Dry_Clover_g'] = preds[:, 2]
    test_data['GDM_g'] = preds[:, 3]
    test_data['Dry_Total_g'] = preds[:, 4]
    
    # Create submission
    cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    sub_df = test_data.set_index('sample_id_prefix')[cols].stack().reset_index()
    sub_df.columns = ['sample_id_prefix', 'target_name', 'target']
    sub_df['sample_id'] = sub_df.sample_id_prefix + '__' + sub_df.target_name
    
    # Save
    sub_df[['sample_id', 'target']].to_csv('submission.csv', index=False)
    
    print("\n✅ submission.csv created!")
    print(sub_df[['sample_id', 'target']].head(10))
