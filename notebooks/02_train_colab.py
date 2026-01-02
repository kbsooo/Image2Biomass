#%% [markdown]
# # CSIRO Image2Biomass - Training Pipeline (Colab)
#
# DINOv2 + Tabular Fusion Model for Multi-output Regression
#
# **Strategy:**
# 1. DINOv2-base backbone (fine-tuned)
# 2. Multi-scale feature extraction (layers 6, 9, 12)
# 3. Cross-attention fusion with tabular features
# 4. Multi-task regression head
# 5. Log transform for targets
# 6. State-based CV (spatial split)

#%% [markdown]
# ## 0. Setup (Colab)

#%%
# !pip install -q timm albumentations transformers accelerate wandb

#%%
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

# Colab에서 Drive 마운트 (필요시 주석 해제)
# from google.colab import drive
# drive.mount('/content/drive')

#%%
# ============================================================
# Config
# ============================================================
class CFG:
    # Paths - Colab 환경에 맞게 수정
    data_dir = Path("./data")  # 또는 "/content/drive/MyDrive/kaggle/csiro-biomass/data"
    output_dir = Path("./outputs")

    # Model
    backbone = "vit_base_patch14_dinov2.lvd142m"  # DINOv2-base
    img_size = 518  # DINOv2 default
    in_chans = 3

    # Tabular
    num_tabular_features = 2  # NDVI, Height
    num_cat_features = 2  # State, Species

    # Targets
    targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'Dry_Total_g', 'GDM_g']
    num_targets = 5

    # Training
    n_folds = 4
    train_folds = [0]  # 빠른 실험용. 전체는 [0,1,2,3]
    epochs = 15
    batch_size = 8
    lr = 1e-4
    backbone_lr = 1e-5  # backbone은 더 낮은 lr
    weight_decay = 1e-2

    # Augmentation
    aug_prob = 0.5

    # Log transform (right-skewed targets)
    use_log_transform = True
    log_eps = 1.0  # log(x + eps) for numerical stability

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    seed = 42

    # Misc
    num_workers = 2
    gradient_accumulation = 2
    mixed_precision = True

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.seed)
CFG.output_dir.mkdir(exist_ok=True, parents=True)

print(f"Device: {CFG.device}")
print(f"PyTorch: {torch.__version__}")

#%% [markdown]
# ## 1. Data Preparation

#%%
def prepare_data(cfg: CFG):
    """Load and prepare data in wide format."""
    train_df = pd.read_csv(cfg.data_dir / "train.csv")

    # Long → Wide format
    tabular_cols = ['image_path', 'Sampling_Date', 'State', 'Species',
                    'Pre_GSHH_NDVI', 'Height_Ave_cm']

    train_wide = train_df.pivot_table(
        index=tabular_cols,
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    # Extract image_id for grouping
    train_wide['image_id'] = train_wide['image_path'].apply(
        lambda x: Path(x).stem
    )

    # Parse date features
    train_wide['Sampling_Date'] = pd.to_datetime(train_wide['Sampling_Date'])
    train_wide['Month'] = train_wide['Sampling_Date'].dt.month

    # Encode categoricals
    le_state = LabelEncoder()
    le_species = LabelEncoder()
    train_wide['State_enc'] = le_state.fit_transform(train_wide['State'])
    train_wide['Species_enc'] = le_species.fit_transform(train_wide['Species'])

    # Create folds (GroupKFold by State for spatial CV)
    # Insight: 같은 State 데이터가 train/val에 섞이면 data leakage 발생
    gkf = GroupKFold(n_splits=cfg.n_folds)
    train_wide['fold'] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(train_wide, groups=train_wide['State'])):
        train_wide.loc[val_idx, 'fold'] = fold

    print(f"Data shape: {train_wide.shape}")
    print(f"Fold distribution:\n{train_wide['fold'].value_counts().sort_index()}")
    print(f"States: {le_state.classes_}")
    print(f"Species: {le_species.classes_}")

    return train_wide, le_state, le_species

#%%
train_df, le_state, le_species = prepare_data(CFG)
train_df.head()

#%% [markdown]
# ## 2. Dataset & Transforms

#%%
def get_transforms(cfg: CFG, mode='train'):
    """Get augmentation transforms."""
    if mode == 'train':
        return A.Compose([
            A.Resize(cfg.img_size, cfg.img_size),
            A.HorizontalFlip(p=cfg.aug_prob),
            A.VerticalFlip(p=cfg.aug_prob),
            A.RandomRotate90(p=cfg.aug_prob),
            # Insight: 색상 변환은 약하게 (녹색/갈색 구분이 중요)
            A.ColorJitter(
                brightness=0.1, contrast=0.1,
                saturation=0.1, hue=0.02,
                p=cfg.aug_prob * 0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
                border_mode=0, p=cfg.aug_prob
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(cfg.img_size, cfg.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

#%%
class BiomassDataset(Dataset):
    def __init__(self, df, cfg: CFG, transforms=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = self.cfg.data_dir / row['image_path']
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transforms:
            image = self.transforms(image=image)['image']

        # Tabular features (numerical)
        tabular = torch.tensor([
            row['Pre_GSHH_NDVI'],
            row['Height_Ave_cm'] / 100.0,  # Normalize height
        ], dtype=torch.float32)

        # Categorical features
        cat_features = torch.tensor([
            row['State_enc'],
            row['Species_enc'],
        ], dtype=torch.long)

        # Targets
        targets = torch.tensor([
            row[t] for t in self.cfg.targets
        ], dtype=torch.float32)

        # Log transform targets
        if self.cfg.use_log_transform:
            targets = torch.log1p(targets)  # log(1 + x)

        return {
            'image': image,
            'tabular': tabular,
            'cat_features': cat_features,
            'targets': targets,
            'image_id': row['image_id']
        }

#%% [markdown]
# ## 3. Model Architecture

#%%
class MultiScaleDINOv2(nn.Module):
    """
    DINOv2 with multi-scale feature extraction.
    Extracts features from multiple transformer layers for richer representations.
    """
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        self.embed_dim = self.backbone.embed_dim  # 768 for base

        # Insight: 다른 layer는 다른 수준의 feature를 캡처
        # Early layers: 텍스처, 엣지 / Late layers: 의미론적 정보
        self.feature_layers = [6, 9, 11]  # 0-indexed

    def forward(self, x):
        # Get intermediate features
        features = []

        # Patch embedding
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)

        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i in self.feature_layers:
                # CLS token
                features.append(x[:, 0])

        # Final layer
        x = self.backbone.norm(x)
        features.append(x[:, 0])

        # Concat multi-scale features
        # Shape: (B, embed_dim * num_layers)
        multi_scale_feat = torch.cat(features, dim=-1)

        return multi_scale_feat, x[:, 1:]  # CLS features, patch tokens

#%%
class TabularEncoder(nn.Module):
    """Encode tabular features (numerical + categorical)."""
    def __init__(self, num_numerical, cat_dims, embed_dim=64):
        super().__init__()
        self.num_numerical = num_numerical

        # Numerical features MLP
        self.num_encoder = nn.Sequential(
            nn.Linear(num_numerical, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim)
            for num_classes in cat_dims
        ])

        # Combine
        total_dim = embed_dim * (1 + len(cat_dims))
        self.combine = nn.Sequential(
            nn.Linear(total_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, numerical, categorical):
        # Numerical
        num_feat = self.num_encoder(numerical)

        # Categorical
        cat_feats = [emb(categorical[:, i]) for i, emb in enumerate(self.cat_embeddings)]

        # Combine
        combined = torch.cat([num_feat] + cat_feats, dim=-1)
        return self.combine(combined)

#%%
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention to fuse tabular features with image patch tokens.
    Tabular queries attend to image patches to find relevant regions.
    """
    def __init__(self, img_dim, tab_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = img_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Project tabular to query
        self.q_proj = nn.Linear(tab_dim, img_dim)
        # Image patches as key/value
        self.k_proj = nn.Linear(img_dim, img_dim)
        self.v_proj = nn.Linear(img_dim, img_dim)

        self.out_proj = nn.Linear(img_dim, img_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tab_feat, img_tokens):
        """
        tab_feat: (B, tab_dim) - tabular features
        img_tokens: (B, N, img_dim) - image patch tokens
        """
        B, N, C = img_tokens.shape

        # Expand tabular to sequence (B, 1, img_dim)
        q = self.q_proj(tab_feat).unsqueeze(1)
        k = self.k_proj(img_tokens)
        v = self.v_proj(img_tokens)

        # Multi-head attention
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Insight: F.scaled_dot_product_attention uses FlashAttention when available
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)

        attn_out = attn_out.transpose(1, 2).reshape(B, 1, C)
        attn_out = self.out_proj(attn_out).squeeze(1)

        return attn_out

#%%
class BiomassModel(nn.Module):
    """
    Full model: DINOv2 + Tabular + Cross-Attention Fusion
    """
    def __init__(self, cfg: CFG, cat_dims):
        super().__init__()
        self.cfg = cfg

        # Image encoder (DINOv2)
        self.image_encoder = MultiScaleDINOv2(cfg.backbone, pretrained=True)
        img_dim = self.image_encoder.embed_dim
        multi_scale_dim = img_dim * 4  # 4 layers of features

        # Tabular encoder
        tab_embed_dim = 128
        self.tabular_encoder = TabularEncoder(
            num_numerical=cfg.num_tabular_features,
            cat_dims=cat_dims,
            embed_dim=tab_embed_dim
        )

        # Cross-attention fusion
        self.cross_attn = CrossAttentionFusion(
            img_dim=img_dim,
            tab_dim=tab_embed_dim,
            num_heads=8
        )

        # Fusion layer
        fusion_dim = multi_scale_dim + tab_embed_dim + img_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Multi-task regression head
        self.head = nn.Linear(256, cfg.num_targets)

    def forward(self, image, tabular, cat_features):
        # Image features
        multi_scale_feat, patch_tokens = self.image_encoder(image)

        # Tabular features
        tab_feat = self.tabular_encoder(tabular, cat_features)

        # Cross-attention: tabular queries image patches
        cross_attn_feat = self.cross_attn(tab_feat, patch_tokens)

        # Fuse all features
        fused = torch.cat([multi_scale_feat, tab_feat, cross_attn_feat], dim=-1)
        fused = self.fusion(fused)

        # Predict all targets
        out = self.head(fused)

        return out

#%% [markdown]
# ## 4. Loss Function

#%%
class BiomassLoss(nn.Module):
    """
    Combined loss for multi-output regression.
    - SmoothL1 (Huber) for robustness to outliers
    - Physics constraint: Total ≈ Green + Dead + Clover
    """
    def __init__(self, cfg: CFG, physics_weight=0.1):
        super().__init__()
        self.cfg = cfg
        self.physics_weight = physics_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

        # Target weights (Clover가 어려우므로 가중치 높임)
        # [Green, Dead, Clover, Total, GDM]
        self.target_weights = torch.tensor([1.0, 1.0, 1.5, 1.0, 1.0])

    def forward(self, pred, target):
        """
        pred: (B, 5) - predicted values (log-transformed)
        target: (B, 5) - target values (log-transformed)
        """
        # Per-target loss
        loss_per_target = self.smooth_l1(pred, target)  # (B, 5)

        # Weighted mean
        weights = self.target_weights.to(pred.device)
        weighted_loss = (loss_per_target * weights).mean()

        # Physics constraint (in original scale)
        if self.physics_weight > 0:
            # Convert back from log scale
            pred_orig = torch.expm1(pred)  # exp(x) - 1

            # Total should equal Green + Dead + Clover
            pred_sum = pred_orig[:, 0] + pred_orig[:, 1] + pred_orig[:, 2]
            pred_total = pred_orig[:, 3]
            physics_loss = F.smooth_l1_loss(pred_sum, pred_total)

            weighted_loss = weighted_loss + self.physics_weight * physics_loss

        return weighted_loss

#%% [markdown]
# ## 5. Training Loop

#%%
def train_one_epoch(model, loader, optimizer, scheduler, criterion, cfg, scaler=None):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training')
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        image = batch['image'].to(cfg.device)
        tabular = batch['tabular'].to(cfg.device)
        cat_features = batch['cat_features'].to(cfg.device)
        targets = batch['targets'].to(cfg.device)

        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
            pred = model(image, tabular, cat_features)
            loss = criterion(pred, targets)
            loss = loss / cfg.gradient_accumulation

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (step + 1) % cfg.gradient_accumulation == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * cfg.gradient_accumulation
        pbar.set_postfix({'loss': f'{loss.item() * cfg.gradient_accumulation:.4f}'})

    return total_loss / len(loader)

#%%
@torch.no_grad()
def validate(model, loader, criterion, cfg):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc='Validation'):
        image = batch['image'].to(cfg.device)
        tabular = batch['tabular'].to(cfg.device)
        cat_features = batch['cat_features'].to(cfg.device)
        targets = batch['targets'].to(cfg.device)

        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
            pred = model(image, tabular, cat_features)
            loss = criterion(pred, targets)

        total_loss += loss.item()

        # Convert back from log scale
        if cfg.use_log_transform:
            pred = torch.expm1(pred)
            targets_orig = torch.expm1(targets)
        else:
            targets_orig = targets

        all_preds.append(pred.cpu())
        all_targets.append(targets_orig.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Compute R² for each target
    r2_scores = {}
    for i, target_name in enumerate(cfg.targets):
        ss_res = np.sum((all_targets[:, i] - all_preds[:, i]) ** 2)
        ss_tot = np.sum((all_targets[:, i] - all_targets[:, i].mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        r2_scores[target_name] = r2

    # Global weighted R² (equal weights for now)
    mean_r2 = np.mean(list(r2_scores.values()))

    return total_loss / len(loader), r2_scores, mean_r2

#%% [markdown]
# ## 6. Main Training

#%%
def train_fold(fold, train_df, cfg):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold}")
    print(f"{'='*50}")

    # Split data
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Val States: {val_data['State'].unique()}")

    # Datasets
    train_dataset = BiomassDataset(
        train_data, cfg,
        transforms=get_transforms(cfg, 'train'),
        mode='train'
    )
    val_dataset = BiomassDataset(
        val_data, cfg,
        transforms=get_transforms(cfg, 'val'),
        mode='val'
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # Model
    cat_dims = [
        train_df['State_enc'].nunique(),
        train_df['Species_enc'].nunique()
    ]
    model = BiomassModel(cfg, cat_dims).to(cfg.device)

    # Optimizer with different LR for backbone
    backbone_params = list(model.image_encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if 'image_encoder' not in n]

    optimizer = AdamW([
        {'params': backbone_params, 'lr': cfg.backbone_lr},
        {'params': other_params, 'lr': cfg.lr}
    ], weight_decay=cfg.weight_decay)

    # Scheduler
    num_training_steps = len(train_loader) * cfg.epochs // cfg.gradient_accumulation
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-7)

    # Loss
    criterion = BiomassLoss(cfg, physics_weight=0.1)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if cfg.mixed_precision else None

    # Training loop
    best_r2 = -float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, cfg, scaler
        )
        val_loss, r2_scores, mean_r2 = validate(model, val_loader, criterion, cfg)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(mean_r2)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val R² (mean): {mean_r2:.4f}")
        for name, r2 in r2_scores.items():
            print(f"  {name}: {r2:.4f}")

        # Save best model
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_r2': best_r2,
                'r2_scores': r2_scores,
                'cat_dims': cat_dims,
            }, cfg.output_dir / f'best_model_fold{fold}.pt')
            print(f"✓ Saved best model (R²: {best_r2:.4f})")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss Curve')

    axes[1].plot(history['val_r2'], label='Val R²', color='green')
    axes[1].axhline(y=best_r2, color='r', linestyle='--', label=f'Best: {best_r2:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R²')
    axes[1].legend()
    axes[1].set_title('Validation R²')

    plt.tight_layout()
    plt.savefig(cfg.output_dir / f'training_history_fold{fold}.png', dpi=150)
    plt.show()

    return best_r2, model

#%%
# Train all folds
all_scores = []
for fold in CFG.train_folds:
    best_r2, _ = train_fold(fold, train_df, CFG)
    all_scores.append(best_r2)

print(f"\n{'='*50}")
print(f"Training Complete!")
print(f"Mean CV R²: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
print(f"{'='*50}")

#%% [markdown]
# ## 7. Inference (Kaggle Submission)

#%%
@torch.no_grad()
def predict_tta(model, image, tabular, cat_features, cfg, n_tta=8):
    """Test-Time Augmentation for robust predictions."""
    model.eval()
    predictions = []

    # Original
    pred = model(image, tabular, cat_features)
    predictions.append(pred)

    # TTA augmentations
    if n_tta > 1:
        # Horizontal flip
        pred = model(torch.flip(image, dims=[3]), tabular, cat_features)
        predictions.append(pred)

    if n_tta > 2:
        # Vertical flip
        pred = model(torch.flip(image, dims=[2]), tabular, cat_features)
        predictions.append(pred)

    if n_tta > 4:
        # HFlip + VFlip
        pred = model(torch.flip(image, dims=[2, 3]), tabular, cat_features)
        predictions.append(pred)

    # Average predictions
    pred_mean = torch.stack(predictions).mean(dim=0)

    # Convert from log scale
    if cfg.use_log_transform:
        pred_mean = torch.expm1(pred_mean)

    return pred_mean

#%%
def create_submission(model, test_df, cfg, le_state, le_species):
    """Create submission.csv from model predictions."""
    model.eval()

    # Prepare test data (wide format)
    test_wide = test_df.copy()
    test_wide = test_wide.drop_duplicates(subset=['image_path'])

    # [IMPORTANT] test.csv에는 tabular features가 없음!
    # Train mean 값으로 대체 (또는 Image-only 모델 사용 권장)
    has_tabular = 'Pre_GSHH_NDVI' in test_wide.columns

    if not has_tabular:
        print("⚠️  Warning: No tabular features in test.csv!")
        print("   Using train mean values as placeholder.")
        test_wide['Pre_GSHH_NDVI'] = 0.657  # train mean
        test_wide['Height_Ave_cm'] = 7.596  # train mean
        test_wide['State'] = 'Vic'  # most common
        test_wide['Species'] = 'Ryegrass_Clover'  # most common

    # Encode (handle unseen categories)
    test_wide['State_enc'] = test_wide['State'].apply(
        lambda x: le_state.transform([x])[0] if x in le_state.classes_ else 0
    )
    test_wide['Species_enc'] = test_wide['Species'].apply(
        lambda x: le_species.transform([x])[0] if x in le_species.classes_ else 0
    )

    transforms = get_transforms(cfg, 'val')
    predictions = []

    for _, row in tqdm(test_wide.iterrows(), total=len(test_wide), desc='Inference'):
        # Load image
        img_path = cfg.data_dir / row['image_path']
        image = np.array(Image.open(img_path).convert('RGB'))
        image = transforms(image=image)['image']
        image = image.unsqueeze(0).to(cfg.device)

        # Tabular
        tabular = torch.tensor([[
            row['Pre_GSHH_NDVI'],
            row['Height_Ave_cm'] / 100.0
        ]], dtype=torch.float32).to(cfg.device)

        cat_features = torch.tensor([[
            row['State_enc'],
            row['Species_enc']
        ]], dtype=torch.long).to(cfg.device)

        # Predict with TTA
        pred = predict_tta(model, image, tabular, cat_features, cfg, n_tta=4)
        predictions.append(pred.cpu().numpy()[0])

    predictions = np.array(predictions)

    # Create submission in long format
    submission_rows = []
    for i, (_, row) in enumerate(test_wide.iterrows()):
        img_id = Path(row['image_path']).stem
        for j, target_name in enumerate(cfg.targets):
            submission_rows.append({
                'sample_id': f"{img_id}__{target_name}",
                'target': max(0, predictions[i, j])  # Ensure non-negative
            })

    submission = pd.DataFrame(submission_rows)
    submission.to_csv(cfg.output_dir / 'submission.csv', index=False)
    print(f"Submission saved: {len(submission)} rows")

    return submission

#%%
# Load best model and create submission
checkpoint = torch.load(CFG.output_dir / f'best_model_fold{CFG.train_folds[0]}.pt')
cat_dims = checkpoint['cat_dims']

model = BiomassModel(CFG, cat_dims).to(CFG.device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model with R²: {checkpoint['best_r2']:.4f}")

# Create submission (using test.csv)
test_df = pd.read_csv(CFG.data_dir / 'test.csv')
submission = create_submission(model, test_df, CFG, le_state, le_species)
submission.head(10)

#%% [markdown]
# ## 8. Save for Kaggle Upload
#
# 1. `outputs/best_model_fold0.pt` → Kaggle Dataset 업로드
# 2. 아래 inference 코드를 Kaggle Notebook에 복사

#%%
print(f"""
============================================================
Kaggle 제출 방법
============================================================

1. 모델 weights 업로드:
   - outputs/best_model_fold0.pt → Kaggle Datasets에 업로드
   - 이름: "csiro-biomass-weights" 등

2. Kaggle Notebook 생성:
   - 아래 inference 코드 복사
   - Add Data: 업로드한 weights dataset 추가
   - GPU 활성화
   - Internet Off
   - Submit

3. Inference 코드 (Kaggle용):
============================================================
""")

kaggle_inference_code = '''
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import torch.nn as nn
import torch.nn.functional as F

# [Model definition code here - copy from above]
# MultiScaleDINOv2, TabularEncoder, CrossAttentionFusion, BiomassModel classes

# Config
class CFG:
    data_dir = Path("/kaggle/input/csiro-biomass")
    weights_dir = Path("/kaggle/input/csiro-biomass-weights")  # 업로드한 weights
    img_size = 518
    targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'Dry_Total_g', 'GDM_g']
    num_targets = 5
    use_log_transform = True
    device = torch.device("cuda")

# Load model
checkpoint = torch.load(CFG.weights_dir / "best_model_fold0.pt")
cat_dims = checkpoint['cat_dims']
model = BiomassModel(CFG, cat_dims).to(CFG.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
test_df = pd.read_csv(CFG.data_dir / "test.csv")
# ... (inference code)

submission.to_csv("submission.csv", index=False)
'''

print(kaggle_inference_code)
