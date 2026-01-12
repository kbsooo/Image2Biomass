# DINOv3 Gold Medal Strategy

> **목표**: LB 0.76+ (골드메달권)
> **현재**: LB 0.50 (11_hybrid_approach.py)
> **참고**: Public Notebook LB 0.70 (070.py)

---

## 1. 현재 상황 분석

### 1.1 왜 0.50에서 멈췄나?

| 문제점 | 상세 |
|--------|------|
| **잘못된 접근법** | LUPI/KD는 Teacher 성능(0.62)이 ceiling이 됨 |
| **작은 Backbone** | DINOv2 ViT-B/14 (86M params) |
| **Frozen Backbone** | 도메인 적응 불가 |
| **이미지 처리 방식** | 전체 이미지를 그대로 사용 |
| **TTA 없음** | 단일 예측 |

### 1.2 Public Notebook (0.70) 성공 요인

```python
# 070.py 핵심 구조
MODEL_NAME = "vit_large_patch16_dinov3_qkvb"  # ViT-Large (~300M)

# 이미지 분할 처리
left_image = image.crop((0, 0, mid_point, height))
right_image = image.crop((mid_point, 0, width, height))

# 3x TTA
transforms = [None, HorizontalFlip, VerticalFlip]

# FiLM fusion
context = (left_feat + right_feat) / 2
gamma, beta = self.film(context)
```

| 요소 | 효과 |
|------|------|
| ViT-Large | 더 강력한 feature extraction |
| Left/Right 분할 | 70×30cm 이미지의 공간 정보 활용 |
| FiLM fusion | 두 영역 간 context 공유 |
| 3x TTA | 예측 안정성 향상 |
| 5-Fold Ensemble | 일반화 성능 향상 |

### 1.3 학습/추론 분리 전략 (중요!)

070.py가 **1분**만에 실행되는 이유: **Inference Only** 코드이기 때문.

```python
# 070.py - 이미 학습된 모델을 로드
MODELS_DIR = '/kaggle/input/csiro-vit-large-dinov3/pytorch/default/1/csiro-dinov3-models'

for model_file in Path(models_dir).glob('*.pth'):
    model.load_state_dict(torch.load(model_file))  # ← 학습 없이 로드만!
```

반면 11_hybrid_approach.py는 **80분** 소요: 5-fold × (15 + 20) epochs = 175 epochs 학습

#### 우리의 Notebook 분리 전략

```
┌─────────────────────────────────────────────────────────────────────┐
│                        학습 파이프라인                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐     ┌──────────────────┐     ┌─────────────┐ │
│  │ 12_train.py      │     │ Kaggle Dataset   │     │ 12_infer.py │ │
│  │ (학습 Notebook)  │ ──▶ │ (가중치 저장소)  │ ──▶ │ (제출용)    │ │
│  │                  │     │                  │     │             │ │
│  │ • 5-Fold 학습    │     │ • fold0.pth     │     │ • 로드만    │ │
│  │ • ~80분 소요     │     │ • fold1.pth     │     │ • ~1분 소요 │ │
│  │ • GPU 집약적     │     │ • fold2.pth     │     │ • 제출 가능 │ │
│  │                  │     │ • fold3.pth     │     │             │ │
│  │                  │     │ • fold4.pth     │     │             │ │
│  └──────────────────┘     └──────────────────┘     └─────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 파일 구조

```
notebooks/
├── 12_dinov3_train.py      # 학습용 (Kaggle에서 실행, 가중치 저장)
├── 12_dinov3_infer.py      # 추론용 (제출용, 1분 내 실행)
└── 12_dinov3_local.py      # 로컬 테스트용 (선택사항)

Kaggle Datasets:
├── pretrained-weights-biomass/
│   ├── dinov2/dinov2_vitb14.pth
│   ├── dinov3_large/model.safetensors    # ← 새로 추가
│   └── ...
│
└── csiro-dinov3-trained/                  # ← 학습된 모델 저장용
    ├── fold0.pth
    ├── fold1.pth
    ├── fold2.pth
    ├── fold3.pth
    └── fold4.pth
```

#### 워크플로우

```
Step 1: 학습 (12_dinov3_train.py)
   ├── Kaggle Notebook에서 실행 (GPU P100/T4)
   ├── 5-Fold 학습 수행 (~80분)
   ├── /kaggle/working/에 fold*.pth 저장
   └── "Save & Run All" → Output으로 가중치 생성

Step 2: Dataset 생성
   ├── Notebook Output을 새 Dataset으로 저장
   └── 예: "csiro-dinov3-trained"

Step 3: 추론 (12_dinov3_infer.py)
   ├── 학습된 가중치 Dataset 연결
   ├── 모델 로드 → 예측 → submission.csv
   └── 실행 시간: ~1분

Step 4: 제출
   └── 12_dinov3_infer.py를 Competition에 제출
```

#### 장점

| 항목 | 설명 |
|------|------|
| **시간 효율** | 제출 시 1분만 소요, 반복 제출 용이 |
| **디버깅 용이** | 학습/추론 분리로 문제 파악 쉬움 |
| **버전 관리** | 가중치를 Dataset으로 관리, 롤백 가능 |
| **리소스 절약** | 추론만 할 때 GPU 시간 절약 |

---

## 2. Phase 1: 0.70 재현 (12_dinov3_baseline.py)

### 2.1 모델 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Image                               │
│                    (70cm × 30cm quadrat)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌──────────────┐    ┌──────────────┐
            │  Left Half   │    │  Right Half  │
            │  (35×30cm)   │    │  (35×30cm)   │
            └──────────────┘    └──────────────┘
                    │                   │
                    ▼                   ▼
            ┌──────────────┐    ┌──────────────┐
            │   Resize     │    │   Resize     │
            │  512 × 512   │    │  512 × 512   │
            └──────────────┘    └──────────────┘
                    │                   │
                    ▼                   ▼
            ┌──────────────────────────────────┐
            │      DINOv3 ViT-Large Backbone   │
            │   (vit_large_patch16_dinov3_qkvb)│
            │         ~300M parameters         │
            │        output: 1024-dim          │
            └──────────────────────────────────┘
                    │                   │
                    ▼                   ▼
              left_feat            right_feat
               (1024)                (1024)
                    │                   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    Context      │
                    │ (left+right)/2  │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │      FiLM       │
                    │  γ, β 생성      │
                    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          left × (1+γ) + β    right × (1+γ) + β
                    │                   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Concatenate   │
                    │   (2048-dim)    │
                    └─────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │  Head    │    │  Head    │    │  Head    │
        │  Green   │    │  Clover  │    │  Dead    │
        └──────────┘    └──────────┘    └──────────┘
              │               │               │
              ▼               ▼               ▼
          Softplus        Softplus        Softplus
              │               │               │
              ▼               ▼               ▼
           Green           Clover           Dead
              │               │               │
              └───────────────┼───────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Physics Layer   │
                    │ GDM = G + C     │
                    │ Total = GDM + D │
                    └─────────────────┘
                              │
                              ▼
            [Green, Dead, Clover, GDM, Total]
```

### 2.2 코드 구현

#### 2.2.1 Configuration

```python
class CFG:
    # === Paths ===
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass")
    OUTPUT_DIR = Path("/kaggle/working")

    # === Model ===
    model_name = "vit_large_patch16_dinov3_qkvb"
    backbone_dim = 1024  # ViT-Large output dimension
    img_size = (512, 512)

    # === Training ===
    n_folds = 5
    epochs = 15
    batch_size = 8  # ViT-Large는 메모리 많이 사용
    lr = 1e-4
    weight_decay = 1e-4

    # === TTA ===
    tta_transforms = ['none', 'hflip', 'vflip']

    seed = 42
    num_workers = 4
    device = "cuda"
```

#### 2.2.2 Dataset with Left/Right Split

```python
class BiomassDataset(Dataset):
    """
    핵심: 이미지를 Left/Right로 분할하여 반환
    - 70cm × 30cm quadrat → 가로로 긴 이미지
    - 좌우 영역의 vegetation이 다를 수 있음
    - 각 영역을 독립적으로 처리 후 fusion
    """
    def __init__(self, df, transform=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img = Image.open(CFG.DATA_PATH / row['image_path']).convert('RGB')
        width, height = img.size
        mid_point = width // 2

        # Split into left and right halves
        # Insight: 좌우 영역의 biomass 분포가 다를 수 있으므로
        #          각각을 독립적인 "view"로 처리
        left_img = img.crop((0, 0, mid_point, height))
        right_img = img.crop((mid_point, 0, width, height))

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        if self.mode == 'train':
            targets = torch.tensor([
                row['Dry_Green_g'],
                row['Dry_Clover_g'],
                row['Dry_Dead_g']
            ], dtype=torch.float32)
            return left_img, right_img, targets
        else:
            return left_img, right_img, row['image_id']
```

#### 2.2.3 FiLM Module

```python
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation

    핵심 아이디어:
    - Left/Right 영역의 평균 feature를 context로 사용
    - 이 context로 각 영역의 feature를 modulate
    - γ (scale)와 β (shift)를 학습하여 cross-region interaction 구현

    수식: output = input × (1 + γ) + β
    - γ는 Tanh로 [-1, 1] 범위로 제한 → (1+γ)는 [0, 2] 범위
    - 이는 feature를 0~2배로 scaling하는 효과
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim * 2)  # γ와 β 동시 생성
        )

    def forward(self, context):
        gamma_beta = self.mlp(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma, beta
```

#### 2.2.4 Main Model

```python
class CSIROModel(nn.Module):
    """
    DINOv3 ViT-Large + FiLM + Physics-constrained Heads
    """
    def __init__(self, model_name, pretrained=True, dropout=0.1):
        super().__init__()

        # DINOv3 ViT-Large backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        feat_dim = self.backbone.num_features  # 1024 for ViT-Large

        # FiLM for cross-region modulation
        self.film = FiLM(feat_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Independent heads for each target
        # Insight: 각 biomass 유형은 다른 visual pattern을 가짐
        #          Green: 녹색 영역, Dead: 갈색/노란색, Clover: 특정 잎 모양
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

        # Softplus for non-negative outputs (biomass는 항상 ≥ 0)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, left_img, right_img):
        # Extract features from both halves
        left_feat = self.backbone(left_img)   # (B, 1024)
        right_feat = self.backbone(right_img) # (B, 1024)

        # Compute context as average of both views
        context = (left_feat + right_feat) / 2

        # Generate modulation parameters
        gamma, beta = self.film(context)

        # Modulate features
        # Insight: context 정보로 각 영역의 feature를 조정
        #          예: 한쪽이 주로 green이면 다른쪽 해석에 영향
        left_mod = left_feat * (1 + gamma) + beta
        right_mod = right_feat * (1 + gamma) + beta

        # Concatenate modulated features
        combined = torch.cat([left_mod, right_mod], dim=1)  # (B, 2048)
        combined = self.dropout(combined)

        # Predict independent targets
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))

        # Physics constraints
        # GDM (Green Dry Matter) = Green + Clover
        # Total = GDM + Dead
        gdm = green + clover
        total = gdm + dead

        # Return all 5 targets: [Green, Dead, Clover, GDM, Total]
        return torch.cat([green, dead, clover, gdm, total], dim=1)
```

### 2.3 Training Pipeline

```python
def train_fold(fold, train_df, cfg):
    """Single fold training"""

    # Split data
    train_data = train_df[train_df['fold'] != fold]
    val_data = train_df[train_df['fold'] == fold]

    # Transforms
    train_transform = T.Compose([
        T.Resize(cfg.img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets & Loaders
    train_ds = BiomassDataset(train_data, train_transform, 'train')
    val_ds = BiomassDataset(val_data, val_transform, 'train')

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2,
                            shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = CSIROModel(cfg.model_name, pretrained=True).to(cfg.device)

    # Optimizer with layer-wise learning rate decay
    # Insight: Backbone은 이미 pretrained이므로 낮은 lr,
    #          Head는 scratch이므로 높은 lr
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head_green.parameters()) + \
                  list(model.head_clover.parameters()) + \
                  list(model.head_dead.parameters()) + \
                  list(model.film.parameters())

    optimizer = AdamW([
        {'params': backbone_params, 'lr': cfg.lr * 0.1},  # Backbone: 1e-5
        {'params': head_params, 'lr': cfg.lr}              # Heads: 1e-4
    ], weight_decay=cfg.weight_decay)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader),
        num_training_steps=len(train_loader) * cfg.epochs
    )

    # Training loop
    best_score = -float('inf')

    for epoch in range(cfg.epochs):
        # Train
        model.train()
        train_loss = 0

        for left, right, targets in tqdm(train_loader):
            left = left.to(cfg.device)
            right = right.to(cfg.device)
            targets = targets.to(cfg.device)

            optimizer.zero_grad()

            outputs = model(left, right)
            # Loss on independent targets only (Green, Clover, Dead)
            # GDM, Total은 자동으로 physics constraint 만족
            loss = F.mse_loss(outputs[:, [0, 2, 1]], targets)  # Reorder to match

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for left, right, targets in val_loader:
                left = left.to(cfg.device)
                right = right.to(cfg.device)

                outputs = model(left, right)
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        # Compute full targets for metric
        full_targets = np.zeros((len(val_targets), 5))
        full_targets[:, 0] = val_targets[:, 0]  # Green
        full_targets[:, 1] = val_targets[:, 2]  # Dead
        full_targets[:, 2] = val_targets[:, 1]  # Clover
        full_targets[:, 3] = val_targets[:, 0] + val_targets[:, 1]  # GDM
        full_targets[:, 4] = full_targets[:, 3] + val_targets[:, 2]  # Total

        cv_score = competition_metric(full_targets, val_preds)

        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {train_loss/len(train_loader):.4f} | CV: {cv_score:.4f}")

        if cv_score > best_score:
            best_score = cv_score
            torch.save(model.state_dict(), cfg.OUTPUT_DIR / f'model_fold{fold}.pth')
            print(f"  ✓ New best!")

    return best_score
```

### 2.4 Test-Time Augmentation (TTA)

```python
def get_tta_dataloaders(df, cfg):
    """
    3x TTA: 원본, HFlip, VFlip

    Insight: TTA는 ensemble의 일종으로, 다양한 view에서의 예측을 평균
    - 비용: inference 시간 3배
    - 효과: 일반적으로 1-3% 성능 향상
    """
    loaders = []

    transforms_list = [
        # Original
        T.Compose([
            T.Resize(cfg.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Horizontal Flip
        T.Compose([
            T.Resize(cfg.img_size),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Vertical Flip
        T.Compose([
            T.Resize(cfg.img_size),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]

    for transform in transforms_list:
        dataset = BiomassDataset(df, transform, mode='test')
        loader = DataLoader(dataset, batch_size=cfg.batch_size,
                           shuffle=False, num_workers=cfg.num_workers)
        loaders.append(loader)

    return loaders


def predict_with_tta(models, tta_loaders, cfg):
    """
    Ensemble prediction: 5 folds × 3 TTA = 15 predictions averaged
    """
    all_preds = []
    all_ids = []

    for loader in tta_loaders:
        loader_preds = []

        for left, right, image_ids in tqdm(loader):
            left = left.to(cfg.device)
            right = right.to(cfg.device)

            batch_preds = []
            for model in models:
                model.eval()
                with torch.no_grad():
                    outputs = model(left, right)
                    batch_preds.append(outputs.cpu().numpy())

            # Average across folds
            avg_pred = np.mean(batch_preds, axis=0)
            loader_preds.append(avg_pred)

            if len(all_ids) == 0:
                all_ids.extend(image_ids)

        all_preds.append(np.concatenate(loader_preds))

    # Average across TTA
    final_preds = np.mean(all_preds, axis=0)

    return final_preds, all_ids
```

### 2.5 예상 결과

| 구성요소 | 예상 기여도 |
|----------|------------|
| DINOv3 ViT-Large | +0.10 (vs ViT-Base) |
| Left/Right Split + FiLM | +0.05 |
| 5-Fold Ensemble | +0.03 |
| 3x TTA | +0.02 |
| **총 예상** | **~0.70** |

---

## 3. Phase 2: 0.76+ 골드메달 전략

### 3.1 개선 전략 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1 완료: ~0.70                                             │
│  ─────────────────────────────────────────────────────────────── │
│  3.2 Extended TTA (8x)                        → +0.01~0.02      │
│  3.3 Heavy Augmentation                       → +0.02~0.03      │
│  3.4 Multi-Backbone Ensemble                  → +0.03~0.05      │
│  3.5 Pseudo-Labeling                          → +0.02~0.03      │
│  3.6 Loss Function Optimization               → +0.01~0.02      │
│  ─────────────────────────────────────────────────────────────── │
│  목표: 0.76+                                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Extended TTA (8x)

```python
def get_extended_tta_transforms():
    """
    8x TTA: Original + 3 Flips + 4 Rotations

    변환:
    1. Original
    2. HFlip
    3. VFlip
    4. HFlip + VFlip (= 180° rotation)
    5. Rotate 90°
    6. Rotate 90° + HFlip
    7. Rotate 270°
    8. Rotate 270° + HFlip

    효과: D4 symmetry group의 모든 변환 커버
    """
    base = [
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    transforms_list = [
        # 1. Original
        T.Compose(base),

        # 2. HFlip
        T.Compose([T.Resize((512, 512)), T.RandomHorizontalFlip(p=1.0)] + base[1:]),

        # 3. VFlip
        T.Compose([T.Resize((512, 512)), T.RandomVerticalFlip(p=1.0)] + base[1:]),

        # 4. HFlip + VFlip
        T.Compose([
            T.Resize((512, 512)),
            T.RandomHorizontalFlip(p=1.0),
            T.RandomVerticalFlip(p=1.0)
        ] + base[1:]),

        # 5. Rotate 90°
        T.Compose([
            T.Resize((512, 512)),
            T.Lambda(lambda x: x.rotate(90))
        ] + base[1:]),

        # 6. Rotate 90° + HFlip
        T.Compose([
            T.Resize((512, 512)),
            T.Lambda(lambda x: x.rotate(90)),
            T.RandomHorizontalFlip(p=1.0)
        ] + base[1:]),

        # 7. Rotate 270°
        T.Compose([
            T.Resize((512, 512)),
            T.Lambda(lambda x: x.rotate(270))
        ] + base[1:]),

        # 8. Rotate 270° + HFlip
        T.Compose([
            T.Resize((512, 512)),
            T.Lambda(lambda x: x.rotate(270)),
            T.RandomHorizontalFlip(p=1.0)
        ] + base[1:]),
    ]

    return transforms_list
```

### 3.3 Heavy Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_heavy_train_transform():
    """
    Heavy augmentation for better generalization

    핵심 전략:
    1. Geometric: Flip, Rotate, ShiftScaleRotate
    2. Color: ColorJitter, RandomBrightnessContrast, HueSaturationValue
    3. Noise: GaussNoise, GaussianBlur
    4. Cutout: CoarseDropout (일부 영역 제거)
    5. Mixup/CutMix: 배치 레벨에서 적용
    """
    return A.Compose([
        A.Resize(512, 512),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            p=0.5
        ),

        # Color augmentations
        A.OneOf([
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.3, hue=0.05, p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2, p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20, p=1.0
            ),
        ], p=0.7),

        # Noise & Blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # Cutout (regularization)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=0.3
        ),

        # Normalize
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class MixupCutmix:
    """
    Batch-level augmentation

    Mixup: 두 이미지와 라벨을 linear interpolation
    CutMix: 한 이미지의 일부를 다른 이미지로 대체

    효과: 결정 경계 smoothing, overfitting 방지
    """
    def __init__(self, mixup_alpha=0.4, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, images, targets):
        if np.random.rand() > self.prob:
            return images, targets

        batch_size = images.size(0)
        indices = torch.randperm(batch_size)

        if np.random.rand() > 0.5:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_images = lam * images + (1 - lam) * images[indices]
            mixed_targets = lam * targets + (1 - lam) * targets[indices]
        else:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

            # Random box
            W, H = images.size(3), images.size(2)
            cut_w = int(W * np.sqrt(1 - lam))
            cut_h = int(H * np.sqrt(1 - lam))
            cx, cy = np.random.randint(W), np.random.randint(H)

            x1 = np.clip(cx - cut_w // 2, 0, W)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            y2 = np.clip(cy + cut_h // 2, 0, H)

            mixed_images = images.clone()
            mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]

            # Adjust lambda based on actual box size
            lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
            mixed_targets = lam * targets + (1 - lam) * targets[indices]

        return mixed_images, mixed_targets
```

### 3.4 Multi-Backbone Ensemble

```python
"""
Multi-Backbone Strategy

핵심 아이디어:
- 각 backbone은 다른 inductive bias를 가짐
- ViT: global attention, long-range dependency
- ConvNeXt: local patterns, hierarchical features
- EfficientNet: compound scaling, efficient computation

앙상블 시 각자의 강점이 결합됨
"""

BACKBONES = {
    'dinov3_large': {
        'name': 'vit_large_patch16_dinov3_qkvb',
        'feat_dim': 1024,
        'weight': 0.4,  # 메인 backbone
    },
    'convnext_large': {
        'name': 'convnext_large.fb_in22k_ft_in1k',
        'feat_dim': 1536,
        'weight': 0.3,
    },
    'swin_v2_base': {
        'name': 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
        'feat_dim': 1024,
        'weight': 0.2,
    },
    'efficientnet_v2_l': {
        'name': 'tf_efficientnetv2_l.in21k_ft_in1k',
        'feat_dim': 1280,
        'weight': 0.1,
    }
}


def train_multi_backbone(train_df, cfg):
    """
    각 backbone별로 독립적으로 학습

    최종 예측 = Σ (weight_i × prediction_i)
    """
    all_models = {}

    for backbone_key, backbone_cfg in BACKBONES.items():
        print(f"\n{'='*60}")
        print(f"Training {backbone_key}")
        print(f"{'='*60}")

        models = []
        for fold in range(cfg.n_folds):
            model = CSIROModel(
                backbone_cfg['name'],
                feat_dim=backbone_cfg['feat_dim'],
                pretrained=True
            )

            # Train fold...
            best_score = train_fold(fold, train_df, model, cfg)

            # Load best
            model.load_state_dict(
                torch.load(cfg.OUTPUT_DIR / f'{backbone_key}_fold{fold}.pth')
            )
            models.append(model)

        all_models[backbone_key] = models

    return all_models


def predict_multi_backbone(all_models, test_loader, cfg):
    """
    Weighted ensemble prediction
    """
    all_preds = []
    weights = []

    for backbone_key, models in all_models.items():
        backbone_cfg = BACKBONES[backbone_key]

        # Predict with all folds
        preds = predict_with_tta(models, test_loader, cfg)
        all_preds.append(preds)
        weights.append(backbone_cfg['weight'])

    # Weighted average
    weights = np.array(weights) / sum(weights)  # Normalize
    final_preds = sum(w * p for w, p in zip(weights, all_preds))

    return final_preds
```

### 3.5 Pseudo-Labeling

```python
"""
Iterative Pseudo-Labeling Strategy

핵심 아이디어:
1. Train set (357 images)으로 모델 학습
2. Test set (800+ images)에 대해 예측
3. 높은 confidence 예측을 pseudo-label로 사용
4. Train + Pseudo-labeled data로 재학습
5. 반복

주의사항:
- Confirmation bias 위험: 잘못된 예측이 강화될 수 있음
- 해결책: 높은 confidence만 사용, 점진적으로 추가
"""

def iterative_pseudo_labeling(train_df, test_df, cfg, n_iterations=3):
    """
    Iterative pseudo-labeling pipeline
    """

    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"Pseudo-Labeling Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")

        # 1. Train on current data
        models = []
        for fold in range(cfg.n_folds):
            model = CSIROModel(cfg.model_name, pretrained=True)
            train_fold(fold, train_df, model, cfg)
            model.load_state_dict(
                torch.load(cfg.OUTPUT_DIR / f'model_fold{fold}.pth')
            )
            models.append(model)

        # 2. Predict on test set with uncertainty
        test_preds, test_ids, uncertainties = predict_with_uncertainty(
            models, test_df, cfg
        )

        # 3. Select high-confidence samples
        # Uncertainty가 낮은 (= confidence가 높은) 샘플 선택
        confidence_threshold = np.percentile(uncertainties, 30)  # Top 30%
        high_conf_mask = uncertainties < confidence_threshold

        print(f"Selected {high_conf_mask.sum()} high-confidence samples")

        # 4. Create pseudo-labeled dataframe
        pseudo_df = test_df[high_conf_mask].copy()
        pseudo_df['Dry_Green_g'] = test_preds[high_conf_mask, 0]
        pseudo_df['Dry_Dead_g'] = test_preds[high_conf_mask, 1]
        pseudo_df['Dry_Clover_g'] = test_preds[high_conf_mask, 2]
        pseudo_df['GDM_g'] = test_preds[high_conf_mask, 3]
        pseudo_df['Dry_Total_g'] = test_preds[high_conf_mask, 4]
        pseudo_df['is_pseudo'] = True

        # 5. Combine with original training data
        train_df['is_pseudo'] = False
        train_df = pd.concat([train_df, pseudo_df], ignore_index=True)

        print(f"Total training samples: {len(train_df)}")

    return train_df


def predict_with_uncertainty(models, test_df, cfg):
    """
    Prediction with uncertainty estimation using MC Dropout

    MC Dropout:
    - Inference 시에도 dropout 활성화
    - 여러 번 예측하여 분산 계산
    - 분산이 크면 uncertainty가 높음
    """
    n_mc_samples = 10

    all_preds = []

    for _ in range(n_mc_samples):
        preds = []
        for model in models:
            model.train()  # Enable dropout
            with torch.no_grad():
                for left, right, _ in test_loader:
                    left = left.to(cfg.device)
                    right = right.to(cfg.device)
                    outputs = model(left, right)
                    preds.append(outputs.cpu().numpy())

        all_preds.append(np.concatenate(preds))

    all_preds = np.array(all_preds)  # (n_mc, n_samples, 5)

    mean_preds = all_preds.mean(axis=0)
    uncertainties = all_preds.std(axis=0).mean(axis=1)  # Average std across targets

    return mean_preds, test_ids, uncertainties
```

### 3.6 Loss Function Optimization

```python
"""
Loss Function Optimization

MSE의 문제점:
- Outlier에 민감
- 모든 error를 동일하게 취급

개선 방향:
1. Huber Loss: outlier에 robust
2. Weighted MSE: 중요한 target에 가중치
3. R² Loss: 직접 metric 최적화
"""

class WeightedMSELoss(nn.Module):
    """
    Target별 가중치를 적용한 MSE

    대회 metric이 weighted R²이므로,
    학습 시에도 동일한 가중치 적용
    """
    def __init__(self):
        super().__init__()
        # Competition weights: Green=0.1, Dead=0.1, Clover=0.1, GDM=0.2, Total=0.5
        self.weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])

    def forward(self, pred, target):
        weights = self.weights.to(pred.device)
        mse = (pred - target) ** 2
        weighted_mse = (mse * weights).mean()
        return weighted_mse


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1)

    δ보다 작은 error: L2 (smooth)
    δ보다 큰 error: L1 (linear, outlier에 robust)
    """
    def __init__(self, delta=10.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        return loss.mean()


class R2Loss(nn.Module):
    """
    Directly optimize R² score

    R² = 1 - SS_res / SS_tot
    Loss = 1 - R² = SS_res / SS_tot

    주의: SS_tot이 0에 가까우면 불안정
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])

    def forward(self, pred, target):
        weights = self.weights.to(pred.device)

        # Per-target R² loss
        ss_res = ((target - pred) ** 2).sum(dim=0)
        ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0) + self.eps

        r2_loss = ss_res / ss_tot  # 1 - R²
        weighted_loss = (r2_loss * weights).sum()

        return weighted_loss


class CombinedLoss(nn.Module):
    """
    여러 loss의 조합

    MSE: 기본 regression loss
    Huber: outlier robustness
    R²: metric alignment
    """
    def __init__(self, mse_weight=0.5, huber_weight=0.3, r2_weight=0.2):
        super().__init__()
        self.mse = WeightedMSELoss()
        self.huber = HuberLoss(delta=10.0)
        self.r2 = R2Loss()

        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.r2_weight = r2_weight

    def forward(self, pred, target):
        return (
            self.mse_weight * self.mse(pred, target) +
            self.huber_weight * self.huber(pred, target) +
            self.r2_weight * self.r2(pred, target)
        )
```

---

## 4. Implementation Roadmap

### 4.1 Timeline (학습/추론 분리 반영)

```
Phase 0: 환경 준비
├── Step 0-1: DINOv3 ViT-Large 가중치 다운로드 (HuggingFace)
├── Step 0-2: pretrained-weights-biomass Dataset 업데이트
└── 소요: ~30분

Phase 1: 0.70 재현
├── Step 1-1: 12_dinov3_train.py 작성 (학습 코드)
├── Step 1-2: 12_dinov3_infer.py 작성 (추론 코드)
├── Step 1-3: Kaggle에서 train.py 실행 (~80분)
├── Step 1-4: Output을 Dataset으로 저장 (csiro-dinov3-trained)
├── Step 1-5: infer.py로 제출 (~1분)
└── 목표: LB 0.70

Phase 2: 0.76+ 개선
├── Step 2-1: 8x TTA 추가 → 0.71~0.72
├── Step 2-2: Heavy Augmentation → 0.72~0.74
├── Step 2-3: Multi-Backbone Ensemble → 0.74~0.76
├── Step 2-4: Pseudo-Labeling → 0.76+
└── 목표: LB 0.76+
```

### 4.2 Notebook 구조

```
notebooks/
├── 12_dinov3_train.py      # 학습용 Notebook
│   ├── 5-Fold 학습
│   ├── CV 점수 출력
│   └── fold*.pth 저장 → /kaggle/working/
│
├── 12_dinov3_infer.py      # 추론용 Notebook (제출용)
│   ├── 학습된 가중치 로드
│   ├── 3x TTA 예측
│   └── submission.csv 생성
│   └── 실행시간: ~1분
│
└── (Phase 2)
    ├── 13_multi_backbone_train.py
    └── 13_multi_backbone_infer.py
```

### 4.3 Required Kaggle Datasets

```yaml
# Dataset 1: 사전학습 가중치 (pretrained-weights-biomass)
pretrained-weights-biomass:
  existing:
    - dinov2/dinov2_vitb14.pth (346MB)
    - efficientnet_b4/efficientnet_b4.pth (78MB)

  to_add_phase1:
    - dinov3_large/model.safetensors (~1.2GB)

  to_add_phase2:
    - convnext_large/model.safetensors (~800MB)
    - swinv2_base/model.safetensors (~400MB)

# Dataset 2: 학습된 모델 가중치 (새로 생성)
csiro-dinov3-trained:
  created_by: 12_dinov3_train.py의 Output
  contents:
    - fold0.pth
    - fold1.pth
    - fold2.pth
    - fold3.pth
    - fold4.pth
  size: ~6GB (1.2GB × 5 folds)
```

### 4.4 실행 순서

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 0: 가중치 준비                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. HuggingFace에서 DINOv3 가중치 다운로드                                │
│    $ python download_weights.py                                         │
│                                                                         │
│ 2. Kaggle Dataset 업데이트                                              │
│    $ kaggle datasets version -p ~/kaggle_weights -m "Add DINOv3"        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1-A: 학습 (12_dinov3_train.py)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. Kaggle Notebook 생성                                                 │
│ 2. GPU 선택 (T4 × 2 권장)                                               │
│ 3. Dataset 연결: pretrained-weights-biomass, csiro-biomass              │
│ 4. "Save & Run All" 실행 (~80분)                                        │
│ 5. Output 확인: fold0~4.pth 생성됨                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1-B: Dataset 저장                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. 학습 Notebook의 Output 탭으로 이동                                   │
│ 2. "New Dataset" 클릭                                                   │
│ 3. 이름: "csiro-dinov3-trained"                                         │
│ 4. 저장 완료                                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1-C: 추론 & 제출 (12_dinov3_infer.py)                             │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. 새 Kaggle Notebook 생성                                              │
│ 2. Dataset 연결:                                                        │
│    - pretrained-weights-biomass (DINOv3 backbone)                       │
│    - csiro-dinov3-trained (학습된 fold weights)                         │
│    - csiro-biomass (competition data)                                   │
│ 3. "Save & Run All" 실행 (~1분)                                         │
│ 4. "Submit to Competition" 클릭                                         │
│ 5. LB 점수 확인 → 목표: 0.70                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.5 Expected Results

| Phase | 구성 | 예상 LB |
|-------|------|--------|
| Phase 1 | DINOv3-L + FiLM + 5-Fold + 3x TTA | 0.70 |
| Phase 2a | + 8x TTA | 0.71~0.72 |
| Phase 2b | + Heavy Aug | 0.72~0.74 |
| Phase 2c | + Multi-Backbone | 0.74~0.76 |
| Phase 2d | + Pseudo-Labeling | 0.76+ |

---

## 5. Key Insights Summary

### 5.1 왜 이전 접근법(LUPI/KD)이 실패했나?

1. **Teacher Ceiling**: Teacher가 0.62인데 Student가 0.9를 달성하는 건 불가능
2. **복잡도 vs 효과**: 복잡한 구조가 항상 좋은 것은 아님
3. **Image-only로도 충분**: 0.70은 순수 이미지만으로 달성 가능

### 5.2 성공의 핵심 요소

1. **Backbone Size Matters**: ViT-Large >> ViT-Base
2. **Image Split**: 가로로 긴 이미지의 공간 정보 활용
3. **FiLM Fusion**: Cross-region context sharing
4. **Ensemble**: Fold + TTA + Multi-backbone
5. **Physics Constraints**: GDM = G+C, Total = GDM+D

### 5.3 골드메달을 위한 차별화

- 단순히 0.70 notebook 복제로는 골드 불가능
- **Multi-backbone ensemble + Pseudo-labeling**이 핵심
- Heavy augmentation으로 357개 데이터의 한계 극복

---

## 6. Appendix: Weight Download Script

```python
"""
가중치 다운로드 스크립트

실행:
cd ~/kaggle_weights
python download_weights.py
"""

from huggingface_hub import hf_hub_download
import os

MODELS = {
    'dinov3_large': 'timm/vit_large_patch16_dinov3_qkvb.lvd1689m',
    'convnext_large': 'timm/convnext_large.fb_in22k_ft_in1k',
    'swinv2_base': 'timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
}

for name, repo_id in MODELS.items():
    print(f"Downloading {name}...")
    os.makedirs(name, exist_ok=True)

    hf_hub_download(
        repo_id=repo_id,
        filename='model.safetensors',
        local_dir=name
    )

    print(f"✓ {name} downloaded")
```
