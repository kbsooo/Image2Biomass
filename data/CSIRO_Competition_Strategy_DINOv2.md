# CSIRO Biomass Competition Strategy
## DINOv2 Large 기반 접근법

---

## 📋 대회 핵심 정보 요약

### 평가 지표: Weighted R² Score
| Target | Weight | 특성 |
|--------|--------|------|
| **Dry_Total_g** | **0.5** | 가장 중요! 전체 바이오매스 |
| **GDM_g** | **0.2** | Green Dry Matter |
| Dry_Green_g | 0.1 | 녹색 식물 |
| Dry_Dead_g | 0.1 | 죽은 식물 |
| Dry_Clover_g | 0.1 | 클로버 (37.8% zero) |

### 현재 상황
- **1위 점수**: 0.79 R²
- **9일 남음** (마감 임박)
- **Public LB**: 53% 데이터 / **Private LB**: 47% 데이터
- **Research Code Competition**: 공개 노트북 없음

---

## 🔑 Discussion에서 얻은 핵심 인사이트

### 1. ⚠️ Overfitting 방지: Sampling_Date로 GroupKFold (126 votes)
```
같은 날짜에 촬영된 이미지들은 비슷한 조건(날씨, 조명 등)을 공유
→ 반드시 Sampling_Date 기준 GroupKFold 사용
→ 일반 KFold 사용 시 심각한 overfitting 발생
```

### 2. Height_Ave_cm과 Dead Biomass 관계 (79 votes)
- Height와 Dry_Dead_g의 상관관계 분석
- 메타데이터 활용 가능성 (단, Test에는 없음!)

### 3. Irish Grass Clover Dataset (67 votes)
- 외부 데이터셋 활용 가능성
- Pre-training 또는 추가 학습 데이터로 활용

### 4. Local CV vs LB Gap (41 votes)
- 큰 gap이 발생할 수 있음
- CV 전략이 매우 중요

### 5. PCA trick for target dependency (32 votes)
- 타겟 간 선형 종속성 해결
- Dry_Total = Dry_Clover + Dry_Dead + Dry_Green 관계

### 6. Post-Processing Findings (18 votes)
- 예측값 후처리 기법 존재

---

## 🏗️ DINOv2 Large 기반 모델 아키텍처

### DINOv2 선택 이유
1. **Self-supervised pretraining**: 자연 이미지에서 강력한 feature 추출
2. **Large 모델**: 1024 dim feature, 높은 표현력
3. **Frozen backbone 가능**: 작은 데이터셋에서 overfitting 방지
4. **Registration token**: 위치 정보 활용 가능

### 권장 아키텍처
```
┌─────────────────────────────────────────────────┐
│                Input Image                       │
│              (518 x 518 RGB)                     │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│         DINOv2 Large (ViT-L/14)                 │
│           Frozen or Fine-tuned                   │
│         Output: [CLS] + Patch tokens             │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│            Feature Aggregation                   │
│   Option 1: [CLS] token only (1024 dim)         │
│   Option 2: [CLS] + Global Avg Pool             │
│   Option 3: Attention Pooling                    │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│            Regression Head                       │
│   Linear(1024, 512) → ReLU → Dropout(0.3)       │
│   Linear(512, 256) → ReLU → Dropout(0.2)        │
│   Linear(256, 5) → 5 targets                    │
└─────────────────────────────────────────────────┘
```

### 코드 구조 예시
```python
import torch
import torch.nn as nn

class DINOv2BiomassModel(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()

        # DINOv2 Large backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)  # 5 targets
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # [B, 1024]

        # Predict targets
        outputs = self.head(features)
        return outputs
```

---

## 📊 학습 전략

### 1. Cross-Validation 설정
```python
from sklearn.model_selection import GroupKFold

# ⚠️ 반드시 Sampling_Date로 그룹화!
gkf = GroupKFold(n_splits=5)
groups = train_df.groupby('image_path')['Sampling_Date'].first()

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    # 같은 날짜의 이미지들이 train/val에 분리되지 않음
    pass
```

### 2. Data Augmentation
```python
import albumentations as A

train_transform = A.Compose([
    A.Resize(518, 518),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(518, 518),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

### 3. Loss Function
```python
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Competition weights
        self.weights = torch.tensor([0.1, 0.1, 0.1, 0.5, 0.2])
        # [Dry_Clover, Dry_Dead, Dry_Green, Dry_Total, GDM]

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        weighted_mse = (mse * self.weights.to(pred.device)).mean()
        return weighted_mse

# 또는 R² Loss 직접 구현
class R2Loss(nn.Module):
    def forward(self, pred, target, weights):
        ss_res = torch.sum(weights * (target - pred) ** 2)
        ss_tot = torch.sum(weights * (target - torch.mean(target)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return 1 - r2  # Loss로 변환
```

### 4. Target Transformation
```python
import numpy as np

# Log1p 변환 (우편향 분포 정규화)
y_train_log = np.log1p(y_train)
y_pred = np.expm1(model_pred)  # 역변환

# 또는 타겟 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train)
```

### 5. Training Configuration
```python
config = {
    'model': 'dinov2_vitl14',
    'image_size': 518,
    'batch_size': 16,  # GPU 메모리에 따라 조정
    'epochs': 30,
    'lr': 1e-4,  # Head만 학습 시
    'lr_backbone': 1e-6,  # Backbone fine-tune 시
    'weight_decay': 1e-4,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'T_0': 10,
    'T_mult': 2,
    'num_folds': 5,
    'seed': 42
}
```

---

## 🚀 성능 향상 전략

### Strategy 1: Multi-Scale Input
```python
# 여러 해상도로 예측 후 앙상블
scales = [448, 518, 588]
predictions = []
for scale in scales:
    pred = model(resize(image, scale))
    predictions.append(pred)
final_pred = torch.stack(predictions).mean(dim=0)
```

### Strategy 2: Test-Time Augmentation (TTA)
```python
def tta_predict(model, image, n_aug=8):
    preds = []
    transforms = [
        lambda x: x,                    # Original
        lambda x: torch.flip(x, [2]),   # H-flip
        lambda x: torch.flip(x, [3]),   # V-flip
        lambda x: torch.flip(x, [2,3]), # HV-flip
        lambda x: torch.rot90(x, 1, [2,3]),  # 90°
        lambda x: torch.rot90(x, 2, [2,3]),  # 180°
        lambda x: torch.rot90(x, 3, [2,3]),  # 270°
    ]

    for t in transforms[:n_aug]:
        aug_image = t(image)
        pred = model(aug_image)
        preds.append(pred)

    return torch.stack(preds).mean(dim=0)
```

### Strategy 3: Ensemble
```python
# 다양한 모델 앙상블
models = [
    'dinov2_vitl14',      # DINOv2 Large
    'dinov2_vitl14_reg',  # DINOv2 Large with registers
    'dinov2_vitg14',      # DINOv2 Giant (메모리 허용 시)
]

# Fold별 모델 앙상블
final_pred = np.zeros((len(test), 5))
for fold in range(5):
    model = load_model(f'model_fold{fold}.pth')
    final_pred += model.predict(test) / 5
```

### Strategy 4: Post-Processing
```python
# 1. Negative 값 클리핑
predictions = np.maximum(predictions, 0)

# 2. 타겟 간 관계 활용
# Dry_Total ≈ Dry_Clover + Dry_Dead + Dry_Green
sum_components = pred_clover + pred_dead + pred_green
pred_total = (pred_total + sum_components) / 2

# 3. GDM 관계 활용
# GDM ≈ Dry_Green + Dry_Clover (대략적)
pred_gdm = np.clip(pred_gdm, pred_green * 0.8, pred_total)
```

### Strategy 5: Pseudo Labeling (Optional)
```python
# Test 데이터에 대해 pseudo label 생성 후 재학습
# ⚠️ 주의: LB probing 위험
```

---

## 📅 9일 실행 계획

### Day 1-2: 기본 파이프라인 구축
- [ ] DINOv2 Large 기본 모델 구현
- [ ] GroupKFold CV 설정
- [ ] 기본 Augmentation 적용
- [ ] Baseline 제출 (목표: 0.65+)

### Day 3-4: 모델 최적화
- [ ] Backbone fine-tuning 실험
- [ ] 다양한 Head 구조 실험
- [ ] Loss function 실험
- [ ] 학습률/스케줄러 튜닝

### Day 5-6: 앙상블 구축
- [ ] 다양한 seed로 학습
- [ ] DINOv2 variants 실험
- [ ] Multi-scale 실험
- [ ] TTA 구현

### Day 7-8: 최종 튜닝
- [ ] 앙상블 가중치 최적화
- [ ] Post-processing 실험
- [ ] CV-LB correlation 분석
- [ ] Final submission 준비

### Day 9: 마지막 제출
- [ ] 최종 앙상블 제출
- [ ] 안전한 백업 제출

---

## ⚠️ 주의사항

1. **CV-LB Gap**: Local CV가 좋아도 LB에서 다를 수 있음
   - 다양한 fold의 결과를 평균
   - Private LB 47%를 고려한 robust한 모델 선택

2. **Overfitting 위험**
   - 357개의 작은 이미지 수
   - Backbone freezing 권장
   - Strong augmentation 필수

3. **타겟 특성**
   - Dry_Clover_g: 37.8% zero → 별도 처리 고려
   - Dry_Total_g: 가중치 0.5 → 가장 집중!

4. **Research Code Competition**
   - 코드 공개 제한
   - 자체 솔루션 개발 필수

---

## 💡 추가 아이디어

1. **외부 데이터**: Irish Grass Clover Dataset으로 pre-training
2. **Multi-task Learning**: 5개 타겟 동시 학습으로 regularization 효과
3. **Auxiliary Loss**: 타겟 간 관계를 auxiliary loss로 활용
4. **Attention Visualization**: 모델이 어디를 보는지 확인하여 insight 획득

---

*Strategy Document - Created: 2026-01-19*
*Competition: CSIRO - Image2Biomass Prediction*
*Model: DINOv2 Large (ViT-L/14)*
