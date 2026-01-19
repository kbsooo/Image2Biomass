# 🚀 Breakthrough Strategy: 0.70 → 0.79+

## 📊 현재 상황 분석

### 현재 점수
- **Your Best Public LB**: 0.70
- **1위 Public LB**: 0.79
- **Gap**: 0.09 (상당히 큰 차이)

### 현재 코드 분석

| Version | 특징 | 문제점 |
|---------|------|--------|
| v20/v26 | DINOv3 Large + FiLM + Dual View | 기본 베이스라인 |
| v22 | Frozen backbone + 작은 Head | 제한된 학습 |
| v25 | VegIdx Late Fusion | 추가 정보지만 효과 제한적 |
| v27 | 단순 앙상블 (Simple/Rank Average) | 최적화되지 않은 앙상블 |

### 🔴 핵심 문제점 발견

#### 1. **CV 전략 오류** ⚠️ (가장 심각)
```python
# 현재 코드
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
df['strat_key'] = df['State'] + '_' + df['Month'].astype(str)
groups = df['image_id']  # image_id로 그룹핑
```

**문제**: Discussion에서 126 votes를 받은 핵심 인사이트는 **Sampling_Date로 그룹핑**해야 한다는 것!
- 같은 날짜에 촬영된 이미지들은 비슷한 조건 공유
- `image_id`로 그룹핑하면 같은 날짜의 다른 이미지가 train/val에 분리됨
- **심각한 data leakage → overfitting**

#### 2. **이미지 해상도 제한**
```python
img_size = (512, 512)
```
- DINOv2/v3의 최적 해상도는 **518x518** (14로 나눠떨어짐)
- 또는 더 큰 해상도 (560, 616, 672 등)

#### 3. **TTA 미사용**
- Inference에서 TTA를 전혀 사용하지 않음
- 목초지 이미지는 회전/플립에 불변 → TTA 효과적

#### 4. **앙상블 최적화 부족**
```python
ENSEMBLE_METHOD = "simple"  # 단순 평균만 사용
```
- 가중치 최적화 없음
- 모델 다양성 부족 (모두 같은 backbone)

#### 5. **Loss Function**
```python
main_loss = F.mse_loss(pred, main_targets)  # 단순 MSE
```
- 대회 평가 지표(Weighted R²)와 다른 loss 사용
- Dry_Total_g가 50% 가중치인데 동일하게 취급

#### 6. **데이터 활용 부족**
- External data (Irish Grass Clover) 미사용
- Pseudo labeling 미적용

---

## 🎯 Breakthrough 전략 (우선순위 순)

### 🔥 Priority 1: CV 전략 수정 (예상 +0.03~0.05)

**가장 중요!! 이것만 고쳐도 큰 향상 예상**

```python
# ❌ 현재 (잘못된 방법)
groups = df['image_id']

# ✅ 수정 (올바른 방법)
groups = df['Sampling_Date']  # 날짜별 그룹핑!

# 또는 더 보수적으로
df['date_group'] = pd.to_datetime(df['Sampling_Date']).dt.strftime('%Y-%m-%d')
groups = df['date_group']
```

```python
def create_proper_folds(df, n_splits=5):
    """Sampling_Date 기반 올바른 CV split"""
    df = df.copy()

    # Sampling_Date를 그룹으로 사용
    df['date_group'] = pd.to_datetime(df['Sampling_Date']).dt.strftime('%Y-%m-%d')

    # State + Month로 stratify (선택적)
    df['strat_key'] = df['State'] + '_' + df['Month'].astype(str)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(
        df,
        df['strat_key'],
        groups=df['date_group']  # ⚠️ 핵심: date로 그룹핑!
    )):
        df.loc[val_idx, 'fold'] = fold

    return df
```

### 🔥 Priority 2: 더 큰 해상도 + TTA (예상 +0.02~0.03)

```python
# 해상도 변경
img_size = (518, 518)  # DINOv2/v3 최적
# 또는
img_size = (560, 560)  # 더 큰 해상도

# TTA 구현
def tta_predict(model, left, right, device, n_tta=8):
    """Test-Time Augmentation"""
    preds = []

    augmentations = [
        (False, False, 0),   # Original
        (True, False, 0),    # H-flip
        (False, True, 0),    # V-flip
        (True, True, 0),     # HV-flip
        (False, False, 1),   # 90°
        (False, False, 2),   # 180°
        (False, False, 3),   # 270°
        (True, False, 1),    # H-flip + 90°
    ]

    for hflip, vflip, rot in augmentations[:n_tta]:
        l, r = left.clone(), right.clone()

        if hflip:
            l = torch.flip(l, [3])
            r = torch.flip(r, [3])
        if vflip:
            l = torch.flip(l, [2])
            r = torch.flip(r, [2])
        if rot > 0:
            l = torch.rot90(l, rot, [2, 3])
            r = torch.rot90(r, rot, [2, 3])

        with torch.no_grad():
            pred = model(l.to(device), r.to(device))
            preds.append(pred.cpu())

    return torch.stack(preds).mean(0)
```

### 🔥 Priority 3: Weighted Loss (예상 +0.01~0.02)

```python
class WeightedR2Loss(nn.Module):
    """대회 평가 지표에 맞춘 Loss"""
    def __init__(self):
        super().__init__()
        # [Green, Dead, Clover, GDM, Total]
        self.weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])

    def forward(self, pred, target):
        # Component loss (Green, Clover, Dead 예측)
        component_pred = pred[:, [0, 2, 1]]  # Green, Clover, Dead
        component_target = target[:, :3]

        # 5개 타겟 구성
        green, clover, dead = pred[:, 0:1], pred[:, 2:3], pred[:, 1:2]
        gdm_pred = green + clover
        total_pred = gdm_pred + dead

        full_pred = torch.cat([green, dead, clover, gdm_pred, total_pred], dim=1)

        # 가중 MSE (Dry_Total_g에 50% 가중치!)
        weights = self.weights.to(pred.device)
        mse = (full_pred - target) ** 2
        weighted_mse = (mse * weights).mean()

        return weighted_mse

# 또는 더 단순하게: Dry_Total_g에 추가 Loss
def total_focused_loss(pred, target, alpha=0.5):
    """Dry_Total_g 중심 Loss"""
    component_loss = F.mse_loss(pred[:, :3], target[:, :3])
    total_loss = F.mse_loss(pred[:, 4], target[:, 4])  # Total
    return component_loss + alpha * total_loss
```

### 🔥 Priority 4: Multi-Resolution Ensemble (예상 +0.01~0.02)

```python
# 다양한 해상도로 학습된 모델 앙상블
resolutions = [448, 518, 560]
models = []

for res in resolutions:
    cfg.img_size = (res, res)
    model = train_model(cfg)
    models.append(model)

# Inference 시 평균
final_pred = np.mean([m.predict(test) for m in models], axis=0)
```

### 🔥 Priority 5: Seed Ensemble (예상 +0.005~0.01)

```python
# 다양한 seed로 학습
seeds = [42, 123, 456, 789, 1024]

all_preds = []
for seed in seeds:
    seed_everything(seed)
    model = train_model(cfg)
    all_preds.append(model.predict(test))

final_pred = np.mean(all_preds, axis=0)
```

### 🔥 Priority 6: 최적 앙상블 가중치 (예상 +0.01)

```python
from scipy.optimize import minimize

def optimize_ensemble_weights(oof_preds_list, oof_targets):
    """OOF 기반 최적 앙상블 가중치 찾기"""
    n_models = len(oof_preds_list)

    def objective(weights):
        # 가중 평균 예측
        weights = np.abs(weights)  # 양수 보장
        weights = weights / weights.sum()  # 합이 1

        ensemble_pred = sum(w * p for w, p in zip(weights, oof_preds_list))

        # Negative R² (최소화 목적)
        return -competition_metric(oof_targets, ensemble_pred)

    # 초기값: 균등 가중치
    x0 = np.ones(n_models) / n_models

    result = minimize(objective, x0, method='Nelder-Mead')

    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()

    return optimal_weights

# 사용
optimal_weights = optimize_ensemble_weights(
    [oof_v20, oof_v22, oof_v25, oof_v26],
    oof_targets
)
print(f"Optimal weights: {optimal_weights}")
```

---

## 📅 9일 실행 계획

### Day 1-2: CV 수정 + 재학습 (가장 중요!)
```
1. Sampling_Date 기반 GroupKFold로 변경
2. 기존 v20 아키텍처로 재학습
3. 새로운 CV 점수 확인 (Local CV가 LB와 더 일치해야 함)
```

### Day 3-4: 해상도 + TTA 실험
```
1. img_size = (518, 518) 또는 (560, 560)로 변경
2. TTA 구현 및 적용
3. 제출 및 LB 확인
```

### Day 5-6: Loss 함수 + 다양성
```
1. Weighted Loss 적용
2. 다른 seed로 추가 모델 학습
3. Multi-resolution 실험
```

### Day 7-8: 앙상블 최적화
```
1. OOF 기반 최적 가중치 찾기
2. 다양한 앙상블 조합 실험
3. Blending 또는 Stacking 시도
```

### Day 9: 최종 제출
```
1. 최적 조합 선택
2. 안전한 백업 제출
3. Final submission
```

---

## 🔧 즉시 적용 가능한 코드 수정

### 1. CV 수정 (v20/v26 기반)

```python
def create_proper_folds(df, n_splits=5):
    """⚠️ 핵심 수정: Sampling_Date 기반 CV"""
    df = df.copy()

    # 날짜 그룹 생성
    df['date_group'] = pd.to_datetime(df['Sampling_Date']).dt.strftime('%Y-%m-%d')

    # Stratification key (State + Month)
    df['strat_key'] = df['State'] + '_' + df['Month'].astype(str)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(
        df,
        df['strat_key'],
        groups=df['date_group']  # ⚠️ date로 그룹핑!
    )):
        df.loc[val_idx, 'fold'] = fold

    # 검증: 같은 날짜가 여러 fold에 있으면 안 됨
    date_fold_counts = df.groupby('date_group')['fold'].nunique()
    if (date_fold_counts > 1).any():
        print("⚠️ WARNING: Some dates are in multiple folds!")
    else:
        print("✓ CV split verified: dates are properly grouped")

    return df
```

### 2. 해상도 변경

```python
class CFG:
    img_size = (518, 518)  # 512 → 518 (DINOv2 최적)
```

### 3. TTA 추가 (Inference)

```python
@torch.no_grad()
def predict_with_tta(model, left, right, device, n_tta=4):
    """간단한 TTA: flip 4가지"""
    preds = []

    for hflip in [False, True]:
        for vflip in [False, True]:
            l = torch.flip(left, [3]) if hflip else left
            r = torch.flip(right, [3]) if hflip else right
            l = torch.flip(l, [2]) if vflip else l
            r = torch.flip(r, [2]) if vflip else r

            pred = model(l.to(device), r.to(device))
            preds.append(pred.cpu())

    return torch.stack(preds).mean(0)
```

---

## 📊 예상 개선 효과

| 전략 | 예상 향상 | 난이도 | 우선순위 |
|------|----------|--------|----------|
| CV 수정 (Sampling_Date) | +0.03~0.05 | 쉬움 | **1 (필수!)** |
| 해상도 518 | +0.01 | 쉬움 | 2 |
| TTA (4-fold) | +0.01~0.02 | 쉬움 | 3 |
| Weighted Loss | +0.01~0.02 | 중간 | 4 |
| Multi-seed | +0.005~0.01 | 쉬움 | 5 |
| 앙상블 최적화 | +0.01 | 중간 | 6 |

**총 예상 향상: +0.07~0.12 → 0.77~0.82 가능!**

---

## ⚠️ 주의사항

1. **CV-LB Correlation 확인**
   - CV 수정 후 Local CV와 LB의 상관관계 확인
   - 상관관계가 높아야 신뢰할 수 있음

2. **Overfitting 주의**
   - 357개 이미지로 작은 데이터셋
   - 너무 복잡한 모델/앙상블은 오히려 해로울 수 있음

3. **Private LB 대비**
   - Public 53% / Private 47% 분할
   - 과도한 LB probing 피하기

---

*Created: 2026-01-19*
*Target: 0.70 → 0.79+*
*Most Important: Fix CV strategy with Sampling_Date grouping!*
