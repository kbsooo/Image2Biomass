# 🎯 Breakthrough 전략: 훈련 vs 추론 분류

## 📚 훈련(Training) 단계에서 할 것

### 1. ⚠️ CV 전략 수정 (가장 중요!)
**예상 향상: +0.03~0.05**

```python
# ❌ 현재
groups = df['image_id']

# ✅ 수정
groups = df['Sampling_Date']  # 날짜별 그룹핑
```

- 모델을 **처음부터 다시 학습**해야 함
- CV가 바뀌면 모든 fold의 train/val 분할이 달라짐

---

### 2. 해상도 변경
**예상 향상: +0.01**

```python
# ❌ 현재
img_size = (512, 512)

# ✅ 수정
img_size = (518, 518)  # 또는 (560, 560)
```

- 다른 해상도로 **재학습 필요**
- 추론 시에도 같은 해상도 사용해야 함

---

### 3. Weighted Loss 적용
**예상 향상: +0.01~0.02**

```python
# ❌ 현재
main_loss = F.mse_loss(pred, main_targets)  # 단순 MSE

# ✅ 수정: Dry_Total_g에 50% 가중치 반영
weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])  # Green, Dead, Clover, GDM, Total
weighted_loss = (mse * weights).mean()
```

- Loss 함수 변경 → **재학습 필요**

---

### 4. Multi-Seed 학습
**예상 향상: +0.005~0.01**

```python
seeds = [42, 123, 456, 789, 1024]
for seed in seeds:
    seed_everything(seed)
    train_model(cfg)  # 각 seed로 별도 학습
```

- 같은 설정, 다른 seed로 **여러 모델 학습**
- 추론 시 앙상블로 사용

---

### 5. Multi-Resolution 학습
**예상 향상: +0.01~0.02**

```python
resolutions = [448, 518, 560]
for res in resolutions:
    cfg.img_size = (res, res)
    train_model(cfg)  # 각 해상도로 별도 학습
```

- 다양한 해상도로 **여러 모델 학습**
- 추론 시 앙상블로 사용

---

### 6. Data Augmentation 강화 (선택적)
**예상 향상: +0.005~0.01**

```python
# 더 강한 augmentation
T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05)
T.RandomRotation(degrees=15)
T.RandomAffine(degrees=0, translate=(0.1, 0.1))
```

- Augmentation 변경 → **재학습 필요**

---

## 🔮 추론(Inference) 단계에서 할 것

### 1. TTA (Test-Time Augmentation)
**예상 향상: +0.01~0.02**

```python
@torch.no_grad()
def predict_with_tta(model, left, right, device, n_tta=4):
    """기존 모델 그대로 사용, 추론만 변경"""
    preds = []

    for hflip in [False, True]:
        for vflip in [False, True]:
            l = torch.flip(left, [3]) if hflip else left
            r = torch.flip(right, [3]) if hflip else right
            l = torch.flip(l, [2]) if vflip else l
            r = torch.flip(r, [2]) if vflip else r

            pred = model(l.to(device), r.to(device))
            preds.append(pred.cpu())

    return torch.stack(preds).mean(0)  # 평균
```

- **기존 학습된 모델 그대로 사용**
- 추론 코드만 수정하면 됨
- 추론 시간 4~8배 증가

---

### 2. 앙상블 가중치 최적화
**예상 향상: +0.01**

```python
# v27_infer.py의 WEIGHTS 최적화
# ❌ 현재: 단순 평균 또는 수동 가중치
ENSEMBLE_METHOD = "simple"

# ✅ 수정: OOF 기반 최적 가중치
from scipy.optimize import minimize

def find_optimal_weights(oof_preds_list, oof_targets):
    def objective(weights):
        weights = np.abs(weights) / np.abs(weights).sum()
        ensemble = sum(w * p for w, p in zip(weights, oof_preds_list))
        return -competition_metric(oof_targets, ensemble)

    result = minimize(objective, np.ones(len(oof_preds_list)))
    return np.abs(result.x) / np.abs(result.x).sum()

optimal_weights = find_optimal_weights([oof_v20, oof_v22, oof_v25, oof_v26], oof_targets)
```

- **기존 모델들 그대로 사용**
- OOF 예측값으로 최적 가중치 계산
- 추론 시 가중 평균 적용

---

### 3. Post-Processing
**예상 향상: +0.005~0.01**

```python
# 예측값 후처리
def post_process(predictions):
    # 1. Negative 값 클리핑
    predictions = np.maximum(predictions, 0)

    # 2. 타겟 간 일관성 보정
    # Dry_Total ≈ Dry_Clover + Dry_Dead + Dry_Green
    green, dead, clover = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    gdm, total = predictions[:, 3], predictions[:, 4]

    # 합계 일관성 체크 및 보정
    sum_components = green + dead + clover
    predictions[:, 4] = (total + sum_components) / 2  # 평균으로 보정

    return predictions
```

- **기존 모델 그대로 사용**
- 예측 결과만 후처리

---

### 4. Rank Average Ensemble
**예상 향상: +0.005**

```python
# v27_infer.py에서 이미 구현됨
ENSEMBLE_METHOD = "rank"  # simple → rank로 변경
```

- **기존 모델 그대로 사용**
- 앙상블 방법만 변경

---

## 📊 요약 테이블

| 전략 | 단계 | 예상 향상 | 재학습 필요 | 난이도 |
|------|------|----------|------------|--------|
| **CV 수정 (Sampling_Date)** | 🏋️ 훈련 | +0.03~0.05 | ✅ 필수 | 쉬움 |
| 해상도 518x518 | 🏋️ 훈련 | +0.01 | ✅ 필수 | 쉬움 |
| Weighted Loss | 🏋️ 훈련 | +0.01~0.02 | ✅ 필수 | 중간 |
| Multi-Seed | 🏋️ 훈련 | +0.005~0.01 | ✅ 필수 | 쉬움 |
| Multi-Resolution | 🏋️ 훈련 | +0.01~0.02 | ✅ 필수 | 쉬움 |
| **TTA** | 🔮 추론 | +0.01~0.02 | ❌ 불필요 | 쉬움 |
| **앙상블 가중치 최적화** | 🔮 추론 | +0.01 | ❌ 불필요 | 중간 |
| **Post-Processing** | 🔮 추론 | +0.005~0.01 | ❌ 불필요 | 쉬움 |
| Rank Average | 🔮 추론 | +0.005 | ❌ 불필요 | 쉬움 |

---

## ⚡ 즉시 적용 가능한 것 (추론만 수정)

**오늘 바로 시도 가능 (기존 v20/v22/v25/v26 모델 사용):**

1. **TTA 추가** → v27_infer.py 수정
2. **Rank Average 사용** → `ENSEMBLE_METHOD = "rank"`
3. **앙상블 가중치 최적화** → OOF 기반 가중치 계산
4. **Post-Processing** → 예측값 후처리

```python
# v27_infer.py 수정 예시

# 1. TTA 함수 추가
@torch.no_grad()
def predict_with_tta(model, loader, device):
    model.eval()
    all_outputs, all_ids = [], []

    for left, right, ids in tqdm(loader):
        # 4-way TTA (original + 3 flips)
        preds = []
        for hf in [False, True]:
            for vf in [False, True]:
                l = torch.flip(left, [3]) if hf else left
                r = torch.flip(right, [3]) if hf else right
                l = torch.flip(l, [2]) if vf else l
                r = torch.flip(r, [2]) if vf else r

                out = model(l.to(device), r.to(device))
                preds.append(out.cpu())

        avg_pred = torch.stack(preds).mean(0)
        all_outputs.append(avg_pred.numpy())
        all_ids.extend(ids)

    return np.concatenate(all_outputs), all_ids

# 2. 앙상블 방법 변경
ENSEMBLE_METHOD = "rank"  # 또는 최적화된 가중치 사용

# 3. Post-processing 추가
final_preds = np.maximum(final_preds, 0)  # 음수 제거
```

---

## 🎯 권장 실행 순서

### Phase 1: 즉시 (추론만 수정) - 오늘
```
1. v27_infer.py에 TTA 추가
2. ENSEMBLE_METHOD = "rank" 변경
3. Post-processing 추가
4. 제출 → LB 확인
```
**예상: 0.70 → 0.71~0.72**

### Phase 2: 단기 (재학습 필요) - 1~3일
```
1. CV 수정 (Sampling_Date 그룹핑) ← 가장 중요!
2. 해상도 518x518로 변경
3. 재학습 및 제출
```
**예상: 0.71 → 0.75~0.77**

### Phase 3: 중기 (추가 최적화) - 4~7일
```
1. Weighted Loss 적용
2. Multi-seed 학습
3. 앙상블 가중치 최적화
```
**예상: 0.77 → 0.79+**

---

*결론: CV 수정(훈련)이 가장 큰 향상을 가져오지만, TTA(추론)는 오늘 바로 적용 가능!*
