# 데이터 증강 기법 문서

## 현재 상황

### 데이터셋 구성
- **Train**: 357장 이미지
- **Public Test**: 1장 이미지
- **타겟**: 5개 연속 값 (Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g)

### 현재 모델 성능
| 모델 | CV Score | Public LB |
|------|----------|-----------|
| Trial 12 (v17) | 0.787 ± 0.034 | 0.69 |
| Trial 22 | 0.784 ± 0.030 | 0.68 |

---

## 현재 사용 중인 데이터 증강

### 위치
[notebooks/17_optuna_optimized_train.py](file:///Users/kbsoo/Codes/kaggle/Image2Biomass/notebooks/17_optuna_optimized_train.py#L186-195)

### 코드
```python
def get_train_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),                    # (512, 512)로 리사이즈
        T.RandomHorizontalFlip(p=0.5),             # 50% 확률로 좌우 반전
        T.RandomVerticalFlip(p=0.5),               # 50% 확률로 상하 반전
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 색상 변환
        T.ToTensor(),                              # Tensor 변환
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # ImageNet 정규화
    ])
```

### Optuna 실험에서 테스트된 증강 전략들

[notebooks/16_hyperparameter_tuning.py](file:///Users/kbsoo/Codes/kaggle/Image2Biomass/notebooks/16_hyperparameter_tuning.py#L143-205)에서 5가지 전략 정의:

#### 1. minimal
```python
T.Resize(img_size)
T.RandomHorizontalFlip(p=0.5)
T.RandomVerticalFlip(p=0.5)
T.ToTensor()
T.Normalize(...)
```

#### 2. moderate
```python
T.Resize(img_size)
T.RandomHorizontalFlip(p=0.5)
T.RandomVerticalFlip(p=0.5)
T.RandomRotation(degrees=10)
T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
T.ToTensor()
T.Normalize(...)
```

#### 3. aggressive
```python
T.Resize(img_size)
T.RandomResizedCrop(img_size, scale=(0.8, 1.0))
T.RandomHorizontalFlip(p=0.5)
T.RandomVerticalFlip(p=0.5)
T.RandomRotation(degrees=15)
T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
T.ToTensor()
T.Normalize(...)
T.RandomErasing(p=0.2)
```

#### 4. color_focus (현재 사용 중)
```python
T.Resize(img_size)
T.RandomHorizontalFlip(p=0.5)
T.RandomVerticalFlip(p=0.5)
T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
T.ToTensor()
T.Normalize(...)
```

#### 5. geometric_focus
```python
T.Resize(img_size)
T.RandomResizedCrop(img_size, scale=(0.7, 1.0))
T.RandomHorizontalFlip(p=0.5)
T.RandomVerticalFlip(p=0.5)
T.RandomRotation(degrees=20)
T.RandomPerspective(distortion_scale=0.2, p=0.3)
T.ToTensor()
T.Normalize(...)
```

### Optuna 실험 결과
30개 trial 중 `color_focus`가 선택된 best trial:
- Trial 12: CV 0.803, aug_strategy="color_focus"

---

## 추가 가능한 데이터 증강 기법

### 1. MixUp

**개념**: 두 이미지와 라벨을 선형 보간하여 새로운 샘플 생성

```python
λ = random.beta(alpha, alpha)
mixed_img = λ * img1 + (1-λ) * img2
mixed_label = λ * label1 + (1-λ) * label2
```

**참고**: Classification용으로 설계됨. Regression에서는 라벨 값이 크게 다른 샘플 간 혼합 시 문제될 수 있음.

---

### 2. C-Mixup (Close-label MixUp)

**출처**: NeurIPS 2021, "C-Mixup: Improving Generalization in Regression"

**개념**: 라벨이 유사한 샘플끼리만 혼합

```python
# 라벨 거리 기반 확률 계산
distances = [abs(label_i - label_j) for j in range(len(dataset))]
probs = [exp(-d / sigma) for d in distances]

# 확률적으로 혼합 대상 선택
j = sample_with_probability(probs)

# MixUp 수행
mixed_img = λ * img_i + (1-λ) * img_j
mixed_label = λ * label_i + (1-λ) * label_j
```

---

### 3. CutMix

**개념**: 한 이미지의 일부 영역을 다른 이미지로 대체

```python
# 랜덤 박스 영역 선택
box = random_box(img_size, λ)

# 이미지 합성
mixed_img = img1.copy()
mixed_img[box] = img2[box]

# 라벨은 면적 비율로 혼합
mixed_label = λ * label1 + (1-λ) * label2
```

---

### 4. Mosaic

**개념**: 4장의 이미지를 하나로 합성

```
┌─────────┬─────────┐
│  img1   │  img2   │
├─────────┼─────────┤
│  img3   │  img4   │
└─────────┴─────────┘
```

```python
# 4장 선택
idxs = random.sample(range(len(images)), 4)

# 분할 비율 랜덤
cx, cy = random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)

# 라벨: 면적 비율로 가중 평균
areas = [cx*cy, (1-cx)*cy, cx*(1-cy), (1-cx)*(1-cy)]
mosaic_label = sum(areas[i] * labels[idxs[i]] for i in range(4))
```

---

### 5. Stable Diffusion 기반 합성 데이터

**개념**: Generative AI로 새로운 학습 이미지 생성

**Option A: Image-to-Image**
```
원본 이미지 → Stable Diffusion → 스타일 변형된 이미지
(원본 target 값 유지)
```

**Option B: Text-to-Image**
```
Prompt: "Grassland with 60% green grass, 20% clover, 20% dead material"
→ 새 이미지 생성
→ Target 값 별도 할당 필요
```

---

### 6. Copy-Paste

**개념**: 한 이미지의 특정 객체를 다른 이미지에 붙이기

```python
# Segmentation mask 필요
clover_patch = src_img * clover_mask
augmented = dst_img.copy()
augmented[position] = blend(dst_img[position], clover_patch)
```

---

### 7. Albumentations 라이브러리 활용

**현재 사용 X**. torchvision.transforms만 사용 중.

Albumentations에서 추가 가능한 변환:
- `ElasticTransform`
- `GridDistortion`
- `OpticalDistortion`
- `CoarseDropout`
- `CLAHE` (대비 제한 적응 히스토그램 평활화)

---

### 8. Test-Time Augmentation (TTA)

**현재 상태**:
```python
# 3x TTA
transforms = [Original, HFlip, VFlip]
```

**확장 가능한 TTA**:
```python
# 7x TTA
transforms = [
    Original,
    HFlip, VFlip, HFlip+VFlip,
    Rotate90, Rotate180, Rotate270
]
```

**현재 TTA 위치**: [notebooks/17_optuna_optimized_infer.py](file:///Users/kbsoo/Codes/kaggle/Image2Biomass/notebooks/17_optuna_optimized_infer.py#L97-116)

---

## 관련 연구

### Regression용 MixUp 논문
- **C-Mixup**: [https://arxiv.org/abs/2110.01077](https://arxiv.org/abs/2110.01077)
- **RegMix**: [https://arxiv.org/abs/2106.03374](https://arxiv.org/abs/2106.03374)
- **RC-Mixup**: [https://arxiv.org/abs/2312.xxxxx](https://arxiv.org/abs/2312.xxxxx)

### Diffusion 기반 데이터 증강
- Stable Diffusion을 활용한 synthetic data 생성이 2024년 연구에서 활발히 연구됨
- 이미지당 비용: ~$0.003 (2024년 12월 기준)
