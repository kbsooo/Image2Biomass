# CSIRO Biomass Competition - EDA Report

## 1. 대회 개요

**목표**: 목초지 이미지와 메타데이터를 기반으로 바이오매스(건조 중량) 예측

**예측 타겟 (5개)**:
- `Dry_Clover_g`: 클로버의 건조 중량
- `Dry_Dead_g`: 죽은 식물의 건조 중량
- `Dry_Green_g`: 녹색 식물의 건조 중량
- `Dry_Total_g`: 전체 건조 중량
- `GDM_g`: Green Dry Matter (녹색 건조 물질)

---

## 2. 데이터 구조

| 항목 | Train | Test |
|------|-------|------|
| 총 행 수 | 1,785 | 5 |
| 고유 이미지 수 | 357 | 1 |
| 이미지당 샘플 수 | 5 (타겟별) | 5 |

**Train 컬럼**: `sample_id`, `image_path`, `Sampling_Date`, `State`, `Species`, `Pre_GSHH_NDVI`, `Height_Ave_cm`, `target_name`, `target`

**Test 컬럼**: `sample_id`, `image_path`, `target_name` (메타데이터 없음!)

### ⚠️ 중요 발견
- **Test 데이터에는 State, Species, NDVI, Height 등 메타데이터가 없음**
- 이는 **이미지만으로 예측**해야 함을 의미
- 결측치 없음 (Train/Test 모두)

---

## 3. 타겟 변수 분석

| Target | Mean | Std | Min | Median | Max | Zero% |
|--------|------|-----|-----|--------|-----|-------|
| Dry_Clover_g | 6.65 | 12.12 | 0.0 | 1.42 | 71.79 | 37.8% |
| Dry_Dead_g | 12.04 | 12.40 | 0.0 | 7.98 | 83.84 | 11.2% |
| Dry_Green_g | 26.62 | 25.40 | 0.0 | 20.80 | 157.98 | 5.0% |
| Dry_Total_g | 45.32 | 27.98 | 1.04 | 40.30 | 185.70 | 0.0% |
| GDM_g | 33.27 | 24.94 | 1.04 | 27.11 | 157.98 | 0.0% |

### 주요 특징
1. **Dry_Clover_g**: 37.8%가 0값 (높은 sparsity) → 분류+회귀 접근 고려
2. **Dry_Dead_g**: 11.2%가 0값, 우편향(right-skewed) 분포
3. **Dry_Green_g**: 범위가 가장 넓음 (0~158g)
4. **Dry_Total_g**: 0값 없음, 가장 안정적인 분포
5. 모든 타겟이 우편향 분포 → **Log 변환 고려**

---

## 4. 타겟 간 상관관계

```
              Dry_Clover  Dry_Dead  Dry_Green  Dry_Total
Dry_Clover_g     1.000    -0.176     -0.277      0.104
Dry_Dead_g      -0.176     1.000      0.096      0.454
Dry_Green_g     -0.277     0.096      1.000      0.830
Dry_Total_g      0.104     0.454      0.830      1.000
```

### 핵심 인사이트
- **Dry_Green_g ↔ Dry_Total_g**: 강한 양의 상관관계 (r=0.830)
- **Dry_Clover_g ↔ Dry_Green_g**: 음의 상관관계 (r=-0.277) - 클로버가 많으면 다른 녹색 식물이 적음
- **Dry_Dead_g ↔ Dry_Total_g**: 중간 양의 상관관계 (r=0.454)
- Dry_Total_g ≈ Dry_Clover_g + Dry_Dead_g + Dry_Green_g 관계 성립

---

## 5. Feature 분석

### 5.1 범주형 변수

**State (주) 분포**:
| State | 이미지 수 | 비율 |
|-------|----------|------|
| Tas (Tasmania) | 138 | 38.7% |
| Vic (Victoria) | 112 | 31.4% |
| NSW (New South Wales) | 75 | 21.0% |
| WA (Western Australia) | 32 | 9.0% |

**Species (종) 분포** (Top 5):
1. Ryegrass_Clover: 98개 (27.5%)
2. Ryegrass: 62개 (17.4%)
3. Phalaris_Clover: 42개 (11.8%)
4. Clover: 41개 (11.5%)
5. Fescue: 28개 (7.8%)

### 5.2 수치형 변수

**Pre_GSHH_NDVI** (정규화 식생 지수):
- Mean: 0.657, Std: 0.152
- Range: 0.16 ~ 0.91
- NDVI가 높을수록 녹색 바이오매스가 많음

**Height_Ave_cm** (평균 높이):
- Mean: 7.60cm, Std: 10.29cm
- Range: 1.0 ~ 70.0cm
- 분포가 매우 편향됨 (대부분 낮은 높이)

### 5.3 Feature vs Target 상관관계

| Feature | Dry_Clover | Dry_Dead | Dry_Green | Dry_Total |
|---------|------------|----------|-----------|-----------|
| NDVI | 0.224 | -0.123 | **0.351** | **0.361** |
| Height | -0.160 | -0.050 | **0.648** | **0.497** |

- **Height_Ave_cm과 Dry_Green_g**: 강한 양의 상관관계 (r=0.648) - 키가 클수록 녹색 바이오매스 증가
- **NDVI와 Dry_Total_g**: 중간 양의 상관관계 (r=0.361)

---

## 6. 시계열 분석

**데이터 수집 기간**: 2015년 1월 ~ 11월 (단일 연도)

**월별 패턴**:
- 여름(1-2월): 높은 Dry_Green_g, 낮은 Dry_Dead_g
- 겨울(6-7월): 낮은 Dry_Green_g
- 봄(9-11월): Dry_Clover_g 증가, Dry_Dead_g 증가

### 계절적 특성
- 호주의 계절 패턴 반영 (남반구)
- 3월 데이터 없음

---

## 7. 이미지 분석

- **이미지 크기**: 약 2.5~4MB (고해상도)
- **이미지 형식**: JPG
- **특징**: 목초지를 위에서 촬영한 사진, 녹색/갈색/죽은 식물 혼재

### 이미지 특성 관찰
- 낮은 Total (1.04g): 대부분 빨간 마커와 맨땅
- 중간 Total (40g): 녹색과 갈색 혼재
- 높은 Total (185g): 빽빽한 녹색 식생

---

## 8. 모델링 전략 제안

### 8.1 데이터 전처리
1. **타겟 변환**: Log1p 변환으로 우편향 분포 정규화
2. **Dry_Clover_g 특별 처리**: 37.8% zero → 2-stage 모델 (분류 후 회귀)

### 8.2 모델 아키텍처
Test에 메타데이터가 없으므로 **이미지 기반 모델** 필수:
- CNN 기반: EfficientNet, ResNet, ConvNeXt
- ViT (Vision Transformer)
- 멀티태스크 학습: 5개 타겟 동시 예측

### 8.3 추가 전략
1. **Image Augmentation**: 회전, 플립, 색상 조정 등
2. **멀티태스크 러닝**: 타겟 간 상관관계 활용
3. **Ensemble**: 여러 모델 앙상블
4. **Transfer Learning**: ImageNet 사전학습 모델 활용

### 8.4 주의사항
- Train/Test 분포 차이 가능성 (Test는 1개 이미지만 공개)
- State/Species 기반 stratified split 권장
- 과적합 방지를 위한 Cross-validation 필수

---

## 9. 생성된 시각화 파일

1. `01_target_distributions.png` - 타겟 변수 히스토그램
2. `02_target_by_state.png` - 주(State)별 타겟 박스플롯
3. `03_correlation_heatmap.png` - 상관관계 히트맵
4. `04_scatter_features_vs_targets.png` - Feature-Target 산점도
5. `05_monthly_distribution.png` - 월별 타겟 분포
6. `06_species_distribution.png` - 종(Species) 분포
7. `07_state_distribution.png` - 주(State) 파이차트
8. `08_feature_distributions.png` - NDVI, Height 분포
9. `09_sample_images.png` - 샘플 이미지
10. `10_target_pairplots.png` - 타겟 간 산점도

---

*Generated: 2026-01-19*
