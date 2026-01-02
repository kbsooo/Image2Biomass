# Image2Biomass Project Instructions

## Competition Overview
- **Host**: CSIRO + MLA + Google Australia
- **Goal**: 목초지(Pasture) 이미지로부터 바이오매스(건조 중량) 예측
- **Metric**: Globally weighted R² (coefficient of determination)
- **Deadline**: 2026-01-21 11:59 UTC

## Target Variables
```
Dry_Green_g    # 녹색 식물 건조 중량
Dry_Dead_g     # 고사 식물 건조 중량
Dry_Clover_g   # 클로버(콩과) 건조 중량
GDM_g          # Green Dry Matter
Dry_Total_g    # 총 건조 중량
```

## Dataset Characteristics
- **Images**: 1,162 top-view images (70cm × 30cm quadrat)
- **Locations**: 19 sites across Australia
- **Seasons**: Multi-seasonal data
- **Features**: Tabular (vegetation height, NDVI 등) + Image

## Technical Stack
- PyTorch 2.x + torchvision
- timm (PyTorch Image Models)
- albumentations (augmentation)
- pandas, scikit-learn

## Code Convention
- `#%%` cell markers for Jupytext compatibility
- Shape assertions: `assert x.shape == (B, C, H, W)`
- Type hints for function signatures
- Visualization for every experiment

## Model Strategy Priorities
1. **Baseline**: ResNet/EfficientNet + Tabular feature concatenation
2. **Multi-task**: 5개 타겟 동시 예측 (shared backbone)
3. **Ensemble**: Image-only + Tabular-only + Fusion models

## Experiment Tracking
- Use wandb or simple CSV logging
- Log: loss, R² per target, learning rate, augmentation config
- Save best checkpoint by validation R²

## File Structure
```
Image2Biomass/
├── data/              # 데이터 (gitignore)
├── src/
│   ├── dataset.py     # Dataset, DataLoader
│   ├── model.py       # Model architectures
│   ├── train.py       # Training loop
│   ├── inference.py   # Submission 생성
│   └── utils.py       # Metrics, visualization
├── notebooks/         # EDA, experiments
├── configs/           # Hyperparameter configs
└── submissions/       # submission.csv files
```

## Key Considerations
- **Multi-output Regression**: R² 계산 시 각 타겟별 가중치 확인 필요
- **Data Leakage**: Location/Season 기반 split 고려 (spatial/temporal CV)
- **Class Imbalance**: 타겟 분포 확인, log transform 고려
- **Augmentation**: 농업 도메인 특성 반영 (회전, 색상 변환 주의)

## Submission Rules
- Notebook 제출 (인터넷 접근 불가)
- GPU runtime 제한 있음
- submission.csv: sample_id, predicted columns
