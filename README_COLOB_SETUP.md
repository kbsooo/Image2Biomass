# 🏆 Image2Biomass Next-Generation Setup

## 📋 노트북 실행 가이드

### 🗂️ 생성된 파일
- `15_nextgen_train.ipynb` - 트레이닝 노트북
- `15_nextgen_inference.ipynb` - 추론 노트북

### 🚀 Colab 실행 순서

1. **트레이닝 노트북** (`15_nextgen_train.ipynb`)
   - [ ] 런타임 → GPU로 변경
   - [ ] 셀 순서대로 실행
   - [ ] Kaggle 로그인 (팝업창)
   - [ ] Google Drive 마운트
   - [ ] 5-fold 트레이닝 (약 2-3시간)

2. **추론 노트북** (`15_nextgen_inference.ipynb`)  
   - [ ] 트레이닝 완료 후 실행
   - [ ] 모델 로드 확인
   - [ ] TTA 적용 추론
   - [ ] submission.csv 생성

### 💡 실행 팁

**Colab 환경:**
```python
# GPU 확인
!nvidia-smi

# 메모리 확인  
!free -h

# 스토리지 확인
!df -h
```

**필수 설치 (Colab):**
```python
!pip install torch torchvision timm transformers scikit-learn pandas pillow tqdm kagglehub
```

### 🎯 성능 목표
- **현재**: 0.61 public score
- **목표**: 0.90+ CV score
- **기대**: Multi-backbone + NeFF + TTA로 0.15+ 향상

### 📁 Google Drive 백업 구조
```
MyDrive/kaggle_models/image2biomass_nextgen/
├── nextgen_run_20250113_143022_cv0.7542/
│   ├── nextgen_fold0.pth
│   ├── nextgen_fold1.pth
│   ├── nextgen_fold2.pth
│   ├── nextgen_fold3.pth
│   ├── nextgen_fold4.pth
│   └── nextgen_results.json
└── submissions/
    └── nextgen_submission_20250113_163045.csv
```

### 🔧 트러블슈팅

**OOM 에러:**
```python
# batch_size 줄이기
cfg.batch_size = 8  # 12 → 8
```

**모델 로딩 실패:**
```python
# Google Drive에서 모델 복사
!cp /content/drive/MyDrive/kaggle_models/image2biomass_nextgen/models/* ./output/
```

**Kaggle 로그인 문제:**
```python
# 수동으로 API 키 설정
!mkdir -p ~/.kaggle
!echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
```

### 📊 예상 실행 시간
| Fold | 추론 시간 | GPU 메모리 |
|------|------------|------------|
| 1/5  | 20-25분    | 12-16GB    |
| 5/5  | 1.5-2시간  | -           |
| TTA  | +10-15분   | -           |

**💰 비용 팁**: Colab Pro 사용 시 약 $3-5 예상

---

✅ **준비 완료! Colab에서 바로 실행 가능합니다.**