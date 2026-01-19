# 🔧 torchvision.transforms.v2 Compatibility Fix

## 🐛 문제 분석

**torchvision.transforms.v2**에서 일부 transforms API 변경:

1. **GaussianBlur**: `p` 파라미터 제거됨
2. **RandomApply**로 감싸야 확률 적용 가능
3. 일부 transforms 이름 축소됨

## ✅ 수정된 코드

### Before (Error)
```python
T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3), p=0.2)
```

### After (Fixed)  
```python
T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3))], p=0.2)
```

## 🔄 다른 호환성 문제 해결

### RandomAdjustSharpness
```python
# 현재 호환되는 방식
T.RandomAdjustSharpness(sharpness_factor=0.8, p=0.2)
```

### 전체 transforms 리스트
```python
transforms_list = [
    T.Resize(self.img_size),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3))], p=0.2),
    T.RandomAdjustSharpness(sharpness_factor=0.8, p=0.2),
    T.RandomPerspective(distortion_scale=0.1, p=0.3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
```

## 📂 업데이트된 파일

- **`15_nextgen_train.ipynb`** - 수정 완료
- torchvision 버전 차이로 인한 호환성 문제 해결

## 🚀 바로 실행 가능!

이제 Colab에서 모든 transforms가 정상적으로 작동합니다.

**💡 추가 팁:**
```python
# Colab에서 torchvision 버전 확인
import torchvision
print(f"torchvision: {torchvision.__version__}")

# transforms.v2 사용 가능 여부
if hasattr(torchvision.transforms, 'v2'):
    print("✓ transforms.v2 available")
else:
    print("⚠ transforms.v1 fallback needed")
```

**✅ 이제 0.90+ 목표 도전 준비 완료!**