# CSIRO Biomass Competition - Diagnostic Analysis

**Date**: 2026-01-16
**Current Best**: LB 0.69 (v20)
**Target**: LB 0.76+ (Gold Medal)

---

## 1. Current Status Summary

### Model Performance History

| Version | CV Score | LB Score | CV-LB Gap | Key Changes |
|---------|----------|----------|-----------|-------------|
| v15 | 0.67 | 0.61 | 0.06 | Baseline, simple head |
| v17 | 0.79 | 0.69 | 0.10 | Optuna optimized (512 hidden, 3 layers) |
| v20 | 0.79 | 0.69 | 0.10 | Similar to v17 |
| v22 | ~0.65 | 0.64 | ~0.01 | Frozen backbone, simple head |

### Key Observation
- **Complex models (v17, v20)**: High CV, moderate LB
- **Simple model (v22)**: Lower CV, similar LB
- **Conclusion**: Model complexity is NOT the root cause of LB underperformance

---

## 2. Data Distribution Analysis

### 2.1 Training Data Statistics

```
Total samples: 357
Year: 2015 only

State Distribution:
  NSW: 75 (21.0%)
  Tas: 138 (38.7%)
  Vic: 112 (31.4%)
  WA: 32 (9.0%)

Season Distribution (Southern Hemisphere):
  Spring (Sep-Nov): 133 (37.3%)
  Winter (Jun-Aug): 131 (36.7%)
  Autumn (Mar-May): 52 (14.6%)
  Summer (Dec-Feb): 41 (11.5%)
```

### 2.2 Critical Finding: State-Season Confounding

```
State x Season Cross-tabulation:

        Autumn  Spring  Summer  Winter
NSW       23      11      41       0    ← No Winter!
Tas       29      71       0      38    ← No Summer!
Vic        0      39       0      73    ← No Autumn, No Summer!
WA         0      12       0      20    ← No Autumn, No Summer!
```

**Implications**:
1. NSW Summer data has NO equivalent in other states
2. Training without NSW = Training without Summer
3. Model cannot disentangle State effects from Season effects

### 2.3 Target Distribution by Group

```
State-wise Dry_Total_g:
        mean    std     min     max
NSW    70.90   35.58   7.60   185.70   ← Highest (Summer effect?)
Tas    36.80   20.70   4.30   107.99
Vic    42.67   20.89  10.19   108.25
WA     31.39   19.62   1.04    72.17   ← Lowest (Clover-only)

Season-wise Dry_Total_g:
         mean    std     min     max
Summer  70.97   30.31  29.00   134.20  ← Highest
Spring  50.84   30.76   2.48   185.70
Autumn  44.37   25.86   7.50   157.90
Winter  32.06   14.92   1.04    73.33  ← Lowest
```

**Key Insight**: State and Season effects are heavily correlated. The model might learn "NSW = high biomass" instead of "Summer = high biomass".

---

## 3. Hypothesis: Why CV-LB Gap Exists

### Hypothesis 1: Shortcut Learning (Location Memorization)

**Theory**: The model learns location-specific patterns instead of actual biomass features.

```
What model should learn:
  "Dense green vegetation" → High Green biomass

What model actually learns:
  "Tasmania texture pattern" → ~37g (average for Tas)
  "NSW color palette" → ~71g (average for NSW)
```

**Evidence**:
- CV is high because same locations appear in train/val (via StratifiedGroupKFold)
- LB drops because test has unseen locations

**Validation Method**: Grad-CAM analysis to check what model focuses on

### Hypothesis 2: Train-Test Distribution Shift

**Theory**: Test set contains fundamentally different data.

```
Train: 357 images, 2015, 4 known states
Test: 800+ images, unknown year?, unknown locations?

Questions:
- Why is test set 2x larger than train?
- Does test include new Australian states?
- Does test include different years (2016+)?
- Does test have different camera/lighting conditions?
```

**Validation Method**: Analyze test image statistics (brightness, color distribution, texture)

### Hypothesis 3: Physics Constraint Exploitation

**Theory**: Model satisfies constraints without learning individual components.

```
Constraint: GDM = Green + Clover, Total = GDM + Dead

Possible exploitation:
- Predict Green = X, Clover = 0, Dead = 0
- All constraints satisfied
- But model never learned to distinguish components
```

**Evidence**: Need to check per-target R² scores

---

## 4. Test Image Analysis Plan

### 4.1 Extractable Features (No Labels Needed)

```python
For each test image:
1. Color Statistics:
   - Mean RGB values
   - Green channel dominance: G / (R + G + B)
   - Color histogram

2. Vegetation Indices:
   - ExG (Excess Green) = 2*G - R - B
   - VARI = (G - R) / (G + R - B + eps)
   - Green-Red ratio = G / R

3. Texture Features:
   - Edge density (Canny edge detection)
   - Local Binary Patterns (LBP)
   - Entropy (randomness measure)

4. Brightness/Contrast:
   - Mean luminance
   - Standard deviation (contrast)
```

### 4.2 Analysis Questions

1. **Distribution Comparison**:
   - Is test image color distribution different from train?
   - Are there clusters in test that don't exist in train?

2. **Correlation Analysis**:
   - In train, how well do vegetation indices correlate with actual biomass?
   - Can we use this correlation as pseudo-labels for test?

3. **Outlier Detection**:
   - Which test images are most different from train?
   - These might be the hardest to predict

---

## 5. Proposed Solutions

### Solution 1: Consistency Regularization (Recommended - Easy)

**Concept**: Same image with different augmentations should give same prediction.

```python
def consistency_loss(model, image):
    aug1 = augment(image)
    aug2 = augment(image)  # Different augmentation

    pred1 = model(aug1)
    pred2 = model(aug2)

    return MSE(pred1, pred2)

total_loss = supervised_loss + lambda * consistency_loss
```

**Why it helps**:
- Forces model to learn augmentation-invariant features
- Prevents learning "color A = location X = biomass Y" shortcuts
- Focuses on actual vegetation characteristics

### Solution 2: RGB Vegetation Index Pseudo-Labels (Medium)

**Concept**: Use vegetation indices as weak supervision signal.

```python
# Step 1: Compute vegetation index for all images
def compute_veg_index(image):
    R, G, B = image.mean(dim=[2,3]).unbind(1)
    exg = 2*G - R - B
    return exg

# Step 2: Learn index-to-biomass mapping on train
# Step 3: Use mapping to create pseudo-labels for test
# Step 4: Fine-tune with pseudo-labels (confident ones only)
```

**Why it helps**:
- Vegetation indices are physics-based (location-independent)
- Provides additional training signal
- Bridges train-test domain gap

### Solution 3: Test-Time Adaptation - TENT (Advanced)

**Concept**: Adapt model to test distribution at inference time.

```python
# During inference
model.eval()
for batch in test_loader:
    # Enable grad for BatchNorm params only
    pred = model(batch)
    entropy = -(pred * torch.log(pred + eps)).sum()
    entropy.backward()
    # Update only BN statistics
    optimizer.step()
```

**Why it helps**:
- Model automatically adjusts to test distribution
- No labels needed
- Reduces domain shift effect

### Solution 4: Domain Adversarial Training (Advanced)

**Concept**: Train model to extract features that are domain-invariant.

```python
class DomainClassifier(nn.Module):
    # Predicts if sample is from train or test

# Training:
# 1. Feature extractor tries to fool domain classifier
# 2. Domain classifier tries to distinguish train/test
# 3. Equilibrium: features are domain-invariant
```

---

## 6. Experiment Priority

### Phase 1: Diagnosis (1-2 hours)
- [ ] Analyze test image statistics
- [ ] Compare train vs test color distributions
- [ ] Run Grad-CAM on current best model

### Phase 2: Quick Wins (2-4 hours)
- [ ] Add consistency regularization to v22
- [ ] Compute vegetation indices for train/test
- [ ] Analyze per-target R² scores

### Phase 3: Advanced (4-8 hours)
- [ ] Implement pseudo-labeling pipeline
- [ ] Try Test-Time Adaptation
- [ ] Domain adversarial training

---

## 7. Success Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| CV (StratifiedGroupKFold) | 0.79 | N/A | Might be misleading |
| CV (Leave-One-State-Out) | ~0.50? | 0.65+ | More realistic |
| LB Score | 0.69 | 0.76+ | Gold medal threshold |
| CV-LB Gap | 0.10 | <0.05 | Indicates generalization |

---

## 8. Key Insights Summary

1. **Model complexity is not the problem** - Both simple and complex models fail similarly on LB

2. **Data confounding is severe** - State and Season are heavily correlated, making it impossible to learn pure biomass features

3. **Test set is likely different** - 800+ images vs 357 train suggests different/broader coverage

4. **Standard CV overestimates performance** - StratifiedGroupKFold allows location leakage

5. **The solution must be domain-invariant** - Features that work only on specific locations will fail

---

## Appendix: Code Snippets for Diagnosis

### A1: Test Image Statistics Analysis

```python
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

def analyze_image(img_path):
    img = np.array(Image.open(img_path).convert('RGB')) / 255.0
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    return {
        'mean_R': R.mean(),
        'mean_G': G.mean(),
        'mean_B': B.mean(),
        'green_ratio': G.mean() / (R.mean() + G.mean() + B.mean() + 1e-8),
        'ExG': (2*G - R - B).mean(),
        'brightness': img.mean(),
        'contrast': img.std(),
    }

# Analyze all test images
test_df = pd.read_csv('data/test.csv')
test_stats = [analyze_image(f'data/{p}') for p in test_df['image_path'].unique()]
test_stats_df = pd.DataFrame(test_stats)

# Compare with train
train_df = pd.read_csv('data/train.csv')
train_stats = [analyze_image(f'data/{p}') for p in train_df['image_path'].unique()]
train_stats_df = pd.DataFrame(train_stats)

print("Train stats:\n", train_stats_df.describe())
print("\nTest stats:\n", test_stats_df.describe())
```

### A2: Grad-CAM Analysis

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load best model
model = load_model('model_fold0.pth')
target_layer = model.backbone.blocks[-1]  # Last transformer block

cam = GradCAM(model=model, target_layers=[target_layer])

# Generate CAM for sample images
for img_path in sample_images:
    img = load_image(img_path)
    grayscale_cam = cam(input_tensor=img)
    visualization = show_cam_on_image(img, grayscale_cam)
    save_visualization(visualization, f'cam_{img_path.stem}.png')
```
