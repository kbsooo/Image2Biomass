# Hybrid Approach: Privileged Knowledge Distillation with DINOv2

## Executive Summary

본 문서는 CSIRO Biomass Prediction 대회를 위한 **Hybrid Approach** 설계를 상세히 기술합니다.

**핵심 문제**: Train 데이터에만 존재하는 Privileged Information (NDVI, Height, State, Species)을 Test time에 활용할 수 없는 상황에서 최적의 예측 성능을 달성하는 것.

**제안 솔루션**: DINOv2 backbone + Privileged Knowledge Distillation + Zero-Inflated Regression

**목표**: R² 0.9 이상

---

## 1. Problem Formulation

### 1.1 데이터 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAIN DATA                                │
├─────────────────────────────────────────────────────────────────┤
│  Image (RGB)      │  Privileged Information    │  Targets       │
│  ─────────────    │  ──────────────────────    │  ───────       │
│  [H, W, 3]        │  • NDVI (float)            │  • Dry_Green   │
│                   │  • Height_cm (float)       │  • Dry_Dead    │
│                   │  • State (categorical)     │  • Dry_Clover  │
│                   │  • Species (categorical)   │  • GDM (계산)   │
│                   │  • Date (temporal)         │  • Total (계산) │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        TEST DATA                                 │
├─────────────────────────────────────────────────────────────────┤
│  Image (RGB)      │  Privileged Information    │  Targets       │
│  ─────────────    │  ──────────────────────    │  ───────       │
│  [H, W, 3]        │  ❌ NOT AVAILABLE          │  ❓ PREDICT    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 이론적 프레임워크: LUPI (Learning Using Privileged Information)

Vapnik & Vashist (2009)가 제안한 LUPI 패러다임:

```
Traditional Learning:  (x, y) → f(x) = ŷ
LUPI:                  (x, x*, y) → f(x) = ŷ
                            ↑
                       Privileged Info
                       (train only)
```

**Key Insight**: Privileged information x*는 직접 사용할 수 없지만,
Teacher model을 통해 "dark knowledge"로 변환하여 Student에게 전달 가능.

---

## 2. Architecture Overview

### 2.1 High-Level Pipeline

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                     HYBRID APPROACH PIPELINE                               ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │                    PHASE 1: TEACHER TRAINING                        │  ║
║  │                                                                      │  ║
║  │   Image ──→ DINOv2 ──→ [CLS]                                        │  ║
║  │                          │                                          │  ║
║  │   NDVI ───┐              │                                          │  ║
║  │   Height ─┼──→ TabNet ───┼──→ Fusion ──→ Physics Head ──→ Biomass  │  ║
║  │   State ──┤              │      │                                   │  ║
║  │   Species ┘              │      │                                   │  ║
║  │                          │      │                                   │  ║
║  │                          └──────┴──→ Soft Targets (저장)            │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║                                    │                                       ║
║                                    ▼                                       ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │                    PHASE 2: STUDENT TRAINING                        │  ║
║  │                                                                      │  ║
║  │   Image ──→ DINOv2 ──→ [CLS] ──→ Student Head ──→ Biomass          │  ║
║  │                                        │                            │  ║
║  │                                        ▼                            │  ║
║  │                              Loss = α·L_hard + (1-α)·L_soft         │  ║
║  │                                     │              │                │  ║
║  │                              MSE(pred, y)    MSE(pred, teacher_pred)│  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║                                    │                                       ║
║                                    ▼                                       ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │                    PHASE 3: INFERENCE                               │  ║
║  │                                                                      │  ║
║  │   Test Image ──→ DINOv2 ──→ Student Head ──→ Physics ──→ Submission│  ║
║  │                              (Image Only)                           │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 2.2 Component Breakdown

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE COMPONENTS                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. BACKBONE: DINOv2 ViT-B/14                                          │
│     ├── Input: [B, 3, 518, 518]                                        │
│     ├── Output: [CLS] token [B, 768]                                   │
│     └── Mode: Frozen or LoRA fine-tune                                 │
│                                                                         │
│  2. TABULAR ENCODER (Teacher only)                                     │
│     ├── Input: [NDVI, Height, State_emb, Species_emb]                  │
│     ├── Architecture: MLP with residual connections                    │
│     └── Output: [B, 128]                                               │
│                                                                         │
│  3. FUSION MODULE (Teacher only)                                       │
│     ├── Method: FiLM (Feature-wise Linear Modulation)                  │
│     ├── γ, β = TabularEncoder(tabular)                                 │
│     └── fused = img_feat * (1 + γ) + β                                 │
│                                                                         │
│  4. PREDICTION HEADS                                                   │
│     ├── Main Head: 3 outputs (Green, Clover, Dead)                     │
│     ├── Clover Head: 2-stage (classifier + regressor)                  │
│     └── Auxiliary Heads (Teacher): NDVI, Height, State, Species        │
│                                                                         │
│  5. PHYSICS CONSTRAINT LAYER                                           │
│     ├── GDM = Softplus(Green) + Softplus(Clover)                       │
│     └── Total = GDM + Softplus(Dead)                                   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 DINOv2 Backbone

#### 3.1.1 왜 DINOv2인가?

| 특성 | EfficientNet-B4 | DINOv2 ViT-B/14 |
|------|-----------------|-----------------|
| Pre-training | ImageNet Supervised | 142M images Self-supervised |
| Feature Quality | Classification-biased | General-purpose |
| Transfer to Agriculture | Good | **Excellent** |
| Fine-tuning 필요 | Yes | Optional (frozen도 가능) |
| Parameters | 19M | 86M |

**연구 근거**:
- [ICCV 2023 Plant Phenotyping Workshop](https://phenomuk.org)에서 DINOv2가 MAE, DINO 대비 식물 관련 태스크에서 우수한 성능 입증
- [2025 Rapeseed AGB 연구](https://www.mdpi.com/2077-0472/15/23/2516)에서 DINOv2 + MTL로 바이오매스 예측 성공

#### 3.1.2 구현 세부사항

```python
import torch
import torch.nn as nn

class DINOv2Backbone(nn.Module):
    """
    DINOv2 ViT-B/14 backbone for feature extraction.

    Output: [CLS] token (768-dim) capturing global image semantics.

    Why [CLS] token?
    - Aggregates information from all patches via self-attention
    - Proven effective for regression tasks (unlike patch tokens)
    - Reduces computational cost vs using all patch tokens
    """
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        freeze: bool = True,
        weights_path: str = None  # Local weights for Kaggle (no internet)
    ):
        super().__init__()

        # Kaggle 환경: 로컬 weights 사용 (인터넷 없음)
        if weights_path and Path(weights_path).exists():
            # timm을 통해 ViT 구조 생성 후 weights 로드
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=False  # 구조만 로드
            )
            state_dict = torch.load(weights_path, weights_only=True)
            self.backbone.load_state_dict(state_dict)
            print(f"✓ DINOv2 loaded from local: {weights_path}")
        else:
            # 로컬 개발: torch hub에서 다운로드
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=True
            )
            print("✓ DINOv2 loaded from torch hub")

        self.feat_dim = 768  # ViT-B/14

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ DINOv2 backbone frozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] - RGB images (H, W should be divisible by 14)
        Returns:
            features: [B, 768] - [CLS] token
        """
        features = self.backbone(x)  # [B, 768]
        return features
```

#### 3.1.3 Input Resolution 고려사항

```
DINOv2 ViT-B/14:
- Patch size: 14×14
- Recommended input: 518×518 (37×37 patches)
- Alternative: 392×392 (28×28 patches) - faster, slightly lower quality
- Minimum: 224×224 (16×16 patches)

For this competition:
- Original images: Variable (likely ~1000×500)
- Recommended: 518×518 (crop or pad to square)
- Alternative: 448×448 (32×32 patches) - good balance
```

### 3.2 Tabular Encoder (Teacher Model Only)

#### 3.2.1 Feature Engineering

```python
class TabularFeatureEncoder(nn.Module):
    """
    Encodes heterogeneous tabular features into a dense representation.

    Features:
    - NDVI: Continuous [0, 1] - vegetation greenness index
    - Height: Continuous [0, 70] cm - canopy height
    - State: Categorical (4 classes) - NSW, Vic, Tas, WA
    - Species: Categorical (12+ classes) - plant species
    - Month: Cyclical [1-12] - seasonal pattern
    """
    def __init__(
        self,
        n_continuous: int = 2,  # NDVI, Height
        n_states: int = 4,
        n_species: int = 15,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        output_dim: int = 128
    ):
        super().__init__()

        # Continuous features: Batch normalization + linear
        self.cont_bn = nn.BatchNorm1d(n_continuous)
        self.cont_linear = nn.Linear(n_continuous, hidden_dim // 2)

        # Categorical embeddings
        self.state_embed = nn.Embedding(n_states, embed_dim)
        self.species_embed = nn.Embedding(n_species, embed_dim)

        # Month: Cyclical encoding (sin/cos)
        self.month_linear = nn.Linear(2, embed_dim)  # sin, cos

        # Fusion MLP
        total_cat_dim = embed_dim * 3  # state + species + month
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + total_cat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(
        self,
        continuous: torch.Tensor,  # [B, 2] - NDVI, Height
        state: torch.Tensor,       # [B] - long
        species: torch.Tensor,     # [B] - long
        month: torch.Tensor        # [B] - int [1-12]
    ) -> torch.Tensor:
        """Returns: [B, output_dim]"""

        # Continuous
        cont = self.cont_bn(continuous)
        cont = self.cont_linear(cont)  # [B, hidden_dim//2]

        # Categorical embeddings
        state_emb = self.state_embed(state)      # [B, embed_dim]
        species_emb = self.species_embed(species) # [B, embed_dim]

        # Month: Cyclical encoding
        month_rad = (month.float() - 1) / 12 * 2 * 3.14159
        month_sin = torch.sin(month_rad).unsqueeze(-1)
        month_cos = torch.cos(month_rad).unsqueeze(-1)
        month_enc = self.month_linear(torch.cat([month_sin, month_cos], dim=-1))

        # Concatenate all
        all_features = torch.cat([
            cont, state_emb, species_emb, month_enc
        ], dim=-1)

        return self.fusion(all_features)  # [B, output_dim]
```

### 3.3 FiLM Fusion Module

#### 3.3.1 이론적 배경

[FiLM (Feature-wise Linear Modulation)](https://arxiv.org/abs/1709.07871)은 conditioning 정보를 neural network에 주입하는 효과적인 방법입니다.

```
Standard approach:  concat([img_feat, tab_feat]) → MLP
FiLM approach:      img_feat * (1 + γ) + β,  where (γ, β) = f(tab_feat)

Why FiLM > Concatenation?
1. Multiplicative interaction이 더 풍부한 표현력 제공
2. Tabular features가 image features를 "modulate" (조절)
3. CLEVR benchmark에서 concat 대비 큰 성능 향상 입증
```

#### 3.3.2 구현

```python
class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation for multi-modal fusion.

    Given image features F and tabular features T:
    - γ, β = Linear(T)
    - Output = F * (1 + γ) + β

    Intuition:
    - γ (scale): Tabular info가 어떤 image features를 강조/억제할지 결정
    - β (shift): Tabular info가 image features에 추가 정보 주입
    """
    def __init__(self, img_dim: int, tab_dim: int):
        super().__init__()

        # Generate γ (scale) and β (shift) from tabular features
        self.gamma_net = nn.Sequential(
            nn.Linear(tab_dim, img_dim),
            nn.Tanh()  # γ ∈ [-1, 1] to prevent exploding values
        )
        self.beta_net = nn.Linear(tab_dim, img_dim)

    def forward(
        self,
        img_feat: torch.Tensor,  # [B, img_dim]
        tab_feat: torch.Tensor   # [B, tab_dim]
    ) -> torch.Tensor:
        """Returns: [B, img_dim] - modulated features"""

        gamma = self.gamma_net(tab_feat)  # [B, img_dim]
        beta = self.beta_net(tab_feat)    # [B, img_dim]

        # FiLM modulation
        # (1 + gamma) ensures multiplicative factor is centered around 1
        return img_feat * (1 + gamma) + beta
```

### 3.4 Zero-Inflated Regression Head for Clover

#### 3.4.1 문제 분석

```
Dry_Clover_g 분포:
├── 37.8% = 0 (정확히 0, 클로버 없음)
├── 50.7% = (0, 20] g
└── 11.5% = > 20g (outliers)

Standard MSE Loss의 문제:
- 0값을 예측하도록 bias됨 (평균이 0 쪽으로 치우침)
- Non-zero 샘플에서 under-prediction
```

#### 3.4.2 Two-Stage Architecture

```python
class ZeroInflatedHead(nn.Module):
    """
    Two-stage prediction for zero-inflated targets (Clover).

    Stage 1: Binary classifier - P(y > 0)
    Stage 2: Positive regressor - E[y | y > 0]

    Final prediction: P(y > 0) * E[y | y > 0]

    Why this works:
    - Classifier learns "does this image contain clover?"
    - Regressor learns "if clover exists, how much?"
    - Decouples the two fundamentally different questions
    """
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()

        # Stage 1: Zero classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Stage 2: Positive amount regressor
        self.regressor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive output
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns:
            p_positive: [B, 1] - probability of non-zero
            amount: [B, 1] - predicted amount if positive
            prediction: [B, 1] - final prediction = p * amount
        """
        p_positive = self.classifier(x)
        amount = self.regressor(x)
        prediction = p_positive * amount

        return p_positive, amount, prediction

    def compute_loss(
        self,
        p_positive: torch.Tensor,
        amount: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Combined loss for zero-inflated regression.

        Args:
            alpha: Weight for classification loss (vs regression loss)
        """
        # Binary targets
        is_positive = (targets > 0).float()

        # Classification loss (BCE)
        cls_loss = F.binary_cross_entropy(p_positive, is_positive)

        # Regression loss (only on positive samples)
        positive_mask = targets > 0
        if positive_mask.sum() > 0:
            reg_loss = F.mse_loss(
                amount[positive_mask],
                targets[positive_mask]
            )
        else:
            reg_loss = torch.tensor(0.0, device=targets.device)

        return alpha * cls_loss + (1 - alpha) * reg_loss
```

### 3.5 Physics-Constrained Output Layer

#### 3.5.1 설계 원리

```
물리적 제약 조건:
1. 모든 바이오매스 값은 비음수: Green, Dead, Clover ≥ 0
2. GDM = Green + Clover (정의)
3. Total = GDM + Dead = Green + Clover + Dead (정의)

구현 전략:
- 독립 변수 3개만 예측 (Green, Clover, Dead)
- Softplus 활성화로 비음수 보장
- GDM, Total은 forward pass에서 계산 (gradient 흐름 유지)
```

#### 3.5.2 구현

```python
class PhysicsConstrainedHead(nn.Module):
    """
    Prediction head with hard physics constraints.

    Key Design Decisions:
    1. Predict 3 independent variables (Green, Clover, Dead)
    2. Compute derived variables (GDM, Total) in forward pass
    3. Use Softplus for non-negativity (smoother gradient than ReLU)

    Competition Target Order: [Green, Dead, Clover, GDM, Total]
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_zero_inflated_clover: bool = True
    ):
        super().__init__()

        self.use_zero_inflated_clover = use_zero_inflated_clover

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Green head (standard regression)
        self.green_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

        # Dead head (standard regression)
        self.dead_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

        # Clover head (zero-inflated or standard)
        if use_zero_inflated_clover:
            self.clover_head = ZeroInflatedHead(hidden_dim // 2)
        else:
            self.clover_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()
            )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, in_features] - input features

        Returns:
            dict with keys:
            - 'independent': [B, 3] - Green, Clover, Dead
            - 'full': [B, 5] - Green, Dead, Clover, GDM, Total (competition order)
            - 'clover_prob': [B, 1] - P(Clover > 0) if zero-inflated
        """
        shared_feat = self.shared(x)

        green = self.green_head(shared_feat)   # [B, 1]
        dead = self.dead_head(shared_feat)     # [B, 1]

        if self.use_zero_inflated_clover:
            clover_prob, clover_amount, clover = self.clover_head(shared_feat)
        else:
            clover = self.clover_head(shared_feat)
            clover_prob = None

        # Physics constraints (hard)
        gdm = green + clover
        total = gdm + dead

        # Competition order: [Green, Dead, Clover, GDM, Total]
        full = torch.cat([green, dead, clover, gdm, total], dim=1)
        independent = torch.cat([green, clover, dead], dim=1)

        result = {
            'independent': independent,
            'full': full,
            'green': green,
            'dead': dead,
            'clover': clover
        }

        if clover_prob is not None:
            result['clover_prob'] = clover_prob
            result['clover_amount'] = clover_amount

        return result
```

---

## 4. Training Pipeline

### 4.1 Phase 1: Teacher Model Training

```python
class TeacherModel(nn.Module):
    """
    Full multi-modal teacher model.
    Uses ALL available information: Image + NDVI + Height + State + Species + Date

    Purpose:
    1. Learn the best possible predictions using all information
    2. Generate "soft targets" for student distillation
    3. (Optional) Predict auxiliary tasks as regularization
    """
    def __init__(self, cfg):
        super().__init__()

        # Image encoder
        self.backbone = DINOv2Backbone(freeze=cfg.freeze_backbone)

        # Tabular encoder
        self.tabular_encoder = TabularFeatureEncoder(
            n_states=cfg.n_states,
            n_species=cfg.n_species,
            output_dim=128
        )

        # Fusion
        self.fusion = FiLMFusion(
            img_dim=self.backbone.feat_dim,
            tab_dim=128
        )

        # Main prediction head
        self.main_head = PhysicsConstrainedHead(
            in_features=self.backbone.feat_dim,
            use_zero_inflated_clover=True
        )

        # Auxiliary heads (optional, for regularization)
        if cfg.use_auxiliary_tasks:
            self.aux_ndvi_head = nn.Linear(self.backbone.feat_dim, 1)
            self.aux_height_head = nn.Linear(self.backbone.feat_dim, 1)
            self.aux_state_head = nn.Linear(self.backbone.feat_dim, cfg.n_states)
            self.aux_species_head = nn.Linear(self.backbone.feat_dim, cfg.n_species)

    def forward(self, image, continuous, state, species, month):
        # Image features
        img_feat = self.backbone(image)  # [B, 768]

        # Tabular features
        tab_feat = self.tabular_encoder(continuous, state, species, month)

        # FiLM fusion
        fused_feat = self.fusion(img_feat, tab_feat)

        # Main predictions
        main_output = self.main_head(fused_feat)

        return {
            'main': main_output,
            'features': fused_feat  # For distillation
        }
```

### 4.2 Phase 2: Student Model Training with Knowledge Distillation

```python
class StudentModel(nn.Module):
    """
    Image-only student model for inference.
    Learns from both ground truth AND teacher's soft targets.
    """
    def __init__(self, cfg):
        super().__init__()

        self.backbone = DINOv2Backbone(freeze=cfg.freeze_backbone)

        self.main_head = PhysicsConstrainedHead(
            in_features=self.backbone.feat_dim,
            use_zero_inflated_clover=True
        )

    def forward(self, image):
        img_feat = self.backbone(image)
        output = self.main_head(img_feat)
        output['features'] = img_feat
        return output


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    L_total = α * L_hard + (1 - α) * L_soft + λ * L_clover

    Where:
    - L_hard: MSE with ground truth labels
    - L_soft: MSE with teacher's predictions (soft targets)
    - L_clover: Zero-inflated loss for clover
    """
    def __init__(
        self,
        alpha: float = 0.5,      # Hard vs soft balance
        temperature: float = 1.0, # Softening temperature (for classification KD)
        clover_weight: float = 0.3
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.clover_weight = clover_weight

    def forward(
        self,
        student_output: dict,
        teacher_output: dict,
        targets: torch.Tensor,  # [B, 3] - Green, Clover, Dead
        target_weights: torch.Tensor = None
    ):
        # Hard loss (ground truth)
        pred = student_output['independent']  # [B, 3]
        hard_loss = F.mse_loss(pred, targets)

        # Soft loss (teacher's knowledge)
        teacher_pred = teacher_output['independent'].detach()
        soft_loss = F.mse_loss(pred, teacher_pred)

        # Zero-inflated clover loss
        clover_loss = torch.tensor(0.0, device=pred.device)
        if 'clover_prob' in student_output:
            is_positive = (targets[:, 1:2] > 0).float()
            clover_loss = F.binary_cross_entropy(
                student_output['clover_prob'],
                is_positive
            )

        # Combine
        total_loss = (
            self.alpha * hard_loss +
            (1 - self.alpha) * soft_loss +
            self.clover_weight * clover_loss
        )

        return {
            'total': total_loss,
            'hard': hard_loss,
            'soft': soft_loss,
            'clover': clover_loss
        }
```

### 4.3 Training Loop

```python
def train_hybrid_pipeline(cfg):
    """
    Full training pipeline.

    Phase 1: Train teacher with full information
    Phase 2: Generate soft targets for all training samples
    Phase 3: Train student with KD loss
    """

    # ========== PHASE 1: Teacher Training ==========
    print("=" * 60)
    print("PHASE 1: Training Teacher Model")
    print("=" * 60)

    teacher = TeacherModel(cfg).to(cfg.device)
    teacher_optimizer = AdamW(
        teacher.parameters(),
        lr=cfg.teacher_lr,
        weight_decay=cfg.weight_decay
    )

    for epoch in range(cfg.teacher_epochs):
        teacher.train()
        for batch in train_loader:
            images, continuous, state, species, month, targets = batch

            output = teacher(images, continuous, state, species, month)

            # Main loss
            loss = F.mse_loss(output['main']['independent'], targets)

            # Auxiliary losses (if enabled)
            if cfg.use_auxiliary_tasks:
                loss += 0.1 * F.mse_loss(output['aux_ndvi'], continuous[:, 0:1])
                loss += 0.1 * F.mse_loss(output['aux_height'], continuous[:, 1:2])

            teacher_optimizer.zero_grad()
            loss.backward()
            teacher_optimizer.step()

    # ========== PHASE 2: Generate Soft Targets ==========
    print("=" * 60)
    print("PHASE 2: Generating Soft Targets")
    print("=" * 60)

    teacher.eval()
    soft_targets = {}

    with torch.no_grad():
        for batch in train_loader:
            images, continuous, state, species, month, targets = batch
            image_ids = batch['image_id']

            output = teacher(images, continuous, state, species, month)

            for i, img_id in enumerate(image_ids):
                soft_targets[img_id] = {
                    'pred': output['main']['full'][i].cpu(),
                    'features': output['features'][i].cpu()
                }

    # ========== PHASE 3: Student Training ==========
    print("=" * 60)
    print("PHASE 3: Training Student Model with KD")
    print("=" * 60)

    student = StudentModel(cfg).to(cfg.device)
    student_optimizer = AdamW(
        student.parameters(),
        lr=cfg.student_lr,
        weight_decay=cfg.weight_decay
    )

    kd_loss_fn = DistillationLoss(alpha=cfg.kd_alpha)

    for epoch in range(cfg.student_epochs):
        student.train()

        for batch in train_loader:
            images, targets = batch['image'], batch['targets']
            image_ids = batch['image_id']

            # Get teacher's soft targets
            teacher_preds = torch.stack([
                soft_targets[img_id]['pred'] for img_id in image_ids
            ]).to(cfg.device)

            # Student forward
            student_output = student(images)

            # KD loss
            loss_dict = kd_loss_fn(
                student_output,
                {'independent': teacher_preds[:, :3]},  # Green, Clover, Dead
                targets
            )

            student_optimizer.zero_grad()
            loss_dict['total'].backward()
            student_optimizer.step()

    return student
```

---

## 5. Cross-Validation Strategy

### 5.1 Fold Design

```
추천: Geographic Stratified K-Fold

왜 Geographic Stratification?
- State별 바이오매스 분포가 크게 다름 (NSW 56.6g vs WA 9.3g)
- 각 fold에 모든 State가 포함되어야 일반화 성능 향상
- Random split 시 특정 State에 overfitting 위험

┌─────────────────────────────────────────────────────────────┐
│  Fold 0  │  Train: Vic, Tas, WA (80%)  │  Val: NSW (20%)  │
│  Fold 1  │  Train: NSW, Tas, WA (80%)  │  Val: Vic (20%)  │
│  Fold 2  │  Train: NSW, Vic, WA (80%)  │  Val: Tas (20%)  │
│  Fold 3  │  Train: NSW, Vic, Tas (80%) │  Val: WA (20%)   │
│  Fold 4  │  Stratified random from all states            │
└─────────────────────────────────────────────────────────────┘

Alternative: Time-based Split (temporal generalization)
- Train: Jan-Aug 2015
- Val: Sep-Nov 2015
```

### 5.2 구현

```python
from sklearn.model_selection import StratifiedKFold

def create_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42):
    """
    Geographic stratified k-fold.
    """
    # Stratify by State
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df['State'])):
        df.loc[val_idx, 'fold'] = fold

    return df
```

---

## 6. Hyperparameters

### 6.1 Recommended Configuration

```python
class CFG:
    # ===== Paths =====
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    OUTPUT_DIR = Path("/kaggle/working")

    # Pretrained weights (Kaggle Dataset: kbsooo/pretrained-weights-biomass)
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass")
    # Local: ~/kaggle_weights/dinov2/dinov2_vitb14.pth

    # ===== Model =====
    backbone = "dinov2_vitb14"
    input_size = 518  # DINOv2 optimal
    freeze_backbone = True  # Start frozen, optionally unfreeze later

    # ===== Teacher =====
    teacher_epochs = 20
    teacher_lr = 2e-4
    use_auxiliary_tasks = True
    aux_weight = 0.2

    # ===== Student =====
    student_epochs = 25
    student_lr = 1e-4
    kd_alpha = 0.5  # Balance between hard and soft loss
    kd_temperature = 2.0

    # ===== Zero-Inflated =====
    use_zero_inflated_clover = True
    clover_cls_weight = 0.5

    # ===== Training =====
    n_folds = 5
    batch_size = 16  # Per GPU
    weight_decay = 1e-4

    # ===== Augmentation =====
    use_mixup = True
    mixup_alpha = 0.2

    # ===== Misc =====
    seed = 42
    num_workers = 4
    device = "cuda"
```

### 6.2 Hyperparameter Sensitivity Analysis

```
High Sensitivity (tune carefully):
├── kd_alpha: 0.3-0.7 범위에서 grid search
├── teacher_epochs: Too few → weak soft targets, Too many → overfitting
└── freeze_backbone: Frozen이 일반적으로 더 안정적

Medium Sensitivity:
├── learning_rate: 1e-4 ~ 2e-4 범위
├── batch_size: 16-32 (larger = more stable)
└── clover_cls_weight: 0.3-0.5

Low Sensitivity:
├── hidden_dim: 128-512 (큰 차이 없음)
├── dropout: 0.2-0.4
└── weight_decay: 1e-5 ~ 1e-3
```

---

## 7. Goal & Evaluation

### 7.1 목표

**R² 0.9 이상** 달성을 목표로 한다.

### 7.2 Competition Metric

```
Weighted R² = Σ(weight_i × R²_i)

Target Weights:
- Dry_Total_g:  0.5  ← 가장 중요!
- GDM_g:        0.2
- Dry_Green_g:  0.1
- Dry_Dead_g:   0.1
- Dry_Clover_g: 0.1
```

### 7.3 실험 계획

각 component의 기여도를 측정하기 위해 단계별 ablation 실험 수행:

1. **DINOv2 Baseline**: Image only → Biomass
2. **+ Teacher-Student KD**: Privileged info 활용
3. **+ Zero-Inflated Clover**: 클로버 예측 개선
4. **+ Auxiliary Tasks**: Multi-task regularization
5. **+ TTA & Ensemble**: 최종 성능 극대화

---

## 8. Implementation Checklist

### Phase 1: Setup
- [x] DINOv2 weights 준비 (Kaggle Dataset: kbsooo/pretrained-weights-biomass)
- [ ] Data pipeline 구현 (with augmentation)
- [ ] Geographic stratified K-Fold 구현

### Phase 2: Teacher Model
- [ ] TabularFeatureEncoder 구현
- [ ] FiLMFusion 구현
- [ ] Teacher model 학습
- [ ] Soft targets 생성 및 저장

### Phase 3: Student Model
- [ ] ZeroInflatedHead 구현
- [ ] DistillationLoss 구현
- [ ] Student model 학습
- [ ] 5-Fold ensemble

### Phase 4: Inference
- [ ] Test-Time Augmentation 구현
- [ ] Submission 생성
- [ ] Physics constraint 검증

---

## 9. Risk Mitigation

### 9.1 Potential Issues

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DINOv2 too large for Kaggle | Medium | High | Use ViT-S/14 or distilled version |
| Teacher overfits to train | High | Medium | Early stopping, aux tasks |
| KD doesn't help | Low | High | Tune alpha, try feature distillation |
| Clover still bad | Medium | Medium | Increase cls_weight, more augmentation |

### 9.2 Fallback Plan

```
If KD approach fails:
1. Use DINOv2 + direct multi-task (no teacher-student)
2. Ensemble with current pseudo-labeling approach
3. Focus on TTA and post-processing
```

---

## 10. References

1. [LUPI - Vapnik & Vashist (2009)](https://www.jmlr.org/papers/volume16/vapnik15b/vapnik15b.pdf)
2. [FiLM - Perez et al. (2018)](https://arxiv.org/abs/1709.07871)
3. [DINOv2 - Oquab et al. (2024)](https://arxiv.org/abs/2304.07193)
4. [Knowledge Distillation - Hinton et al. (2015)](https://arxiv.org/abs/1503.02531)
5. [Noisy Student - Xie et al. (2020)](https://arxiv.org/abs/1911.04252)
6. [Plant Phenotyping with ViT (ICCV 2023)](https://phenomuk.org)
7. [Multi-Task Learning Overview - Ruder (2017)](https://www.ruder.io/multi-task/)

---

*Document Version: 1.0*
*Last Updated: 2026-01-11*
*Author: Claude Code Assistant*
