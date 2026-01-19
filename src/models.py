"""
Model architectures for Hybrid Approach.

Components:
- DINOv2Backbone: ViT-B/14 feature extractor
- TabularFeatureEncoder: Encodes NDVI, Height, State, Species
- FiLMFusion: Feature-wise Linear Modulation
- ZeroInflatedHead: Two-stage prediction for Clover
- PhysicsConstrainedHead: Ensures physical consistency
- TeacherModel: Uses all information (image + tabular)
- StudentModel: Uses only images
"""
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG


class DINOv2Backbone(nn.Module):
    """
    DINOv2 ViT-B/14 backbone for feature extraction.
    
    Output: [CLS] token (768-dim) capturing global image semantics.
    
    Why DINOv2?
    - Self-supervised on 142M images
    - Excellent transfer to agriculture/plant tasks
    - [CLS] token proven effective for regression
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        freeze: bool = True,
        weights_path: Optional[Path] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        
        # Set feature dimension based on model
        dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
        }
        self.feat_dim = dims.get(model_name, 768)
        
        self._load_backbone(weights_path)
        
        if freeze:
            self._freeze_backbone()
    
    def _load_backbone(self, weights_path: Optional[Path] = None):
        """Load DINOv2 model, handling both online and offline cases."""
        try:
            if weights_path and Path(weights_path).exists():
                # Kaggle offline: load from local file
                weight_file = Path(weights_path) / f"{self.model_name}.pth"
                if weight_file.exists():
                    # Load model structure first
                    self.backbone = torch.hub.load(
                        'facebookresearch/dinov2',
                        self.model_name,
                        pretrained=False,
                        trust_repo=True,
                    )
                    state_dict = torch.load(weight_file, map_location='cpu', weights_only=True)
                    self.backbone.load_state_dict(state_dict)
                    print(f"✓ DINOv2 loaded from: {weight_file}")
                else:
                    raise FileNotFoundError(f"Weight file not found: {weight_file}")
            else:
                # Online: download from torch hub
                self.backbone = torch.hub.load(
                    'facebookresearch/dinov2',
                    self.model_name,
                    pretrained=True,
                    trust_repo=True,
                )
                print(f"✓ DINOv2 loaded from torch hub")
        except Exception as e:
            # Fallback: use timm
            print(f"Warning: DINOv2 hub load failed ({e}), trying timm...")
            import timm
            timm_name = f"vit_base_patch14_dinov2.lvd142m"
            self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
            print(f"✓ DINOv2 loaded via timm: {timm_name}")
    
    def _freeze_backbone(self):
        """Freeze all parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ DINOv2 backbone frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] - RGB images (H, W should be divisible by 14)
        Returns:
            features: [B, feat_dim] - [CLS] token
        """
        # DINOv2 returns [CLS] token directly when called
        features = self.backbone(x)
        return features


class TabularFeatureEncoder(nn.Module):
    """
    Encodes heterogeneous tabular features into a dense representation.
    
    Features:
    - NDVI: Continuous [0, 1] - vegetation greenness
    - Height: Continuous [0, 70] cm - canopy height
    - State: Categorical (4 classes) - NSW, Vic, Tas, WA
    - Species: Categorical (30+ classes) - plant species
    - Month: Cyclical [1-12] - seasonal pattern
    """
    
    def __init__(
        self,
        n_continuous: int = 2,
        n_states: int = 4,
        n_species: int = 30,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        output_dim: int = 128,
    ):
        super().__init__()
        
        # Continuous features
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
            nn.ReLU(),
        )
    
    def forward(
        self,
        continuous: torch.Tensor,   # [B, 2] - NDVI, Height
        state: torch.Tensor,        # [B] - long
        species: torch.Tensor,      # [B] - long
        month: torch.Tensor,        # [B] - int [1-12]
    ) -> torch.Tensor:
        """Returns: [B, output_dim]"""
        
        # Continuous
        cont = self.cont_bn(continuous)
        cont = self.cont_linear(cont)  # [B, hidden_dim//2]
        
        # Categorical embeddings
        state_emb = self.state_embed(state)       # [B, embed_dim]
        species_emb = self.species_embed(species)  # [B, embed_dim]
        
        # Month: Cyclical encoding
        month_rad = (month.float() - 1) / 12 * 2 * math.pi
        month_sin = torch.sin(month_rad).unsqueeze(-1)
        month_cos = torch.cos(month_rad).unsqueeze(-1)
        month_enc = self.month_linear(torch.cat([month_sin, month_cos], dim=-1))
        
        # Concatenate all
        all_features = torch.cat([
            cont, state_emb, species_emb, month_enc
        ], dim=-1)
        
        return self.fusion(all_features)


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation for multi-modal fusion.
    
    Given image features F and tabular features T:
    - γ, β = Linear(T)
    - Output = F * (1 + γ) + β
    
    Why FiLM > Concatenation?
    - Multiplicative interaction provides richer expressiveness
    - Tabular features "modulate" image features
    - Proven on CLEVR benchmark
    """
    
    def __init__(self, img_dim: int, tab_dim: int):
        super().__init__()
        
        # Generate γ (scale) and β (shift) from tabular features
        self.gamma_net = nn.Sequential(
            nn.Linear(tab_dim, img_dim),
            nn.Tanh(),  # γ ∈ [-1, 1] to prevent exploding
        )
        self.beta_net = nn.Linear(tab_dim, img_dim)
    
    def forward(
        self,
        img_feat: torch.Tensor,  # [B, img_dim]
        tab_feat: torch.Tensor,  # [B, tab_dim]
    ) -> torch.Tensor:
        """Returns: [B, img_dim] - modulated features"""
        
        gamma = self.gamma_net(tab_feat)  # [B, img_dim]
        beta = self.beta_net(tab_feat)    # [B, img_dim]
        
        # FiLM modulation: (1 + gamma) ensures centered around 1
        return img_feat * (1 + gamma) + beta


class ZeroInflatedHead(nn.Module):
    """
    Two-stage prediction for zero-inflated targets (Clover).
    
    Stage 1: Binary classifier - P(y > 0)
    Stage 2: Positive regressor - E[y | y > 0]
    Final prediction: P(y > 0) * E[y | y > 0]
    
    Why this works:
    - 37.8% of Clover values are exactly 0
    - Decouples "is clover present?" from "how much?"
    """
    
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        
        # Stage 1: Zero classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Stage 2: Positive amount regressor
        self.regressor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensures positive output
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class PhysicsConstrainedHead(nn.Module):
    """
    Prediction head with hard physics constraints.
    
    Key Design Decisions:
    1. Predict 3 independent variables (Green, Clover, Dead)
    2. Compute derived variables (GDM, Total) in forward pass
    3. Use Softplus for non-negativity
    
    Competition Target Order: [Green, Dead, Clover, GDM, Total]
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_zero_inflated_clover: bool = True,
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
            nn.Dropout(dropout),
        )
        
        # Green head
        self.green_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        
        # Dead head
        self.dead_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        
        # Clover head
        if use_zero_inflated_clover:
            self.clover_head = ZeroInflatedHead(hidden_dim // 2)
        else:
            self.clover_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, in_features]
            
        Returns:
            dict with:
            - 'independent': [B, 3] - Green, Clover, Dead
            - 'full': [B, 5] - Green, Dead, Clover, GDM, Total
            - 'clover_prob': [B, 1] - P(Clover > 0) if zero-inflated
        """
        shared_feat = self.shared(x)
        
        green = self.green_head(shared_feat)
        dead = self.dead_head(shared_feat)
        
        if self.use_zero_inflated_clover:
            clover_prob, clover_amount, clover = self.clover_head(shared_feat)
        else:
            clover = self.clover_head(shared_feat)
            clover_prob = None
            clover_amount = None
        
        # Physics constraints (hard)
        gdm = green + clover
        total = gdm + dead
        
        # Competition order: [Green, Dead, Clover, GDM, Total]
        full = torch.cat([green, dead, clover, gdm, total], dim=1)
        # Independent: [Green, Clover, Dead]
        independent = torch.cat([green, clover, dead], dim=1)
        
        result = {
            'independent': independent,
            'full': full,
            'green': green,
            'dead': dead,
            'clover': clover,
        }
        
        if clover_prob is not None:
            result['clover_prob'] = clover_prob
            result['clover_amount'] = clover_amount
        
        return result


class TeacherModel(nn.Module):
    """
    Full multi-modal teacher model.
    Uses ALL available information: Image + NDVI + Height + State + Species + Date
    
    Purpose:
    1. Learn best possible predictions using all information
    2. Generate "soft targets" for student distillation
    """
    
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        
        # Image encoder
        self.backbone = DINOv2Backbone(
            model_name=cfg.backbone,
            freeze=cfg.freeze_backbone,
            weights_path=cfg.get_weights_path(),
        )
        
        # Tabular encoder
        self.tabular_encoder = TabularFeatureEncoder(
            n_states=cfg.n_states,
            n_species=cfg.n_species,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.tabular_hidden_dim,
            output_dim=128,
        )
        
        # Fusion
        self.fusion = FiLMFusion(
            img_dim=self.backbone.feat_dim,
            tab_dim=128,
        )
        
        # Main prediction head
        self.main_head = PhysicsConstrainedHead(
            in_features=self.backbone.feat_dim,
            use_zero_inflated_clover=cfg.use_zero_inflated_clover,
        )
        
        # Auxiliary heads (for regularization)
        if cfg.use_auxiliary_tasks:
            self.aux_ndvi_head = nn.Linear(self.backbone.feat_dim, 1)
            self.aux_height_head = nn.Linear(self.backbone.feat_dim, 1)
            self.aux_state_head = nn.Linear(self.backbone.feat_dim, cfg.n_states)
            self.aux_species_head = nn.Linear(self.backbone.feat_dim, cfg.n_species)
    
    def forward(
        self,
        image: torch.Tensor,
        continuous: torch.Tensor,
        state: torch.Tensor,
        species: torch.Tensor,
        month: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: [B, 3, H, W]
            continuous: [B, 2] - NDVI, Height
            state: [B] - long
            species: [B] - long
            month: [B] - int
            
        Returns:
            dict with 'main', 'features', and optionally 'aux_*'
        """
        # Image features
        img_feat = self.backbone(image)  # [B, 768]
        
        # Tabular features
        tab_feat = self.tabular_encoder(continuous, state, species, month)
        
        # FiLM fusion
        fused_feat = self.fusion(img_feat, tab_feat)
        
        # Main predictions
        main_output = self.main_head(fused_feat)
        
        result = {
            'main': main_output,
            'features': fused_feat,
        }
        
        # Auxiliary predictions
        if self.cfg.use_auxiliary_tasks:
            result['aux_ndvi'] = self.aux_ndvi_head(fused_feat)
            result['aux_height'] = self.aux_height_head(fused_feat)
            result['aux_state'] = self.aux_state_head(fused_feat)
            result['aux_species'] = self.aux_species_head(fused_feat)
        
        return result


class StudentModel(nn.Module):
    """
    Image-only student model for inference.
    Learns from both ground truth AND teacher's soft targets.
    
    This is what we use at test time (no tabular features available).
    """
    
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        
        # Same backbone as teacher
        self.backbone = DINOv2Backbone(
            model_name=cfg.backbone,
            freeze=cfg.freeze_backbone,
            weights_path=cfg.get_weights_path(),
        )
        
        # Direct prediction head (no tabular fusion)
        self.main_head = PhysicsConstrainedHead(
            in_features=self.backbone.feat_dim,
            use_zero_inflated_clover=cfg.use_zero_inflated_clover,
        )
    
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: [B, 3, H, W]
            
        Returns:
            dict with 'independent', 'full', etc.
        """
        img_feat = self.backbone(image)
        output = self.main_head(img_feat)
        output['features'] = img_feat
        return output
