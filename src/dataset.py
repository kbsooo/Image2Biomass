"""
Dataset and data utilities for Hybrid Approach.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import CFG


def create_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Create Geographic Stratified K-Fold splits.
    
    Stratifies by State to ensure each fold has similar distribution
    of geographic regions (NSW, Vic, Tas, WA).
    
    Args:
        df: DataFrame with 'State' column
        n_folds: Number of folds
        seed: Random seed
        
    Returns:
        DataFrame with 'fold' column added
    """
    df = df.copy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df['State'])):
        df.loc[val_idx, 'fold'] = fold
    
    return df


def pivot_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format train.csv to wide format (one row per image).
    
    Input format:  sample_id, image_path, ..., target_name, target
    Output format: image_id, image_path, ..., Dry_Green_g, Dry_Dead_g, ...
    """
    # Extract image_id from sample_id
    df = df.copy()
    df['image_id'] = df['sample_id'].str.split('__').str[0]
    
    # Get non-target columns (same for all rows of an image)
    meta_cols = ['image_id', 'image_path', 'Sampling_Date', 'State', 
                 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    meta_df = df[meta_cols].drop_duplicates()
    
    # Pivot target columns
    target_df = df.pivot(index='image_id', columns='target_name', values='target')
    target_df = target_df.reset_index()
    
    # Merge
    result = meta_df.merge(target_df, on='image_id')
    
    return result


def get_train_transforms(cfg: CFG) -> A.Compose:
    """Training augmentations optimized for plant imagery."""
    return A.Compose([
        A.Resize(cfg.input_size, cfg.input_size),
        
        # Geometric - careful with plants (don't flip too aggressively)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.15, 
            rotate_limit=15, 
            p=0.5,
            border_mode=cv2.BORDER_REFLECT_101
        ),
        
        # Color augmentations - important for vegetation
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.7),
        
        # Simulate different lighting conditions
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        
        # Blur/noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
        ], p=0.2),
        
        # Normalize with ImageNet stats (good for DINOv2)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_valid_transforms(cfg: CFG) -> A.Compose:
    """Validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(cfg.input_size, cfg.input_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_tta_transforms(cfg: CFG) -> List[A.Compose]:
    """Test-Time Augmentation transforms."""
    base = [
        A.Resize(cfg.input_size, cfg.input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    transforms = [
        A.Compose(base),  # Original
        A.Compose([A.HorizontalFlip(p=1.0)] + base),  # HFlip
        A.Compose([A.VerticalFlip(p=1.0)] + base),    # VFlip
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base),  # Both
    ]
    
    return transforms[:cfg.tta_transforms]


class BiomassDataset(Dataset):
    """
    Dataset for CSIRO Biomass Prediction.
    
    Returns:
        - image: [3, H, W] tensor
        - continuous: [2] tensor (NDVI, Height)
        - state: int (0-3)
        - species: int (0-N)
        - month: int (1-12)
        - targets: [3] tensor (Green, Clover, Dead)
        - image_id: str
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        data_path: Path,
        transforms: Optional[A.Compose] = None,
        is_train: bool = True,
        state_mapping: Optional[Dict[str, int]] = None,
        species_mapping: Optional[Dict[str, int]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.data_path = Path(data_path)
        self.transforms = transforms
        self.is_train = is_train
        
        # Create/use mappings
        if state_mapping is None:
            unique_states = sorted(df['State'].unique())
            self.state_mapping = {s: i for i, s in enumerate(unique_states)}
        else:
            self.state_mapping = state_mapping
            
        if species_mapping is None:
            unique_species = sorted(df['Species'].unique())
            self.species_mapping = {s: i for i, s in enumerate(unique_species)}
        else:
            self.species_mapping = species_mapping
            
        # Parse dates for month extraction
        self.df['month'] = pd.to_datetime(self.df['Sampling_Date']).dt.month
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.data_path / row['image_path']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        # Tabular features
        continuous = torch.tensor([
            row['Pre_GSHH_NDVI'],
            row['Height_Ave_cm'] / 70.0  # Normalize to ~[0, 1]
        ], dtype=torch.float32)
        
        state = self.state_mapping.get(row['State'], 0)
        species = self.species_mapping.get(row['Species'], 0)
        month = int(row['month'])
        
        result = {
            'image': image,
            'continuous': continuous,
            'state': state,
            'species': species,
            'month': month,
            'image_id': row['image_id'],
        }
        
        # Add targets if training
        if self.is_train:
            targets = torch.tensor([
                row['Dry_Green_g'],
                row['Dry_Clover_g'],
                row['Dry_Dead_g'],
            ], dtype=torch.float32)
            result['targets'] = targets
            
            # Full targets for metrics
            full_targets = torch.tensor([
                row['Dry_Green_g'],
                row['Dry_Dead_g'],
                row['Dry_Clover_g'],
                row['GDM_g'],
                row['Dry_Total_g'],
            ], dtype=torch.float32)
            result['full_targets'] = full_targets
        
        return result


def create_dataloaders(
    df: pd.DataFrame,
    fold: int,
    cfg: CFG,
) -> Tuple[DataLoader, DataLoader, Dict, Dict]:
    """
    Create train and validation dataloaders for a specific fold.
    
    Returns:
        train_loader, val_loader, state_mapping, species_mapping
    """
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    
    data_path = cfg.get_data_path()
    
    # Create mappings from full dataset
    state_mapping = {s: i for i, s in enumerate(sorted(df['State'].unique()))}
    species_mapping = {s: i for i, s in enumerate(sorted(df['Species'].unique()))}
    
    train_dataset = BiomassDataset(
        train_df,
        data_path,
        transforms=get_train_transforms(cfg),
        is_train=True,
        state_mapping=state_mapping,
        species_mapping=species_mapping,
    )
    
    val_dataset = BiomassDataset(
        val_df,
        data_path,
        transforms=get_valid_transforms(cfg),
        is_train=True,
        state_mapping=state_mapping,
        species_mapping=species_mapping,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, state_mapping, species_mapping
