# 🧪 Quick Test Script for NextGen Pipeline
# 
# This script tests individual components before full training

import sys
import os
from pathlib import Path

# Check if running in correct environment
print(f"Python: {sys.version}")
print(f"Current directory: {Path.cwd()}")

# Check data availability
data_path = Path("/kaggle/input/csiro-biomass")
if data_path.exists():
    print("✓ Kaggle data available")
    print(f"Files: {list(data_path.glob('*'))}")
else:
    print("❌ Kaggle data not found")
    # Check for local data
    local_data = Path("./data")
    if local_data.exists():
        print(f"✓ Local data found: {list(local_data.glob('*'))}")
    else:
        print("❌ No data found")

# Test basic imports
try:
    import numpy as np
    import pandas as pd
    print("✓ NumPy/Pandas available")
except ImportError as e:
    print(f"❌ NumPy/Pandas error: {e}")

try:
    from PIL import Image
    print("✓ PIL available")
except ImportError as e:
    print(f"❌ PIL error: {e}")

# Test PyTorch availability
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch error: {e}")

# Test timm availability  
try:
    import timm
    print(f"✓ timm {timm.__version__}")
    
    # Test model creation
    if torch.cuda.is_available():
        try:
            model = timm.create_model('convnextv2_base.fcmae', pretrained=False, num_classes=0)
            print(f"✓ ConvNeXtV2 test: {sum(p.numel() for p in model.parameters())} params")
            del model
        except Exception as e:
            print(f"❌ ConvNeXtV2 error: {e}")
except ImportError as e:
    print(f"❌ timm error: {e}")

print("\n🔧 Environment check complete!")
print("To run full training:")
print("1. Ensure dependencies are installed: pip install torch torchvision timm pandas pillow")
print("2. Update data paths in config")
print("3. Run: python3 15_nextgen_train.py")