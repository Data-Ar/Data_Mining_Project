#!/usr/bin/env python
"""Quick smoke test to verify the code works end-to-end."""

import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from dr_odir_classifier.data import DRNPZFolderDataset, get_transforms
from dr_odir_classifier.models import build_model
from torch.utils.data import DataLoader

def find_odir_data():
    """Try to find ODIR_Data in common locations."""
    possible_paths = [
        "./ODIR_Data",
        "../ODIR_Data",
        "../../data_mining_pro/ODIR_Data",
        "/medailab/medailab/sulaiman/data_mining_pro/ODIR_Data",
    ]
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "train")):
            return path
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=None, 
                       help="Path to ODIR_Data directory")
    args = parser.parse_args()
    
    print("Quick smoke test...")
    
    # Find data directory
    if args.root_dir:
        root_dir = args.root_dir
    else:
        root_dir = find_odir_data()
    
    ds = None
    if root_dir is None:
        print("⚠ ODIR_Data not found. Skipping data loading test.")
        print("  Use --root_dir to specify the path, or place ODIR_Data in the project root.")
        data_test = False
    else:
        print(f"Using data directory: {root_dir}")
        # Test data loading
        train_tf, _ = get_transforms(224)
        try:
            ds = DRNPZFolderDataset(os.path.join(root_dir, "train"), transform=train_tf)
            print(f"✓ Data loading: {len(ds)} samples")
            data_test = True
        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            data_test = False
    
    # Test model
    try:
        model = build_model("resnet18", pretrained=True)
        print("✓ Model building: resnet18")
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        return False
    
    # Test dataloader (only if data loading succeeded)
    if data_test:
        try:
            loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
            batch = next(iter(loader))
            print(f"✓ DataLoader: batch shape {batch[0].shape}")
        except Exception as e:
            print(f"✗ DataLoader failed: {e}")
            return False
    
    print("\n✓ All basic tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

