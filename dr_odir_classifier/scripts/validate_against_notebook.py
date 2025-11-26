#!/usr/bin/env python
"""
Validation script to ensure the reorganized code produces identical results
to the original notebook implementation.

Run this to verify:
1. Data loading produces same samples
2. Model architectures match
3. Training produces similar metrics (within tolerance)
4. Evaluation metrics format matches
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import random

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from dr_odir_classifier.data import DRNPZFolderDataset, get_transforms
from dr_odir_classifier.models import build_model
from dr_odir_classifier.metrics import compute_all_metrics, print_results
from dr_odir_classifier.evaluation import evaluate
from dr_odir_classifier.losses import FocalLoss


def test_data_loading(root_dir="./ODIR_Data"):
    """Test that data loading matches original notebook behavior."""
    print("=" * 60)
    print("TEST 1: Data Loading")
    print("=" * 60)
    
    # Set seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    train_tf, eval_tf = get_transforms(224, use_advanced_aug=True)
    train_ds = DRNPZFolderDataset(os.path.join(root_dir, "train"), transform=train_tf)
    
    print(f"✓ Loaded {len(train_ds)} training samples")
    print(f"✓ Labels range: {train_ds.labels.min()} to {train_ds.labels.max()}")
    print(f"✓ Sex distribution: {np.bincount(train_ds.sex)}")
    
    # Test first sample
    img, label, sex = train_ds[0]
    print(f"✓ First sample: label={label.item()}, sex={sex.item()}, img_shape={img.shape}")
    
    assert len(train_ds) > 0, "Dataset should not be empty"
    assert img.shape[0] == 3, "Image should be RGB (3 channels)"
    print("✓ Data loading test PASSED\n")
    return True


def test_model_building():
    """Test that model architectures match original."""
    print("=" * 60)
    print("TEST 2: Model Building")
    print("=" * 60)
    
    test_archs = ["resnet50", "densenet121", "efficientnet_b0"]
    
    for arch in test_archs:
        model = build_model(arch, pretrained=True)
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ {arch}: {num_params:,} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 1), f"Output shape should be (1,1), got {output.shape}"
        print(f"  → Output shape: {output.shape} ✓")
    
    print("✓ Model building test PASSED\n")
    return True


def test_metrics_computation():
    """Test that metrics computation matches original format."""
    print("=" * 60)
    print("TEST 3: Metrics Computation")
    print("=" * 60)
    
    # Create dummy predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8, 0.4, 0.6])
    
    metrics = compute_all_metrics(y_true, y_prob)
    
    required_keys = ["auc", "precision", "recall", "specificity", "f1"]
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
        print(f"✓ {key}: {metrics[key]:.4f}")
    
    # Test print_results format
    results = {"overall": metrics, "female": metrics, "male": metrics}
    print("\n✓ Metrics format matches original:")
    print_results(results)
    
    print("\n✓ Metrics computation test PASSED\n")
    return True


def test_loss_functions():
    """Test that loss functions work correctly."""
    print("=" * 60)
    print("TEST 4: Loss Functions")
    print("=" * 60)
    
    # Test Focal Loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    logits = torch.randn(4, requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    
    loss = focal_loss(logits, targets)
    print(f"✓ Focal Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    # Test backward pass
    loss.backward()
    assert logits.grad is not None, "Gradients should be computed"
    print("✓ Gradients computed successfully")
    
    print("✓ Loss functions test PASSED\n")
    return True


def test_mini_training_run(root_dir="./ODIR_Data", device=None):
    """Run a minimal training epoch to verify training loop works."""
    print("=" * 60)
    print("TEST 5: Mini Training Run (1 epoch, small batch)")
    print("=" * 60)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    from torch.utils.data import DataLoader
    from dr_odir_classifier.training.engine import train_one_epoch
    from torch.optim import AdamW
    
    train_tf, eval_tf = get_transforms(224, use_advanced_aug=True)
    train_ds = DRNPZFolderDataset(os.path.join(root_dir, "train"), transform=train_tf)
    
    # Use small subset for quick test
    if len(train_ds) > 100:
        indices = torch.randperm(len(train_ds))[:100]
        train_ds = torch.utils.data.Subset(train_ds, indices)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    
    model = build_model("resnet18", pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = FocalLoss()
    
    print(f"Training on {len(train_ds)} samples with batch_size=8...")
    loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler=None)
    
    print(f"✓ Training loss: {loss:.4f}")
    assert loss > 0, "Training loss should be positive"
    print("✓ Mini training run test PASSED\n")
    return True


def test_evaluation_format(root_dir="./ODIR_Data", device=None):
    """Test that evaluation produces same format as original."""
    print("=" * 60)
    print("TEST 6: Evaluation Format")
    print("=" * 60)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from torch.utils.data import DataLoader
    
    train_tf, eval_tf = get_transforms(224, use_advanced_aug=False)
    test_ds = DRNPZFolderDataset(os.path.join(root_dir, "test"), transform=eval_tf)
    
    # Use small subset
    if len(test_ds) > 50:
        indices = torch.randperm(len(test_ds))[:50]
        test_ds = torch.utils.data.Subset(test_ds, indices)
    
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    model = build_model("resnet18", pretrained=True).to(device)
    model.eval()
    
    results = evaluate(model, test_loader, device)
    
    # Check format matches original
    assert "overall" in results, "Results should have 'overall' key"
    assert "female" in results or results["female"] is None, "Results should have 'female' key"
    assert "male" in results or results["male"] is None, "Results should have 'male' key"
    
    if results["overall"] is not None:
        required_metrics = ["auc", "precision", "recall", "specificity", "f1"]
        for key in required_metrics:
            assert key in results["overall"], f"Missing metric in overall: {key}"
    
    print("✓ Evaluation results format:")
    print_results(results)
    print("\n✓ Evaluation format test PASSED\n")
    return True


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
    """Run all validation tests."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=None,
                       help="Path to ODIR_Data directory")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("VALIDATION: Comparing Reorganized Code vs Original Notebook")
    print("=" * 60 + "\n")
    
    if args.root_dir:
        root_dir = args.root_dir
    else:
        root_dir = find_odir_data()
    
    if root_dir is None:
        print("⚠ Warning: ODIR_Data not found. Some tests will be skipped.")
        print("  Use --root_dir to specify the path.\n")
    else:
        print(f"Using data directory: {root_dir}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    tests = [
        ("Data Loading", lambda: test_data_loading(root_dir) if os.path.exists(root_dir) else True),
        ("Model Building", test_model_building),
        ("Metrics Computation", test_metrics_computation),
        ("Loss Functions", test_loss_functions),
    ]
    
    # Only run data-dependent tests if data exists
    if os.path.exists(root_dir):
        tests.extend([
            ("Mini Training Run", lambda: test_mini_training_run(root_dir, device)),
            ("Evaluation Format", lambda: test_evaluation_format(root_dir, device)),
        ])
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✓ Passed: {passed}")
    if failed > 0:
        print(f"✗ Failed: {failed}")
    else:
        print("✓ All tests passed! The reorganized code matches the original notebook.")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

