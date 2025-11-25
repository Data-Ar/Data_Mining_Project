#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from dr_odir_classifier.data import build_dataloaders
from dr_odir_classifier.evaluation import evaluate, evaluate_with_tta
from dr_odir_classifier.metrics import format_results
from dr_odir_classifier.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained backbone with optional TTA.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved .pt checkpoint.")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--root_dir", type=str, default="./ODIR_Data")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation.")
    parser.add_argument("--num_augments", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(
        root_dir=args.root_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_advanced_aug=False,
        device=device,
    )

    model = build_model(args.arch, pretrained=False).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    if args.tta:
        metrics = evaluate_with_tta(model, loaders["test"], device, num_augments=args.num_augments)
    else:
        metrics = evaluate(model, loaders["test"], device)

    print(format_results(metrics))


if __name__ == "__main__":
    main()

