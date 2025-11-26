#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from dr_odir_classifier.config import load_config
from dr_odir_classifier.training.engine import TrainingConfig, train_multiple_backbones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one or more DR backbones.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--root_dir", type=str, default="./ODIR_Data")
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    parser.add_argument("--csv_path", type=str, default="./artifacts/dr_results.csv")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--extra_arches", type=str, nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--plot_results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = TrainingConfig(
            arch=args.arch,
            root_dir=args.root_dir,
            out_dir=args.out_dir,
            csv_path=args.csv_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            plot_results=args.plot_results,
            extra_arches=args.extra_arches,
        )

    train_multiple_backbones(cfg)


if __name__ == "__main__":
    main()

