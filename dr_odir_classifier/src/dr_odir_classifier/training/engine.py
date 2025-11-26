from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..data import build_dataloaders
from ..evaluation import evaluate
from ..losses import FocalLoss
from ..metrics import compute_all_metrics
from ..models import build_model


@dataclass
class TrainingConfig:
    arch: str
    root_dir: str = "./ODIR_Data"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    out_dir: str = "./dr_notebook_results"
    csv_path: Optional[str] = "./dr_results.csv"
    device: Optional[torch.device] = None
    use_focal_loss: bool = True
    use_class_weights: bool = True
    early_stopping_patience: int = 10
    use_mixed_precision: bool = True
    use_advanced_aug: bool = True
    plot_results: bool = False
    checkpoint_tag: Optional[str] = None
    extra_arches: List[str] = field(default_factory=list)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images).squeeze(1)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images).squeeze(1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def train_backbone(config: TrainingConfig) -> Tuple[nn.Module, Dict[str, Dict[str, float]]]:
    """Train a single backbone and return the fitted model and test metrics."""

    os.makedirs(config.out_dir, exist_ok=True)
    device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(
        root_dir=config.root_dir,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_advanced_aug=config.use_advanced_aug,
        device=device,
    )

    model = build_model(config.arch, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True, min_lr=1e-7)

    if config.use_class_weights:
        labels = loaders["train"].dataset.labels.astype(int)
        class_counts = np.bincount(labels)
        total = len(labels)
        class_weights = torch.tensor([total / (2.0 * count) for count in class_counts], dtype=torch.float32, device=device)
        pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else torch.tensor(1.0, device=device)
    else:
        pos_weight = torch.tensor(1.0, device=device)

    loss_fn = FocalLoss() if config.use_focal_loss else nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision and device.type == "cuda" else None

    best_val_auc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, loss_fn, device, scaler)

        val_probs, val_labels = [], []
        model.eval()
        with torch.no_grad():
            for images, labels, _ in loaders["val"]:
                images = images.to(device)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(images).squeeze(1)
                        probs = torch.sigmoid(logits).cpu().numpy()
                else:
                    logits = model(images).squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(probs)
                val_labels.append(labels.numpy())

        metrics = compute_all_metrics(np.concatenate(val_labels), np.concatenate(val_probs))
        scheduler.step(metrics["auc"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{config.arch}] Epoch {epoch}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AUC: {metrics['auc']:.4f} | "
            f"Val F1: {metrics['f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if metrics["auc"] > best_val_auc:
            best_val_auc = metrics["auc"]
            best_state = model.state_dict()
            patience_counter = 0
            print(f"  ✓ New best AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"  ⚠ Early stopping after {config.early_stopping_patience} epochs without improvement")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        tag = config.checkpoint_tag or config.arch
        ckpt_path = os.path.join(config.out_dir, f"best_{tag}.pt")
        torch.save(
            {
                "model_state_dict": best_state,
                "best_val_auc": best_val_auc,
                "arch": config.arch,
                "img_size": config.img_size,
            },
            ckpt_path,
        )
        print(f"Best checkpoint saved to {ckpt_path}")

    test_results = evaluate(model, loaders["test"], device)
    return model, test_results


def train_multiple_backbones(config: TrainingConfig) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Train several backbones sequentially and persist their metrics."""

    arch_list = [config.arch] + config.extra_arches if config.extra_arches else [config.arch]
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for idx, arch in enumerate(arch_list):
        print("\n" + "=" * 45)
        print(f"TRAINING BACKBONE: {arch}")
        print("=" * 45 + "\n")

        current_cfg = TrainingConfig(**{**config.__dict__, "arch": arch})
        model, metrics = train_backbone(current_cfg)
        results[arch] = metrics

        if config.csv_path:
            _append_metrics_to_csv(config.csv_path, arch, metrics)

    if config.plot_results:
        from ..plots import plot_backbone_comparison

        plot_path = os.path.join(config.out_dir, "backbone_auc_comparison.png")
        plot_backbone_comparison(results, metric="auc", save_path=plot_path)
        print(f"Comparison plot saved to {plot_path}")

    return results


def _append_metrics_to_csv(csv_path: str, arch: str, metrics: Dict[str, Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["backbone", "group", "auc", "precision", "recall", "specificity", "f1"])
        for group, values in metrics.items():
            if values is None:
                writer.writerow([arch, group, "NA", "NA", "NA", "NA", "NA"])
            else:
                writer.writerow(
                    [
                        arch,
                        group,
                        values["auc"],
                        values["precision"],
                        values["recall"],
                        values["specificity"],
                        values["f1"],
                    ]
                )

