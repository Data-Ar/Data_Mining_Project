from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from .metrics import compute_all_metrics


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Dict[str, float]]:
    """Evaluate a model without test-time augmentation."""

    model.eval()
    all_probs, all_labels, all_sex = [], [], []

    with torch.no_grad():
        for images, labels, sex in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_sex.append(sex.numpy())

    return _group_metrics(all_labels, all_probs, all_sex)


def evaluate_with_tta(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_augments: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Evaluate using simple test-time augmentation strategies."""

    model.eval()
    all_probs, all_labels, all_sex = [], [], []

    with torch.no_grad():
        for images, labels, sex in loader:
            images = images.to(device)
            probs_list: List[torch.Tensor] = []

            logits = model(images).squeeze(1)
            probs_list.append(torch.sigmoid(logits))

            if num_augments >= 2:
                probs_list.append(torch.sigmoid(model(torch.flip(images, dims=[3])).squeeze(1)))
            if num_augments >= 3:
                probs_list.append(torch.sigmoid(model(torch.flip(images, dims=[2])).squeeze(1)))
            if num_augments >= 4:
                probs_list.append(torch.sigmoid(model(torch.flip(images, dims=[2, 3])).squeeze(1)))
            if num_augments >= 5:
                probs_list.append(torch.sigmoid(model(torch.rot90(images, k=1, dims=[2, 3])).squeeze(1)))

            avg_probs = torch.stack(probs_list).mean(dim=0).cpu().numpy()
            all_probs.append(avg_probs)
            all_labels.append(labels.numpy())
            all_sex.append(sex.numpy())

    return _group_metrics(all_labels, all_probs, all_sex)


def _group_metrics(all_labels, all_probs, all_sex):
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    sex = np.concatenate(all_sex).astype(int)

    results: Dict[str, Optional[Dict[str, float]]] = {"overall": compute_all_metrics(labels, probs)}

    for code, name in [(0, "female"), (1, "male")]:
        idx = sex == code
        results[name] = compute_all_metrics(labels[idx], probs[idx]) if idx.any() else None

    return results


def plot_confusion_matrix(true, prob, save_path: str) -> None:
    """Persist a confusion matrix for binary classification."""

    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=14)

    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

