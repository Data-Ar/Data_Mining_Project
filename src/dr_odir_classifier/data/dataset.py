from __future__ import annotations

import glob
import os
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import get_transforms


class DRNPZFolderDataset(Dataset):
    """Dataset that reads pre-processed .npz samples exported from ODIR."""

    def __init__(self, folder_path: str, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {folder_path}")

        self.images = []
        self.labels = []
        self.sex = []

        for path in self.files:
            data = np.load(path)
            self.images.append(data["slo_fundus"])
            self.labels.append(float(data["dr_class"]))
            self.sex.append(int(data["male"]))

        self.images = np.array(self.images, dtype=object)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.sex = np.array(self.sex, dtype=np.int64)
        print(f"Loaded {len(self.images)} samples from {folder_path}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        if img.ndim == 2:
            pil_img = Image.fromarray(img, mode="L").convert("RGB")
        elif img.ndim == 3:
            if img.shape[2] == 3:
                pil_img = Image.fromarray(img, mode="RGB")
            elif img.shape[2] == 1:
                pil_img = Image.fromarray(img.squeeze(), mode="L").convert("RGB")
            else:
                raise ValueError(f"Unsupported image shape {img.shape}")
        else:
            raise ValueError(f"Unsupported image shape {img.shape}")

        if self.transform is not None:
            pil_img = self.transform(pil_img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        sex = torch.tensor(self.sex[idx], dtype=torch.int64)
        return pil_img, label, sex


def build_dataloaders(
    root_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    use_advanced_aug: bool,
    device: torch.device,
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders with the configured transforms."""

    train_tf, eval_tf = get_transforms(img_size, use_advanced_aug=use_advanced_aug)

    datasets = {
        "train": DRNPZFolderDataset(os.path.join(root_dir, "train"), transform=train_tf),
        "val": DRNPZFolderDataset(os.path.join(root_dir, "val"), transform=eval_tf),
        "test": DRNPZFolderDataset(os.path.join(root_dir, "test"), transform=eval_tf),
    }

    loaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
    return loaders

