from __future__ import annotations

import torchvision.transforms as T


def get_transforms(img_size: int, use_advanced_aug: bool = True):
    """Return training and evaluation transforms."""

    if use_advanced_aug:
        train_tf = T.Compose(
            [
                T.Resize((int(img_size * 1.1), int(img_size * 1.1))),
                T.RandomCrop((img_size, img_size)),
                T.RandomRotation(15),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ]
        )
    else:
        train_tf = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomRotation(10),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    eval_tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf

