from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataSpec:
    root: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path


def resolve_data_dirs(dataset_dir: Path) -> DataSpec:
    root = Path(dataset_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    if not val_dir.exists():
        alt = root / "valid"
        val_dir = alt if alt.exists() else val_dir
    test_dir = root / "test"
    return DataSpec(root=root, train_dir=train_dir, val_dir=val_dir, test_dir=test_dir)


def build_transforms(image_size: int, mean: tuple[float, float, float], std: tuple[float, float, float], train: bool) -> transforms.Compose:
    base = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        aug = [
            transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)], p=0.6),
            transforms.RandomRotation(degrees=25),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.85, 1.15), shear=None),
        ]
        base = base + aug
    base = base + [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(base)


def build_datasets(dataset_dir: Path, image_size: int, mean: tuple[float, float, float], std: tuple[float, float, float]):
    spec = resolve_data_dirs(dataset_dir)
    if not spec.train_dir.exists():
        raise FileNotFoundError(f"找不到训练集目录: {spec.train_dir}")
    if not spec.val_dir.exists():
        raise FileNotFoundError(f"找不到验证集目录: {spec.val_dir} (支持 val/ 或 valid/)")
    train_ds = datasets.ImageFolder(str(spec.train_dir), transform=build_transforms(image_size, mean, std, train=True))
    val_ds = datasets.ImageFolder(str(spec.val_dir), transform=build_transforms(image_size, mean, std, train=False))
    test_ds = datasets.ImageFolder(str(spec.test_dir), transform=build_transforms(image_size, mean, std, train=False)) if spec.test_dir.exists() else None
    return train_ds, val_ds, test_ds, train_ds.classes


def compute_class_weights(train_ds: datasets.ImageFolder) -> torch.Tensor:
    targets = np.asarray(train_ds.targets, dtype=np.int64)
    num_classes = len(train_ds.classes)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / np.mean(inv)
    return torch.tensor(w, dtype=torch.float32)

