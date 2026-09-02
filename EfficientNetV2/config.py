from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    dataset_dir: Path
    image_size: int = 448
    model: str = "efficientnet_v2_s"
    pretrained: bool = True
    epochs: int = 200
    batch_size: int = 64
    num_workers: int = 8
    lr: float = 5e-4
    weight_decay: float = 5e-2
    label_smoothing: float = 0.05
    amp: bool = True
    data_parallel: bool = True
    seed: int = 0
    patience: int = 20
    monitor: str = "macro_f1"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    mixup: float = 0.0
    cutmix: float = 0.0
    focal_gamma: float = 2.0
    loss: str = "ce"
    log_interval: int = 50
    run_dir: Path | None = None

