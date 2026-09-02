from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max", min_delta: float = 0.0):
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.patience <= 0:
            return False
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False
        improve = (value - self.best) > self.min_delta if self.mode == "max" else (self.best - value) > self.min_delta
        if improve:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    pred = pred.astype(np.int64)
    target = target.astype(np.int64)
    k = (target >= 0) & (target < num_classes)
    idx = num_classes * target[k] + pred[k]
    cm = np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    return cm


def prf_from_cm(cm: np.ndarray) -> dict[str, float]:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    macro_p = float(np.mean(precision)) if precision.size else 0.0
    macro_r = float(np.mean(recall)) if recall.size else 0.0
    macro_f1 = float(np.mean(f1)) if f1.size else 0.0
    total = float(np.sum(support))
    weights = (support / total) if total > 0 else np.zeros_like(support)
    weighted_f1 = float(np.sum(f1 * weights)) if f1.size else 0.0
    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def save_checkpoint(path: Path, model: torch.nn.Module, cfg: Any, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg,
        "extra": extra,
    }
    torch.save(obj, path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def try_plot_curves(run_dir: Path, history: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not history:
        return
    x = [h["epoch"] for h in history]
    def plot_one(keys: list[str], out_name: str, title: str) -> None:
        plt.figure()
        for k in keys:
            plt.plot(x, [h.get(k, 0.0) for h in history], label=k)
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.title(title)
        plt.legend()
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(run_dir / "plots" / out_name, dpi=200, bbox_inches="tight")
        plt.close()
    plot_one(["train_loss", "val_loss"], "loss.png", "loss")
    plot_one(["train_acc", "val_acc"], "acc.png", "acc")
    plot_one(["val_macro_f1", "val_weighted_f1"], "f1.png", "f1")


def try_plot_confusion_matrix(run_dir: Path, cm: np.ndarray, class_names: list[str], normalize: bool = True) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    mat = cm.astype(np.float64)
    if normalize:
        s = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, s, out=np.zeros_like(mat), where=s > 0)
    plt.figure(figsize=(10, 8))
    plt.imshow(mat, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v > 0:
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(run_dir / "plots" / "confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close()
