from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from .dataset import build_transforms
from .utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--image", type=str, default="")
    p.add_argument("--dir", type=str, default="")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--topk", type=int, default=3)
    return p.parse_args()


def load_model(ckpt_path: Path, num_classes: int) -> nn.Module:
    model = efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return model


@torch.no_grad()
def predict_one(model: nn.Module, img_path: Path, tfm, device: torch.device, class_names: list[str], topk: int) -> list[tuple[str, float]]:
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1).squeeze(0)
    v, i = torch.topk(prob, k=min(int(topk), prob.numel()))
    out = []
    for score, idx in zip(v.tolist(), i.tolist()):
        out.append((class_names[idx], float(score)))
    return out


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    classes = ckpt.get("extra", {}).get("classes", None)
    if not classes:
        raise RuntimeError("checkpoint 中未找到类别列表(extra.classes)。请使用 EfficientNetV2/train.py 生成的 best_model.pth。")

    device = get_device()
    weights = EfficientNet_V2_S_Weights.DEFAULT
    # torchvision >= 0.13 起 meta 里不再有 mean/std，改由 transforms() 提供
    mean, std = ((tuple(weights.meta["mean"]), tuple(weights.meta["std"])) if "mean" in weights.meta
                 else (tuple(weights.transforms().mean), tuple(weights.transforms().std)))
    tfm = build_transforms(int(args.image_size), mean=mean, std=std, train=False)
    model = load_model(ckpt_path, num_classes=len(classes))
    model.to(device)
    model.eval()

    if args.image:
        p = Path(args.image)
        res = predict_one(model, p, tfm, device, classes, topk=int(args.topk))
        print(p)
        for name, score in res:
            print(f"{name}\t{score:.6f}")
        return
    if args.dir:
        d = Path(args.dir)
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            imgs.extend(sorted(d.glob(ext)))
        for p in imgs:
            res = predict_one(model, p, tfm, device, classes, topk=int(args.topk))
            s = " | ".join([f"{n}:{sc:.4f}" for n, sc in res])
            print(f"{p.name}\t{s}")
        return
    raise ValueError("请提供 --image 或 --dir")


if __name__ == "__main__":
    main()

