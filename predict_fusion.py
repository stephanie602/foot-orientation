#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict.py 的融合版：对分类版式的文件夹跑 CNN × 几何RF。

predict.py 只能吃像素，而 RF 需要 bbox + 关键点——那些信息在裁剪时被丢掉了。
这里靠 cls_manifest.csv 回查原始 YOLO 标签，把几何补回来。

用法：
    python predict_fusion.py --dir EfficientNetV2/foot_classification_cls/test/Top-Center
    python predict_fusion.py --dir ... --no-rf      # 退化成 predict.py 的行为，便于对照
"""
from __future__ import annotations

import argparse
import csv
import pathlib
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch

torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from torchvision.models import EfficientNet_V2_S_Weights

from EfficientNetV2.dataset import build_transforms
from EfficientNetV2.predict import load_model

from foot_geom import (CLASSES9, CLASSES5, C5, N_FEAT, IMAGE_SIZE, CROP_PAD,
                       parse_label_line, order_keypoints, geom, crop_square, scores)


SRC = Path("EfficientNetV2/foot_classification")   # 原始 YOLO-pose 数据集
CKPT = Path("best_model.pth")


def label_boxes(split: str, stem: str):
    """回查某张原图的所有框。索引必须和 manifest 的 box 列对齐，所以这里不过滤类别。"""
    p = SRC / split / "labels" / f"{stem}.txt"
    if not p.exists():
        return []
    return [r for r in (parse_label_line(l) for l in p.read_text().splitlines()) if r]


def build_rf() -> RandomForestClassifier:
    """在 train split 上现训几何 RF（参数与 export_coreml.py 导出的那个一致）。"""
    from foot_geom import load_geom_split
    X, y = load_geom_split(SRC, "train")
    print(f"RF 训练样本 {len(X)}", flush=True)
    return RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(X, y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="分类版式的类别文件夹")
    ap.add_argument("--manifest", default="cls_manifest.csv")
    ap.add_argument("--no-rf", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    d = Path(a.dir)
    truth = d.name                      # 文件夹名即真值类别
    imgs = sorted(d.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"{d} 下没有 jpg")

    # 用 manifest 把裁剪图名 → (split, 原图stem, 框号)。按文件名索引，避免路径变动失效。
    idx = {}
    for r in csv.DictReader(open(a.manifest, encoding="utf-8-sig")):
        idx[Path(r["dst"]).name] = (r["split"], Path(r["src"]).stem, int(r["box"]))

    w = EfficientNet_V2_S_Weights.DEFAULT
    mean, std = ((tuple(w.meta["mean"]), tuple(w.meta["std"])) if "mean" in w.meta
                 else (tuple(w.transforms().mean), tuple(w.transforms().std)))
    tfm = build_transforms(448, mean=mean, std=std, train=False)
    net = load_model(CKPT, num_classes=5).eval()
    rf = None if a.no_rf else build_rf()

    rows = []
    n_cnn = n_rf = n_fu = n_nogeom = 0
    for n, p in enumerate(imgs, 1):
        x = tfm(Image.open(p).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            p_cnn = torch.softmax(net(x), 1).numpy()[0]
        p_rf = p_fu = None
        if rf is not None:
            meta = idx.get(p.name)
            if meta:
                split, stem, bi = meta
                bs = label_boxes(split, stem)
                if bi < len(bs):
                    _, box, kps = bs[bi]
                    p_rf = rf.predict_proba(np.array([geom(box, order_keypoints(kps))]))[0]
                    p_fu = p_cnn * p_rf
                    p_fu = p_fu / p_fu.sum()
            if p_rf is None:
                n_nogeom += 1

        c_cnn = CLASSES5[int(p_cnn.argmax())]
        c_rf = CLASSES5[int(p_rf.argmax())] if p_rf is not None else ""
        c_fu = CLASSES5[int(p_fu.argmax())] if p_fu is not None else ""
        n_cnn += int(c_cnn == truth)
        n_rf += int(c_rf == truth)
        n_fu += int(c_fu == truth)
        rows.append(dict(file=p.name, truth=truth, cnn=c_cnn,
                         cnn_p=f"{p_cnn.max():.4f}", rf=c_rf, fusion=c_fu,
                         fusion_p=f"{p_fu.max():.4f}" if p_fu is not None else ""))
        if n % 50 == 0:
            print(f"  {n}/{len(imgs)}", flush=True)

    t = len(imgs)
    print(f"\n=== {d}  真值={truth}  n={t} ===")
    print(f"  CNN 单独   {n_cnn:4d}/{t}  = {n_cnn/t:.4f}")
    if rf is not None:
        print(f"  RF 单独    {n_rf:4d}/{t}  = {n_rf/t:.4f}")
        print(f"  融合       {n_fu:4d}/{t}  = {n_fu/t:.4f}")
        if n_nogeom:
            print(f"  ⚠️ {n_nogeom} 张在 manifest/标签里找不到几何，未参与 RF")
    print("\nCNN 判成:", dict(Counter(r["cnn"] for r in rows).most_common()))
    if rf is not None:
        print("融合判成:", dict(Counter(r["fusion"] for r in rows).most_common()))

    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n逐张结果 → {a.out}")


if __name__ == "__main__":
    main()
