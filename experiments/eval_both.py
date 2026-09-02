#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在两套 test 标签上评估 best_model.pth，对比差别。

  A. 原始人工标注      v19/test/labels          —— 推送前的状态
  B. 模型重刷后的标注  v19_relabeled/test/labels —— 现在 Roboflow 上的状态

B 的分数恒等于 100%（标签就是这个模型的输出），列出来是为了说明它不能用于衡量。
"""
from __future__ import annotations

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

CLASSES9 = ["Bottom-Center", "Bottom-Lateral", "Bottom-Medial", "Center-Center",
            "Center-Lateral", "Center-Medial", "Top-Center", "Top-Lateral", "Top-Medial"]
KEEP = ["Bottom-Center", "Center-Center", "Center-Lateral", "Center-Medial", "Top-Center"]
KID = {c: i for i, c in enumerate(KEEP)}
CKPT = "best_model.pth"


def kp_order(k):
    prs = list(combinations(range(3), 2))
    d = [np.linalg.norm(k[i][:2] - k[j][:2]) for i, j in prs]
    i, j = prs[int(np.argmin(d))]
    h = ({0, 1, 2} - {i, j}).pop()
    return [k[h], k[i], k[j]]


def geom(box, k):
    cx, cy, bw, bh = box
    heel, ta, tb = k[0][:2], k[1][:2], k[2][:2]
    toe = (ta + tb) / 2
    ax = toe - heel
    L = float(np.linalg.norm(ax)) or 1e-6
    th = float(np.arctan2(ax[1], ax[0]))
    gap = float(np.linalg.norm(ta - tb))
    v1, v2 = ta - heel, tb - heel
    area = abs(float(v1[0] * v2[1] - v1[1] * v2[0])) / 2
    cen = (heel + ta + tb) / 3
    u = ax / L
    n = np.array([-u[1], u[0]])
    d = np.array([cx, cy]) - heel
    return [L, th, np.sin(th), np.cos(th), gap / L, area / (L * L), gap,
            bw, bh, bw / max(bh, 1e-6), bw * bh, (bw * bh) / (L * L),
            cx, cy, heel[0], heel[1], toe[0], toe[1], cen[0], cen[1],
            (cx - cen[0]) / L, (cy - cen[1]) / L, float(d @ u) / L, float(d @ n) / L]


def parse(line):
    p = line.split()
    if len(p) < 14:
        return None
    return (int(p[0]), [float(x) for x in p[1:5]],
            [np.array([float(p[5 + 3 * i]), float(p[6 + 3 * i]), float(p[7 + 3 * i])])
             for i in range(3)])


def sq_crop(img, xc, yc, bw, bh, pad=0.2):
    h, w = img.shape[:2]
    nw, nh = bw * (1 + pad), bh * (1 + pad)
    x1, y1 = max(0, int((xc - nw / 2) * w)), max(0, int((yc - nh / 2) * h))
    x2, y2 = min(w, int((xc + nw / 2) * w)), min(h, int((yc + nh / 2) * h))
    c = img[y1:y2, x1:x2]
    if c.size == 0:
        return None
    ch, cw = c.shape[:2]
    s = max(ch, cw)
    return cv2.copyMakeBorder(c, (s - ch) // 2, (s - ch + 1) // 2,
                              (s - cw) // 2, (s - cw + 1) // 2,
                              cv2.BORDER_CONSTANT, value=[114, 114, 114])


def metrics(y, p, k=5):
    acc = float((y == p).mean())
    f1 = []
    for i in range(k):
        tp = int(((p == i) & (y == i)).sum())
        fp = int(((p == i) & (y != i)).sum())
        fn = int(((p != i) & (y == i)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return acc, float(np.mean(f1))


def main() -> None:
    # 几何模型（用 train 拟合，与重刷时一致）
    X, y = [], []
    for lbl in sorted(Path("v19/train/labels").glob("*.txt")):
        for line in lbl.read_text().splitlines():
            r = parse(line)
            if not r:
                continue
            cls, box, k = r
            if CLASSES9[cls] not in KID:
                continue
            X.append(geom(box, kp_order(k))); y.append(KID[CLASSES9[cls]])
    rf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(
        np.array(X), np.array(y))
    print(f"几何模型: {len(X)} 样本\n")

    wt = EfficientNet_V2_S_Weights.DEFAULT
    mean, std = ((tuple(wt.meta["mean"]), tuple(wt.meta["std"])) if "mean" in wt.meta
                 else (tuple(wt.transforms().mean), tuple(wt.transforms().std)))
    tfm = build_transforms(448, mean=mean, std=std, train=False)
    net = load_model(Path(CKPT), num_classes=5).eval()

    # 一次推理，两套标签各自比对
    preds, gt_orig, gt_new = [], [], []
    lbls = sorted(Path("v19/test/labels").glob("*.txt"))
    for n, lbl in enumerate(lbls, 1):
        ip = Path("v19/test/images") / (lbl.stem + ".jpg")
        new_lbl = Path("v19_relabeled/test/labels") / lbl.name
        if not ip.exists() or not new_lbl.exists():
            continue
        oldl = lbl.read_text().splitlines()
        newl = new_lbl.read_text().splitlines()
        img = None
        for li, line in enumerate(oldl):
            r = parse(line)
            if not r:
                continue
            cls, box, k = r
            if CLASSES9[cls] not in KID:
                continue
            rn = parse(newl[li]) if li < len(newl) else None
            if rn is None:
                continue
            if img is None:
                img = cv2.imread(str(ip))
            c = sq_crop(img, *box)
            if c is None:
                continue
            k = kp_order(k)
            x = tfm(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))).unsqueeze(0)
            with torch.no_grad():
                pc = torch.softmax(net(x), 1).numpy()[0]
            pg = rf.predict_proba(np.array([geom(box, k)]))[0]
            preds.append(int((pc * pg).argmax()))
            gt_orig.append(KID[CLASSES9[cls]])
            gt_new.append(KID[CLASSES9[rn[0]]])
        if n % 300 == 0:
            print(f"  {n}/{len(lbls)}", flush=True)

    p = np.array(preds); a = np.array(gt_orig); b = np.array(gt_new)
    print(f"\n评估样本 {len(p)}\n")
    for tag, gt in (("A. 原始人工标注（推送前）", a), ("B. 模型重刷标注（现在 Roboflow 上）", b)):
        acc, f1 = metrics(gt, p)
        print(f"{tag}")
        print(f"   accuracy={acc:.4f}   macro_f1={f1:.4f}")
        for i, c in enumerate(KEEP):
            m = gt == i
            if m.sum():
                print(f"     {c:16s} n={int(m.sum()):4d}  recall={float((p[m]==i).mean()):.4f}")
        print()
    print(f"两套标签本身的差异: {int((a != b).sum())} / {len(a)} "
          f"({(a != b).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
