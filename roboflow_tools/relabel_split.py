#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 CNN × 24维几何 融合模型重刷 test 标签。

流程：
  1. 用 train 的标注拟合几何模型（随机森林，24 维特征）
  2. 对 test 每个框做融合预测（CNN softmax × 几何 softmax）
  3. 写出新标签，并记录每一处改动

几何特征里最关键的是「趾宽/足长」——它度量透视压缩，反映相机相对足底的
垂直位置，正是 Top/Center/Bottom 三类真正编码的信息，而裁剪会把它丢掉。

只有 5 类有模型输出；4 个角落类保持原标注不动。
"""
from __future__ import annotations

import argparse
import csv
import pathlib

from itertools import combinations
from pathlib import Path

import cv2
from collections import Counter
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
CID = {c: i for i, c in enumerate(CLASSES9)}
KEEP = ["Bottom-Center", "Center-Center", "Center-Lateral", "Center-Medial", "Top-Center"]
KID = {c: i for i, c in enumerate(KEEP)}
PAD = 0.2


def kp_order(k):
    """三点中最近的一对是趾尖，剩下那个是脚跟。"""
    prs = list(combinations(range(3), 2))
    d = [np.linalg.norm(k[i][:2] - k[j][:2]) for i, j in prs]
    i, j = prs[int(np.argmin(d))]
    h = ({0, 1, 2} - {i, j}).pop()
    return [k[h], k[i], k[j]], h != 0


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
    box = [float(x) for x in p[1:5]]
    k = [np.array([float(p[5 + 3 * i]), float(p[6 + 3 * i]), float(p[7 + 3 * i])])
         for i in range(3)]
    return int(p[0]), box, k


def sq_crop(img, xc, yc, bw, bh, pad=PAD):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True, help="含 train/test 的数据目录")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--image-size", type=int, default=448)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)

    # 1) 几何模型
    X, y = [], []
    for lbl in sorted((src / "train" / "labels").glob("*.txt")):
        for line in lbl.read_text().splitlines():
            r = parse(line)
            if not r:
                continue
            cls, box, k = r
            if CLASSES9[cls] not in KID:
                continue
            k, _ = kp_order(k)
            X.append(geom(box, k)); y.append(KID[CLASSES9[cls]])
    X, y = np.array(X), np.array(y)
    print(f"几何模型训练样本 {len(X)}，特征 {X.shape[1]} 维")
    rf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(X, y)

    # 2) 载入 CNN
    wt = EfficientNet_V2_S_Weights.DEFAULT
    mean, std = ((tuple(wt.meta["mean"]), tuple(wt.meta["std"])) if "mean" in wt.meta
                 else (tuple(wt.transforms().mean), tuple(wt.transforms().std)))
    tfm = build_transforms(a.image_size, mean=mean, std=std, train=False)
    net = load_model(Path(a.model), num_classes=5).eval()

    # 3) 重刷 test
    out = dst / a.split / "labels"
    out.mkdir(parents=True, exist_ok=True)
    lbls = sorted((src / a.split / "labels").glob("*.txt"))
    n_box = n_skip = n_kp = n_chg = 0
    changed, log = Counter(), []

    for fi, lbl in enumerate(lbls):
        ip = src / a.split / "images" / (lbl.stem + ".jpg")
        img = None
        lines = []
        for li, line in enumerate(lbl.read_text().splitlines()):
            r = parse(line)
            if r is None:
                if line.strip():
                    lines.append(line)
                continue
            cls, box, k = r
            n_box += 1
            k, swapped = kp_order(k)
            n_kp += swapped
            name = CLASSES9[cls]
            new = cls
            conf = None
            if name in KID and ip.exists():
                if img is None:
                    img = cv2.imread(str(ip))
                c = sq_crop(img, *box) if img is not None else None
                if c is not None:
                    x = tfm(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))).unsqueeze(0)
                    with torch.no_grad():
                        pc = torch.softmax(net(x), 1).numpy()[0]
                    pg = rf.predict_proba(np.array([geom(box, k)]))[0]
                    f = pc * pg
                    f = f / f.sum()
                    kbest = int(f.argmax())
                    conf = float(f[kbest])
                    new = CID[KEEP[kbest]]
                    if new != cls:
                        n_chg += 1
                        changed[(name, KEEP[kbest])] += 1
                        log.append(dict(file=lbl.name, line=li, old=name,
                                        new=KEEP[kbest], conf=f"{conf:.4f}"))
            else:
                n_skip += 1
            o = [str(new)] + [f"{v:.17g}" for v in box]
            for kp in k:
                o += [f"{kp[0]:.17g}", f"{kp[1]:.17g}", f"{kp[2]:.17g}"]
            lines.append(" ".join(o))
        (out / lbl.name).write_text("\n".join(lines) + "\n")
        if (fi + 1) % 300 == 0:
            print(f"  {fi+1}/{len(lbls)}  已改 {n_chg}", flush=True)

    print(f"\n{a.split}: {n_box} 个框")
    print(f"  角落类保持原样 {n_skip}")
    print(f"  关键点重排     {n_kp}")
    print(f"  类别改写       {n_chg}")
    for (o, nn), c in changed.most_common(10):
        print(f"    {o:16s} → {nn:16s} {c:4d}")
    if log:
        with (dst / f"relabel_{a.split}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0]))
            w.writeheader(); w.writerows(log)
    print(f"\n✅ {out}")


if __name__ == "__main__":
    main()
