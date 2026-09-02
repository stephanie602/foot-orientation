#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 v20/test 上 Top-Center 的表现：拆开 CNN 单独 / RF 单独 / 融合，
并测试不同 crop padding 的敏感度，同时给出 Top-Center 的混淆去向。
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

C9 = ["Bottom-Center", "Bottom-Lateral", "Bottom-Medial", "Center-Center",
      "Center-Lateral", "Center-Medial", "Top-Center", "Top-Lateral", "Top-Medial"]
KEEP = ["Bottom-Center", "Center-Center", "Center-Lateral", "Center-Medial", "Top-Center"]
KID = {c: i for i, c in enumerate(KEEP)}

ROOT = Path("v20")
CKPT = Path("best_model.pth")
PADS = [0.0, 0.1, 0.2, 0.3]


def kpo(k):
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
    return [L, th, np.sin(th), np.cos(th), gap / L, area / (L * L), gap, bw, bh,
            bw / max(bh, 1e-6), bw * bh, (bw * bh) / (L * L),
            cx, cy, heel[0], heel[1], toe[0], toe[1], cen[0], cen[1],
            (cx - cen[0]) / L, (cy - cen[1]) / L, float(d @ u) / L, float(d @ n) / L]


def parse(l):
    p = l.split()
    if len(p) < 14:
        return None
    return (int(p[0]), [float(x) for x in p[1:5]],
            [np.array([float(p[5 + 3 * i]), float(p[6 + 3 * i]), float(p[7 + 3 * i])]) for i in range(3)])


def crop(img, xc, yc, bw, bh, pad):
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


def report(tag, P, G):
    P, G = np.array(P), np.array(G)
    acc = float((P == G).mean())
    f1 = []
    for i in range(5):
        tp = int(((P == i) & (G == i)).sum())
        fp = int(((P == i) & (G != i)).sum())
        fn = int(((P != i) & (G == i)).sum())
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    line = f"{tag:28s} acc={acc:.4f} macroF1={np.mean(f1):.4f}"
    for i, c in enumerate(KEEP):
        m = G == i
        if m.sum():
            line += f"  {c.split('-')[0][:2]}{c.split('-')[1][:2]}={float((P[m] == i).mean()):.3f}"
    print(line, flush=True)
    return P, G


def main() -> None:
    X, y = [], []
    for lbl in sorted((ROOT / "train" / "labels").glob("*.txt")):
        for l in lbl.read_text().splitlines():
            r = parse(l)
            if not r:
                continue
            c, b, k = r
            if C9[c] not in KID:
                continue
            X.append(geom(b, kpo(k)))
            y.append(KID[C9[c]])
    rf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(np.array(X), np.array(y))
    print(f"RF 训练样本 {len(X)}", flush=True)

    wt = EfficientNet_V2_S_Weights.DEFAULT
    mn, sd = ((tuple(wt.meta["mean"]), tuple(wt.meta["std"])) if "mean" in wt.meta
              else (tuple(wt.transforms().mean), tuple(wt.transforms().std)))
    tfm = build_transforms(448, mean=mn, std=sd, train=False)
    net = load_model(CKPT, num_classes=5).eval()

    # 逐 pad 收集 CNN 概率；RF 概率与 pad 无关，只算一次
    PC = {p: [] for p in PADS}
    PG, G = [], []
    files = sorted((ROOT / "test" / "labels").glob("*.txt"))
    for n, f in enumerate(files, 1):
        ip = ROOT / "test" / "images" / (f.stem + ".jpg")
        if not ip.exists():
            continue
        img = None
        for l in f.read_text().splitlines():
            r = parse(l)
            if not r:
                continue
            c, b, k = r
            if C9[c] not in KID:
                continue
            if img is None:
                img = cv2.imread(str(ip))
            ko = kpo(k)
            ok = True
            probs = {}
            for pad in PADS:
                cc = crop(img, *b, pad=pad)
                if cc is None:
                    ok = False
                    break
                x = tfm(Image.fromarray(cv2.cvtColor(cc, cv2.COLOR_BGR2RGB))).unsqueeze(0)
                with torch.no_grad():
                    probs[pad] = torch.softmax(net(x), 1).numpy()[0]
            if not ok:
                continue
            for pad in PADS:
                PC[pad].append(probs[pad])
            PG.append(rf.predict_proba(np.array([geom(b, ko)]))[0])
            G.append(KID[C9[c]])
        if n % 300 == 0:
            print(f"  {n}/{len(files)}", flush=True)

    G = np.array(G)
    PG = np.array(PG)
    print(f"\n评估样本 {len(G)}（v20/test 共 1273 个框，其余 {1273-len(G)} 个属于模型没有的 4 个类，被跳过）\n")
    print("列末缩写: BoCe=Bottom-Center CeCe=Center-Center CeLa=Center-Lateral CeMe=Center-Medial ToCe=Top-Center\n")

    report("RF 单独(几何)", PG.argmax(1), G)
    print()
    for pad in PADS:
        pc = np.array(PC[pad])
        report(f"CNN 单独 pad={pad}", pc.argmax(1), G)
    print()
    for pad in PADS:
        pc = np.array(PC[pad])
        report(f"融合 CNN×RF pad={pad}", (pc * PG).argmax(1), G)

    # Top-Center 的混淆去向（CNN 单独, pad=0.2）
    print("\n=== Top-Center 被 CNN 单独(pad=0.2)判成了什么 ===")
    pc = np.array(PC[0.2]).argmax(1)
    m = G == KID["Top-Center"]
    print(f"真值 Top-Center n={int(m.sum())}")
    for k, v in Counter(KEEP[i] for i in pc[m].tolist()).most_common():
        print(f"   → {k:16s} {v:4d}  ({v/int(m.sum())*100:.1f}%)")

    print("\n=== 融合(pad=0.2) 下 Top-Center 的去向 ===")
    pf = (np.array(PC[0.2]) * PG).argmax(1)
    for k, v in Counter(KEEP[i] for i in pf[m].tolist()).most_common():
        print(f"   → {k:16s} {v:4d}  ({v/int(m.sum())*100:.1f}%)")


if __name__ == "__main__":
    main()
