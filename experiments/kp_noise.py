#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""关键点噪声敏感度：RF / CNN / 融合 在关键点不准时各自退化多少。

之前所有 RF 数字都用的是标注里的真值关键点。上线时关键点来自检测模型，
必然有误差。这里对关键点加高斯噪声（以脚长 L 为单位），看结论还站不站得住。
"""
from __future__ import annotations

import pathlib
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

R = Path("EfficientNetV2/foot_classification")
CKPT = Path("best_model.pth")
CACHE = Path("cnn_probs_test.npz")
NOISE = [0.0, 0.02, 0.05, 0.10, 0.20]   # 相对脚长 L 的高斯 sigma


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


def crop(img, xc, yc, bw, bh, pad=0.2):
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


def load_split(split):
    """返回 [(box, kp_ordered, label_id)]"""
    out = []
    for lb in sorted((R / split / "labels").glob("*.txt")):
        for l in lb.read_text().splitlines():
            r = parse(l)
            if not r:
                continue
            c, b, k = r
            if C9[c] not in KID:
                continue
            out.append((lb.stem, b, kpo(k), KID[C9[c]]))
    return out


def scores(P, G):
    acc = float((P == G).mean())
    rec = []
    for i in range(5):
        m = G == i
        rec.append(float((P[m] == i).mean()) if m.sum() else 0.0)
    f1 = []
    for i in range(5):
        tp = int(((P == i) & (G == i)).sum()); fp = int(((P == i) & (G != i)).sum()); fn = int(((P != i) & (G == i)).sum())
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return acc, float(np.mean(f1)), rec


def main() -> None:
    tr = load_split("train")
    te = load_split("test")
    print(f"train {len(tr)}  test {len(te)}", flush=True)

    rf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(
        np.array([geom(b, k) for _, b, k, _ in tr]), np.array([y for *_, y in tr]))

    G = np.array([y for *_, y in te])

    # CNN 概率只算一次并缓存
    if CACHE.exists():
        PC = np.load(CACHE)["p"]
        print("复用缓存的 CNN 概率", flush=True)
    else:
        wt = EfficientNet_V2_S_Weights.DEFAULT
        mn, sd = ((tuple(wt.meta["mean"]), tuple(wt.meta["std"])) if "mean" in wt.meta
                  else (tuple(wt.transforms().mean), tuple(wt.transforms().std)))
        tfm = build_transforms(448, mean=mn, std=sd, train=False)
        net = load_model(CKPT, num_classes=5).eval()
        PC = []
        cur, img = None, None
        for n, (stem, b, k, y) in enumerate(te, 1):
            if stem != cur:
                cur = stem
                img = cv2.imread(str(R / "test" / "images" / f"{stem}.jpg"))
            cc = crop(img, *b)
            x = tfm(Image.fromarray(cv2.cvtColor(cc, cv2.COLOR_BGR2RGB))).unsqueeze(0)
            with torch.no_grad():
                PC.append(torch.softmax(net(x), 1).numpy()[0])
            if n % 200 == 0:
                print(f"  CNN {n}/{len(te)}", flush=True)
        PC = np.array(PC)
        np.savez(CACHE, p=PC)

    print(f"\n{'配置':26s} {'acc':>7s} {'macroF1':>8s}   " + " ".join(f"{c.split('-')[0][:2]+c.split('-')[1][:2]:>6s}" for c in KEEP))

    def line(tag, P):
        a, f, r = scores(P, G)
        print(f"{tag:26s} {a:7.4f} {f:8.4f}   " + " ".join(f"{v:6.3f}" for v in r), flush=True)

    line("CNN 单独", PC.argmax(1))

    rng = np.random.default_rng(0)
    for s in NOISE:
        # 每个样本用 5 次噪声重复，取平均表现
        accs_rf, accs_fu = [], []
        Prf_last = Pfu_last = None
        for rep in range(5 if s > 0 else 1):
            Xn = []
            for _, b, k, _ in te:
                L = np.linalg.norm((k[1][:2] + k[2][:2]) / 2 - k[0][:2]) or 1e-6
                kn = [np.array([p[0] + rng.normal(0, s * L), p[1] + rng.normal(0, s * L), p[2]]) for p in k]
                Xn.append(geom(b, kpo(kn) if s > 0 else k))
            PG = rf.predict_proba(np.array(Xn))
            Prf, Pfu = PG.argmax(1), (PC * PG).argmax(1)
            accs_rf.append(float((Prf == G).mean())); accs_fu.append(float((Pfu == G).mean()))
            Prf_last, Pfu_last = Prf, Pfu
        line(f"RF 单独  噪声σ={s:.2f}", Prf_last)
        line(f"融合     噪声σ={s:.2f}", Pfu_last)
        if s > 0:
            print(f"{'':26s} (RF acc 5次均值 {np.mean(accs_rf):.4f} / 融合 {np.mean(accs_fu):.4f})")
        print()


if __name__ == "__main__":
    main()
