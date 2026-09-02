#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""足部朝向分类的公共定义：类别、标签解析、关键点排序、几何特征、裁剪。

这些函数原先在十个脚本里各抄了一份。geom() 现在是 CoreML 模型
(model_geom_rf.mlpackage) 的输入契约——24 维特征的顺序必须和 iOS 侧
逐位一致，改一处不同步其余的会导致静默的错误分类（不报错，只是变差）。
所以只留这一份定义。
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

# Roboflow 项目里的 9 个类
CLASSES9 = ["Bottom-Center", "Bottom-Lateral", "Bottom-Medial", "Center-Center",
            "Center-Lateral", "Center-Medial", "Top-Center", "Top-Lateral", "Top-Medial"]

# 模型实际支持的 5 个类（best_model.pth 的 extra.classes）
CLASSES5 = ["Bottom-Center", "Center-Center", "Center-Lateral", "Center-Medial", "Top-Center"]
C5 = {c: i for i, c in enumerate(CLASSES5)}

N_FEAT = 24          # geom() 的输出维度
IMAGE_SIZE = 448     # CNN 输入边长
CROP_PAD = 0.2       # bbox 外扩比例
CROP_FILL = 114      # 补正方形用的灰度值


def parse_label_line(line: str):
    """YOLO-pose 一行 → (类别id, [cx,cy,bw,bh], [3个 (x,y,v)])；不合法返回 None。"""
    p = line.split()
    if len(p) < 14:
        return None
    return (int(p[0]),
            [float(x) for x in p[1:5]],
            [np.array([float(p[5 + 3 * i]), float(p[6 + 3 * i]), float(p[7 + 3 * i])])
             for i in range(3)])


def order_keypoints(k):
    """标签里 3 个关键点没有语义顺序。

    取两两距离最近的一对当脚趾，剩下那个当脚跟，返回 [heel, toe_a, toe_b]。
    这是按几何反推的启发式，不是标注规范里写明的顺序。
    """
    prs = list(combinations(range(3), 2))
    d = [np.linalg.norm(k[i][:2] - k[j][:2]) for i, j in prs]
    i, j = prs[int(np.argmin(d))]
    h = ({0, 1, 2} - {i, j}).pop()
    return [k[h], k[i], k[j]]


def geom(box, k):
    """24 维几何特征。k 必须是 order_keypoints() 排过序的。

    ⚠️ 顺序即契约：与 model_geom_rf.mlpackage 的输入一一对应，不要重排或插入。
    """
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


def crop_square(img, box, pad: float = CROP_PAD):
    """按 bbox 外扩 pad 裁剪，再用灰边补成正方形。img 是 BGR (cv2)。

    训练用的 foot_classification_v3 的原始裁剪脚本已丢失，这里是按同一约定
    重建的。pad 对准确率影响很大（Top-Center recall 在 pad 0→0.3 之间
    从 0.53 变到 0.73），换值前先重跑一遍评估。
    """
    import cv2
    xc, yc, bw, bh = box
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
                              cv2.BORDER_CONSTANT, value=[CROP_FILL] * 3)


def iter_boxes(root: Path, split: str, only5: bool = True):
    """遍历一个 split，逐个吐出 (标签路径, 图像路径, 类别名, box, 排序后的关键点)。"""
    ldir, idir = Path(root) / split / "labels", Path(root) / split / "images"
    for lb in sorted(ldir.glob("*.txt")):
        for r in (parse_label_line(l) for l in lb.read_text().splitlines()):
            if not r:
                continue
            cid, box, kps = r
            name = CLASSES9[cid]
            if only5 and name not in C5:
                continue
            yield lb, idir / f"{lb.stem}.jpg", name, box, order_keypoints(kps)


def load_geom_split(root: Path, split: str):
    """整个 split 的 (X[N,24], y[N])，供 RF 训练/评估。"""
    X, y = [], []
    for _, _, name, box, kps in iter_boxes(root, split):
        X.append(geom(box, kps))
        y.append(C5[name])
    return np.array(X, dtype=np.float64), np.array(y)


def scores(P, G, n_cls: int = len(CLASSES5)):
    """(accuracy, macro_f1, 每类 recall)。"""
    P, G = np.asarray(P), np.asarray(G)
    acc = float((P == G).mean()) if len(G) else 0.0
    rec, f1 = [], []
    for i in range(n_cls):
        m = G == i
        rec.append(float((P[m] == i).mean()) if m.sum() else 0.0)
        tp = int(((P == i) & (G == i)).sum())
        fp = int(((P == i) & (G != i)).sum())
        fn = int(((P != i) & (G == i)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return acc, float(np.mean(f1)), rec
