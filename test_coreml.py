#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用导出的 CoreML 产物端到端测融合，并和 PyTorch/sklearn 的参考实现对齐。

测的是真正要上线的那两个 .mlpackage，不是 Python 里的模型。
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import coremltools as ct

from foot_geom import (CLASSES9, CLASSES5, C5, N_FEAT, IMAGE_SIZE, CROP_PAD,
                       parse_label_line, order_keypoints, geom, crop_square, scores)

SRC = Path("EfficientNetV2/foot_classification")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--coreml", default="coreml", help="放着两个 .mlpackage 的目录")
    ap.add_argument("--cnn", default="", help="CNN 包路径，覆盖 --coreml 的默认命名")
    ap.add_argument("--rf", default="", help="RF 包路径，覆盖 --coreml 的默认命名")
    ap.add_argument("--compare-torch", action="store_true",
                    help="同时跑 PyTorch/sklearn 参考实现，逐样本比对")
    a = ap.parse_args()

    cnn_path = Path(a.cnn) if a.cnn else Path(a.coreml) / "FootCNN.mlpackage"
    rf_path = Path(a.rf) if a.rf else Path(a.coreml) / "FootGeomRF.mlpackage"
    print(f"CNN: {cnn_path}\nRF : {rf_path}", flush=True)
    cnn = ct.models.MLModel(str(cnn_path))
    rfm = ct.models.MLModel(str(rf_path))
    k_cnn = cnn.get_spec().description.predictedProbabilitiesName
    k_rf = rfm.get_spec().description.predictedProbabilitiesName

    ref_net = ref_rf = ref_tfm = None
    if a.compare_torch:
        import joblib, torch
        from torchvision.models import EfficientNet_V2_S_Weights
        from EfficientNetV2.dataset import build_transforms
        from EfficientNetV2.predict import load_model
        w = EfficientNet_V2_S_Weights.DEFAULT
        mean, std = ((tuple(w.meta["mean"]), tuple(w.meta["std"])) if "mean" in w.meta
                     else (tuple(w.transforms().mean), tuple(w.transforms().std)))
        ref_tfm = build_transforms(IMAGE_SIZE, mean=mean, std=std, train=False)
        ref_net = load_model(Path("best_model.pth"), num_classes=5).eval()
        ref_rf = joblib.load("rf_geom.joblib")
        globals()["torch"] = torch

    P_cnn, P_rf, P_fu, G = [], [], [], []
    R_fu = []
    d_cnn = d_rf = 0.0
    files = sorted((SRC / a.split / "labels").glob("*.txt"))
    for n, lb in enumerate(files, 1):
        img = None
        for cls, box, kps in [r for r in (parse_label_line(l) for l in lb.read_text().splitlines()) if r]:
            name = CLASSES9[cls]
            if name not in C5:
                continue
            if img is None:
                img = cv2.imread(str(SRC / a.split / "images" / f"{lb.stem}.jpg"))
                if img is None:
                    break
            patch = crop_square(img, box)
            if patch is None:
                continue
            pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            g = np.array(geom(box, order_keypoints(kps)), dtype=np.float64)

            pc = cnn.predict({"image": pil})[k_cnn]
            pc = np.array([pc[c] for c in CLASSES5])
            pr = rfm.predict({"geom": g})[k_rf]
            pr = np.array([pr[c] for c in CLASSES5])
            fu = pc * pr

            P_cnn.append(int(pc.argmax())); P_rf.append(int(pr.argmax())); P_fu.append(int(fu.argmax()))
            G.append(C5[name])

            if ref_net is not None:
                with torch.no_grad():
                    rpc = torch.softmax(ref_net(ref_tfm(Image.fromarray(
                        cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))).unsqueeze(0)), 1).numpy()[0]
                rpr = ref_rf.predict_proba(g[None])[0]
                d_cnn = max(d_cnn, float(np.abs(rpc - pc).max()))
                d_rf = max(d_rf, float(np.abs(rpr - pr).max()))
                R_fu.append(int((rpc * rpr).argmax()))
        if n % 200 == 0:
            print(f"  {n}/{len(files)}", flush=True)

    G = np.array(G)
    print(f"\n=== CoreML 产物 / {a.split}  样本 {len(G)} ===")
    print(f"{'':14s} {'acc':>8s} {'macroF1':>9s}   " +
          " ".join(f"{c:>14s}" for c in CLASSES5))
    for tag, P in [("CNN 单独", P_cnn), ("RF 单独", P_rf), ("融合", P_fu)]:
        acc, f1, rec = scores(np.array(P), G)
        print(f"{tag:14s} {acc:8.4f} {f1:9.4f}   " + " ".join(f"{v:14.4f}" for v in rec))

    print("\n融合判成:", dict(Counter(CLASSES5[i] for i in P_fu).most_common()))

    if R_fu:
        R_fu = np.array(R_fu); Pf = np.array(P_fu)
        acc_r, _, _ = scores(R_fu, G)
        print(f"\n--- 与 PyTorch/sklearn 参考实现对比 ---")
        print(f"  CNN 概率最大绝对差 {d_cnn:.2e}   RF 概率最大绝对差 {d_rf:.2e}")
        print(f"  融合结果不一致 {int((R_fu != Pf).sum())}/{len(Pf)}")
        print(f"  参考实现 acc={acc_r:.4f}   CoreML acc={float((Pf==G).mean()):.4f}")


if __name__ == "__main__":
    main()
