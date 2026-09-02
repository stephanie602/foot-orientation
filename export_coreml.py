#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把融合模型导出成 CoreML。

融合 = CNN(图像) × RF(几何特征)。CoreML 里神经网络和树集成是两种不同的模型类型，
无法合成一张计算图，所以导出两个 .mlpackage，相乘那一步放在 App 侧（就一行）：

    p = cnnProbs .* rfProbs ;  label = argmax(p)

产物：
    FootCNN.mlpackage   image(448×448 RGB)      → classLabel, classProbability
    FootGeomRF.mlpackage  geom(24 float)        → classLabel, classProbability

RF 因 sklearn 1.9 > coremltools 支持上限(1.5.1)，转换 API 被禁用，
这里直接遍历 sklearn 的树结构手工构建 TreeEnsembleClassifier，不依赖那个 API。
"""
from __future__ import annotations

import argparse
import pathlib
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

import coremltools as ct
from coremltools.models import datatypes
from coremltools.models.tree_ensemble import TreeEnsembleClassifier

from EfficientNetV2.predict import load_model

from foot_geom import (CLASSES9, CLASSES5, C5, N_FEAT, IMAGE_SIZE, CROP_PAD,
                       parse_label_line, order_keypoints, geom, crop_square, scores)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class WrappedCNN(nn.Module):
    """把 ImageNet 归一化和 softmax 焊进图里。

    CoreML 的 ImageType 只有标量 scale + 三通道 bias，做不了逐通道除以不同的 std，
    所以归一化放在这里做，ImageType 那边只负责 /255。
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):                      # x 已是 [0,1]
        x = (x - self.mean) / self.std
        return torch.softmax(self.net(x), dim=1)


def export_cnn(ckpt: Path, out: Path, fp16: bool) -> nn.Module:
    net = load_model(ckpt, num_classes=len(CLASSES5)).eval()
    wrapped = WrappedCNN(net).eval()
    ex = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, ex)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
                             scale=1 / 255.0, bias=[0, 0, 0], color_layout=ct.colorlayout.RGB)],
        classifier_config=ct.ClassifierConfig(CLASSES5),
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.short_description = "足部朝向分类 CNN (EfficientNetV2-S, 5类)。输入 448x448 RGB。"
    mlmodel.input_description["image"] = "裁剪并补成正方形的足部图（bbox 外扩 0.2，灰边 114）"
    mlmodel.save(str(out))
    return wrapped


def export_rf(rf, out: Path):
    """遍历 sklearn 的树，逐节点搭 CoreML TreeEnsembleClassifier。

    sklearn 的判定是 x[feature] <= threshold → 走左子树，对应 BranchOnValueLessThanEqual。
    每棵树的叶子值归一化成概率再除以树数，这样所有树的叶子相加 = predict_proba 的平均。
    """
    n_trees = len(rf.estimators_)
    tec = TreeEnsembleClassifier(features=[("geom", datatypes.Array(N_FEAT))],
                                 class_labels=CLASSES5,
                                 output_features="classProbability")
    tec.set_default_prediction_value([0.0] * len(CLASSES5))

    n_nodes = 0
    for t, est in enumerate(rf.estimators_):
        tr = est.tree_
        for nid in range(tr.node_count):
            left, right = int(tr.children_left[nid]), int(tr.children_right[nid])
            if left == -1:                                   # 叶子
                v = np.asarray(tr.value[nid][0], dtype=np.float64)
                s = v.sum()
                v = (v / s if s > 0 else np.full(len(CLASSES5), 1.0 / len(CLASSES5))) / n_trees
                tec.add_leaf_node(t, nid, v.tolist())   # 内部会自行 enumerate
            else:
                tec.add_branch_node(t, nid, int(tr.feature[nid]), float(tr.threshold[nid]),
                                    "BranchOnValueLessThanEqual", left, right)
            n_nodes += 1
        if (t + 1) % 50 == 0:
            print(f"  树 {t+1}/{n_trees}  累计节点 {n_nodes}", flush=True)

    m = ct.models.MLModel(tec.spec)
    m.short_description = ("足部朝向几何 RF (5类)。输入 24 维特征，"
                           "由 bbox + 3个关键点算出，见 predict_fusion.py 的 geom()。")
    m.input_description["geom"] = "24 维几何特征（顺序必须与 geom() 一致）"
    m.save(str(out))
    print(f"  总节点 {n_nodes}")


def verify_cnn(wrapped: nn.Module, path: Path, n: int = 5):
    """随机图上比对 PyTorch 与 CoreML 的输出。"""
    m = ct.models.MLModel(str(path))
    key = m.get_spec().description.predictedProbabilitiesName
    from PIL import Image
    worst = 0.0
    for _ in range(n):
        arr = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        with torch.no_grad():
            ref = wrapped(torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 255.0).numpy()[0]
        got = m.predict({"image": Image.fromarray(arr)})[key]
        got = np.array([got[c] for c in CLASSES5])
        worst = max(worst, float(np.abs(ref - got).max()))
    print(f"  CNN 最大绝对误差 {worst:.2e}")
    return worst


def verify_rf(rf, path: Path, n: int = 200):
    m = ct.models.MLModel(str(path))
    key = m.get_spec().description.predictedProbabilitiesName
    X = np.random.randn(n, N_FEAT) * 0.5
    ref = rf.predict_proba(X)
    worst = 0.0
    mism = 0
    for i in range(n):
        got = m.predict({"geom": X[i].astype(np.float64)})[key]
        got = np.array([got[c] for c in CLASSES5])
        worst = max(worst, float(np.abs(ref[i] - got).max()))
        mism += int(ref[i].argmax() != got.argmax())
    print(f"  RF 最大绝对误差 {worst:.2e}   argmax 不一致 {mism}/{n}")
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best_model.pth")
    ap.add_argument("--rf", default="rf_geom.joblib")
    ap.add_argument("--outdir", default="coreml")
    ap.add_argument("--fp16", action="store_true", help="CNN 用 fp16，体积减半")
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--skip-rf", action="store_true")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    if not a.skip_cnn:
        print("== 导出 CNN ==", flush=True)
        w = export_cnn(Path(a.ckpt), out / "FootCNN.mlpackage", a.fp16)
        verify_cnn(w, out / "FootCNN.mlpackage")

    if not a.skip_rf:
        print("== 导出 RF ==", flush=True)
        rf = joblib.load(a.rf)
        export_rf(rf, out / "FootGeomRF.mlpackage")
        verify_rf(rf, out / "FootGeomRF.mlpackage")

    print("\n产物:")
    for p in sorted(out.glob("*.mlpackage")):
        sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        print(f"  {p}  {sz/1e6:.1f} MB")


if __name__ == "__main__":
    main()
