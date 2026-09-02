# 测试流程

从零到验收，六步。每步都给了**预期输出**——对不上就停下来查，别往下走。

前置：`pip install -r requirements.txt`，并准备好 `best_model.pth`。

---

## 第 0 步：准备数据

从 Roboflow 导出 v20，**YOLOv8 格式**，解压到 `EfficientNetV2/foot_classification/`：

```bash
export ROBOFLOW_API_KEY=xxxx
curl -sL "$(curl -s "https://api.roboflow.com/mago-ai/keypoint-categorization-foot-3/20/yolov8?api_key=$ROBOFLOW_API_KEY" \
  | python -c 'import sys,json;print(json.load(sys.stdin)["export"]["link"])')" -o rf.zip
mkdir -p EfficientNetV2/foot_classification && unzip -q rf.zip -d EfficientNetV2/foot_classification && rm rf.zip
```

> 压缩包 3.3 GB，10 MB/s 大约 5 分钟。用 `roboflow` 这个 Python 包的 `.download()` 在大数据集上会中途断，用 curl。

**预期**：

```
train: imgs=8948  lbls=8948
valid: imgs=2558  lbls=2558
test:  imgs=1276  lbls=1276
```

---

## 第 1 步：生成分类版式数据集

```bash
python make_cls_dataset.py --src EfficientNetV2/foot_classification \
                           --out foot_classification_cls --classes 9
```

**预期**：`test/Top-Center/` 下 135 张，产出约 1.6 GB，同时写出 `cls_manifest.csv`。

`--dry-run` 可先估体积。磁盘紧张就加 `--max-size 448`。

> `cls_manifest.csv` 不是可有可无的中间产物——第 3 步靠它把裁剪图映射回原始标签取几何。

---

## 第 2 步：单模型基线（复现 64.4%）

```bash
python -m EfficientNetV2.predict --model best_model.pth \
       --dir foot_classification_cls/test/Top-Center
```

**预期**：135 行，其中 top-1 为 `Top-Center` 的**恰好 87 行 = 64.4%**，44 行错判成 `Center-Center`。

这一步是基线，用来对照融合的增益。**注意**：判错时的平均置信度 0.732，判对时 0.792——只差 0.06，还有 9 张是 0.9 以上的高置信度错误。**所以不能靠置信度阈值筛这批错误。**

---

## 第 3 步：融合推理（复现 92.6%）

```bash
python predict_fusion.py --dir foot_classification_cls/test/Top-Center --out tc_fusion.csv
```

**预期**：

```
CNN 单独     87/135  = 0.6444
RF 单独     125/135  = 0.9259
融合        125/135  = 0.9259
```

加 `--no-rf` 可退化成第 2 步的行为，用于交叉验证两条路径一致。

---

## 第 4 步：导出 CoreML

```bash
python export_coreml.py --fp16
```

**预期**：

```
CNN 最大绝对误差 5.28e-03          ← fp16 舍入，正常
RF  最大绝对误差 3.89e-16   argmax 不一致 0/200   ← 必须是这个量级
coreml/FootCNN.mlpackage     40.5 MB
coreml/FootGeomRF.mlpackage  20.3 MB
```

**RF 的误差必须是 1e-15 量级。** 它是手工逐节点构建的，如果这个数变大，说明树结构映射错了（多半是 sklearn 版本变更导致 `tree_.value` 的归一化语义变了），**不要继续往下走**。

CNN 那 5e-03 是 fp16 的正常舍入。不想要就去掉 `--fp16`，体积翻倍到 81 MB。

---

## 第 5 步：端到端验收（最终门槛）

```bash
python test_coreml.py --split test --compare-torch
```

**预期**：

```
=== CoreML 产物 / test  样本 734 ===
                    acc   macroF1   Bottom-Center  Center-Center  Center-Lateral  Center-Medial  Top-Center
CNN 单独           0.8433   0.8451        0.8969         0.8000          0.9493         0.9371      0.6519
RF 单独            0.7629   0.7546        0.4948         0.7902          0.8188         0.7044      0.9259
融合               0.9455   0.9491        0.9794         0.9415          0.9420         0.9434      0.9333

--- 与 PyTorch/sklearn 参考实现对比 ---
  CNN 概率最大绝对差 6.66e-02   RF 概率最大绝对差 1.03e-14
  融合结果不一致 4/734
  参考实现 acc=0.9455   CoreML acc=0.9455
```

**验收门槛**：

| 指标 | 门槛 | 说明 |
|---|---|---|
| 融合 accuracy | ≥ 0.94 | |
| 融合 Top-Center recall | ≥ 0.92 | 这是融合存在的理由，掉了就说明 RF 没生效 |
| RF 概率最大绝对差 | ≤ 1e-12 | 超了就是树映射错了 |
| 融合结果不一致 | ≤ 10/734 | fp16 造成的，4 个是正常水平 |

测已安装的产物用：

```bash
python test_coreml.py --split test --cnn model.mlpackage --rf model_geom_rf.mlpackage
```

---

## 第 6 步：上线前必做（目前还没做）

**量出关键点检测模型的实际误差。** 上面所有 RF / 融合的数字都建立在**真值关键点**上。线上没有真值。

`experiments/kp_noise.py` 给出了敏感度曲线：

| 关键点误差（占脚长） | RF 单独 | 融合 | 相对纯 CNN (0.843) |
|---|---|---|---|
| 0（真值） | 0.763 | 0.946 | +0.10 |
| 5% | 0.663 | 0.909 | +0.07 |
| 10% | 0.522 | 0.876 | +0.03 |
| 20% | 0.384 | 0.852 | **+0.01（收益消失）** |

**结论：关键点误差超过脚长的 20%，融合就不值得上了**——多一个模型的复杂度换不到一个点。

所以上线前必须先跑一遍你们的关键点模型，量出误差落在哪一档，再决定要不要走融合路线。这一步没做之前，0.9455 只是上界，不是线上能拿到的数。

---

## 常见问题

**`KeyError: 'mean'`** — torchvision ≥ 0.13 把 `mean`/`std` 从 `weights.meta` 挪到了 `weights.transforms()`。仓库里已修，遇到说明你在跑旧副本。

**`scikit-learn version 1.9.0 is not supported`** — 这条警告可以忽略。它说的是 coremltools 的 sklearn 转换 API 被禁用了，而 `export_coreml.py` 本来就不用那个 API。

**融合分数和 CNN 一模一样** — RF 没生效。检查 `cls_manifest.csv` 在不在，以及 `predict_fusion.py` 有没有报「找不到几何」。

**改了 `foot_geom.py` 的 `geom()`** — 必须重新导出 RF 并同步 iOS 侧的特征计算。24 维的**顺序就是契约**，错位不会报错，只会静默地变差。
