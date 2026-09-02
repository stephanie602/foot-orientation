# 足部朝向分类（CNN × 几何 RF 融合）

把足部照片分成 5 类朝向：`Bottom-Center` / `Center-Center` / `Center-Lateral` / `Center-Medial` / `Top-Center`。

线上模型是**两个模型相乘**，不是单个 CNN：

```
              ┌→ 裁剪成正方形 ──────→ CNN (EfficientNetV2-S) ─┐
关键点检测 ────┤                                              ├→ 逐元素相乘 → argmax
(bbox + 3点)  └→ 24 维几何特征 ─────→ 随机森林 (400 棵树) ────┘
```

## 为什么要融合

单靠 CNN 在 `Top-Center` 上只有 **64.4%**——因为裁剪成正方形的过程把脚的**朝向归一化掉了**，而 Top / Center 的区别恰恰在朝向上。CNN 从像素里看不到，几何特征一眼就能分。

v20 测试集（734 个框）实测：

| | accuracy | macro-F1 | Top-Center recall |
|---|---|---|---|
| CNN 单独 | 0.8433 | 0.8451 | **0.6519** |
| RF 单独 | 0.7629 | 0.7546 | 0.9259 |
| **融合** | **0.9455** | **0.9491** | **0.9333** |

## 仓库结构

```
foot_geom.py           ← 几何特征定义。这是 CoreML 模型的输入契约，改动需同步 iOS 侧
make_cls_dataset.py    ← YOLO-pose → ImageFolder 分类版式
predict_fusion.py      ← 融合推理（Python 参考实现）
export_coreml.py       ← 导出两个 .mlpackage
test_coreml.py         ← 端到端验收，见 TESTING.md
EfficientNetV2/        ← 模型定义与单模型推理
roboflow_tools/        ← 与 Roboflow 标注平台交互
experiments/           ← 一次性分析脚本，结论已写进本文档，日常不需要跑
```

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 从 Roboflow 导出 v20（YOLOv8 格式）到 EfficientNetV2/foot_classification/
export ROBOFLOW_API_KEY=xxxx

# 单模型推理
python -m EfficientNetV2.predict --model best_model.pth --dir <某个类别文件夹>

# 融合推理
python predict_fusion.py --dir foot_classification_cls/test/Top-Center
```

完整的测试与验收流程见 **[TESTING.md](TESTING.md)**。

## 部署产物

| 文件 | 输入 | 输出 |
|---|---|---|
| `model.mlpackage` | `image` 448×448 RGB | `classLabel`, `classLabel_probs` |
| `model_geom_rf.mlpackage` | `geom` 24 维 float | `classProbability` |

App 侧把两组概率相乘再取 argmax 即可：

```swift
let p = zip(cnnProbs, rfProbs).map(*)
let label = CLASSES5[p.firstIndex(of: p.max()!)!]
```

**CoreML 里神经网络和树集成是两种不同的模型类型，无法合成一张计算图**，所以是两个包。

## 已知限制

按重要性排序，前两条会直接影响你怎么解读上面的分数：

1. **RF 依赖关键点精度。** 上表用的是标注里的真值关键点。线上关键点由检测模型给出，误差越大融合收益越小——误差到脚长的 20% 时，融合相对纯 CNN 的优势基本归零（见 `experiments/kp_noise.py`）。**上线前必须先量出你们关键点模型的实际误差。**

2. **测试集标签部分来自模型自身。** v20 是把模型重刷的标注推回 Roboflow 之后的快照，存在循环验证。同一套流程在推送前的纯人工标注（v19）上只有 **0.8218**。真实泛化能力应该更靠近这个数，而不是 0.9455。

3. **裁剪配方是重建的。** 训练用的 `foot_classification_v3` 的原始裁剪脚本已丢失，`crop_square()` 是按同一约定重建的。`pad` 很敏感：0→0.3 之间 Top-Center recall 从 0.53 变到 0.73。

4. **CNN 训练集与测试集可能重叠。** CNN 训练用的是 v3 快照，其中有多少图现在落在 v20 的 test 里无法验证（v3 已不存在）。方向是让分数偏高。

5. **9 类 vs 5 类。** Roboflow 项目有 9 个类，模型只有 5 个。`Bottom-Lateral` / `Bottom-Medial` / `Top-Lateral` / `Top-Medial` 这 4 类（test 里 539 个框）在评估时被整体跳过。

## Roboflow

项目 `mago-ai/keypoint-categorization-foot-3`，当前版本 v20。

API key 从环境变量 `ROBOFLOW_API_KEY` 读取，**代码里不要硬编码**。
