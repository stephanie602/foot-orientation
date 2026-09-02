#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把重刷后的 test 标签推送到 Roboflow，逐张验证关键点未受损。

已知问题（推送前必读）：
  Roboflow 项目里 Bottom-Center 这个类的关键点绑定在 id 2/3/4，
  而其它 8 个类是 id 0/1/2。因此任何写成 Bottom-Center 的标注，
  读回时会显示 5 个关键点（前两个是 (0,0) 空槽）。
  这是该类既有的 schema 缺陷 —— 现存 808 张 Bottom-Center 图同样如此，
  不是本次推送引入的。修复只能在 Roboflow 网页端的 keypoint 设置里做。

因此验证逻辑对 Bottom-Center 放宽：只要真实的 3 个点坐标正确即可。

用法：
    python push_roboflow.py --labels <重刷后的labels目录> --map rf_name2id.json \
        --dry-run                 # 先看会改什么，不写
    python push_roboflow.py --labels ... --map ... --limit 1    # 单张试写
    python push_roboflow.py --labels ... --map ...              # 全量
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from roboflow import Roboflow

def _rf_client():
    """从环境变量取 Roboflow key，绝不硬编码。

        export ROBOFLOW_API_KEY=xxxx
    """
    import os
    k = os.environ.get("ROBOFLOW_API_KEY")
    if not k:
        raise SystemExit("请先设置环境变量 ROBOFLOW_API_KEY（Roboflow 后台 → Settings → API Keys）")
    return Roboflow(api_key=k)


CLASSES9 = ["Bottom-Center", "Bottom-Lateral", "Bottom-Medial", "Center-Center",
            "Center-Lateral", "Center-Medial", "Top-Center", "Top-Lateral", "Top-Medial"]
LABELMAP = {str(i): c for i, c in enumerate(CLASSES9)}
OFFSET_CLASSES = {"Bottom-Center"}      # 关键点 id 从 2 开始的类


def orig_name(export_name: str) -> str:
    return re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", export_name) + ".jpg"


def read_label(p: Path):
    rows = []
    for line in p.read_text().splitlines():
        f = line.split()
        if len(f) >= 14:
            rows.append((int(f[0]), [float(x) for x in f[1:5]],
                         [(float(f[5 + 3 * i]), float(f[6 + 3 * i])) for i in range(3)]))
    return rows


def verify(ann, expect_cls: str, expect_kp, W, H, tol=3.0):
    """返回 (ok, 说明)。坐标以像素比较，容差 tol 像素。"""
    boxes = (ann or {}).get("boxes") or []
    if not boxes:
        return False, "读回没有 boxes"
    b = boxes[0]
    if b.get("label") != expect_cls:
        return False, f"类别不符: {b.get('label')} != {expect_cls}"
    kps = sorted(b.get("keypoints", []), key=lambda k: k["id"])
    real = [k for k in kps if not (float(k["x"]) == 0 and float(k["y"]) == 0)]
    if len(real) != 3:
        return False, f"真实关键点 {len(real)} 个（应为 3）"
    for (ex, ey), k in zip(expect_kp, real):
        if abs(float(k["x"]) - ex * W) > tol or abs(float(k["y"]) - ey * H) > tol:
            return False, (f"关键点偏差过大: 期望({ex*W:.0f},{ey*H:.0f}) "
                           f"实际({float(k['x']):.0f},{float(k['y']):.0f})")
    phantom = len(kps) - 3
    if phantom and expect_cls not in OFFSET_CLASSES:
        return False, f"出现 {phantom} 个空槽，而该类不应有"
    return True, (f"OK（{expect_cls} 固有 {phantom} 个空槽）" if phantom else "OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--map", required=True, help="name→image_id 的 json")
    ap.add_argument("--changes", default="", help="relabel_test.csv；只推有改动的")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="push_log.csv")
    a = ap.parse_args()

    idx = json.load(open(a.map))
    lbl_dir = Path(a.labels)

    targets = sorted(lbl_dir.glob("*.txt"))
    if a.changes:
        want = {r["file"] for r in csv.DictReader(open(a.changes, encoding="utf-8-sig"))}
        targets = [p for p in targets if p.name in want]
        print(f"只推送有类别改动的 {len(targets)} 张")
    if a.limit:
        targets = targets[: a.limit]

    missing = [p.name for p in targets if orig_name(p.stem) not in idx]
    if missing:
        print(f"⚠️  {len(missing)} 张找不到 image_id，将跳过: {missing[:3]}")
        targets = [p for p in targets if orig_name(p.stem) in idx]
    print(f"待推送 {len(targets)} 张\n")

    if a.dry_run:
        for p in targets[:10]:
            rows = read_label(p)
            print(f"  {p.name[:40]}  → {CLASSES9[rows[0][0]]}")
        print("\n(dry-run，未写入)")
        return

    rf = _rf_client()
    pj = rf.workspace("mago-ai").project("keypoint-categorization-foot-3")

    ok = fail = 0
    log = []
    for i, p in enumerate(targets, 1):
        name = orig_name(p.stem)
        iid = idx[name]
        rows = read_label(p)
        if not rows:
            continue
        cls, box, kp = rows[0]
        expect = CLASSES9[cls]
        try:
            pj.save_annotation(annotation_path=str(p), annotation_labelmap=LABELMAP,
                               image_id=iid, annotation_overwrite=True)
            ann = pj.image(iid).get("annotation")
            W, H = ann.get("width"), ann.get("height")
            good, msg = verify(ann, expect, kp, W, H)
        except Exception as e:
            good, msg = False, f"{type(e).__name__}: {str(e)[:120]}"
        ok, fail = (ok + 1, fail) if good else (ok, fail + 1)
        log.append(dict(file=p.name, image_id=iid, label=expect,
                        ok=int(good), note=msg))
        if not good:
            print(f"  ❌ {name}: {msg}", flush=True)
        if i % 50 == 0:
            print(f"  {i}/{len(targets)}  成功 {ok} 失败 {fail}", flush=True)
        time.sleep(0.05)

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader(); w.writerows(log)
    print(f"\n完成: 成功 {ok}  失败 {fail}")
    print(f"日志 → {a.out}")
    if fail:
        print("⚠️  存在失败项，请查看日志后再决定是否重试")


if __name__ == "__main__":
    main()
