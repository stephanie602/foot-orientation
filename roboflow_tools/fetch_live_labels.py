#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Roboflow 实时读取标注，写成 YOLO-pose 标签，用于验证本地重建是否准确。

版本快照（v18/v19）是冻结的，不含本次会话的改动；实时标注才是当前状态。
"""
from __future__ import annotations

import argparse
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
CID = {c: i for i, c in enumerate(CLASSES9)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--src", default="v19", help="用于取文件名清单")
    ap.add_argument("--map", default="rf_name2id.json")
    ap.add_argument("--dst", default="v19_live")
    a = ap.parse_args()

    idx = json.load(open(a.map))
    out = Path(a.dst) / a.split / "labels"
    out.mkdir(parents=True, exist_ok=True)

    rf = _rf_client()
    pj = rf.workspace("mago-ai").project("keypoint-categorization-foot-3")

    lbls = sorted((Path(a.src) / a.split / "labels").glob("*.txt"))
    n_ok = n_miss = n_err = 0
    for n, lbl in enumerate(lbls, 1):
        name = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", lbl.stem) + ".jpg"
        iid = idx.get(name)
        if not iid:
            n_miss += 1
            continue
        ann = None
        for t in range(4):
            try:
                ann = pj.image(iid).get("annotation")
                break
            except Exception:
                time.sleep(2 * (t + 1))
        if not ann or not ann.get("boxes"):
            n_err += 1
            continue
        W, H = ann["width"], ann["height"]
        rows = []
        for b in ann["boxes"]:
            lab = b.get("label")
            if lab not in CID:
                continue
            cx, cy = float(b["x"]) / W, float(b["y"]) / H
            bw, bh = float(b["width"]) / W, float(b["height"]) / H
            kp = [k for k in sorted(b.get("keypoints", []), key=lambda z: z["id"])
                  if not (float(k["x"]) == 0 and float(k["y"]) == 0)]
            if len(kp) != 3:
                continue
            r = [str(CID[lab]), f"{cx:.10f}", f"{cy:.10f}", f"{bw:.10f}", f"{bh:.10f}"]
            for k in kp:
                r += [f"{float(k['x'])/W:.10f}", f"{float(k['y'])/H:.10f}", "2"]
            rows.append(" ".join(r))
        if rows:
            (out / lbl.name).write_text("\n".join(rows) + "\n")
            n_ok += 1
        if n % 200 == 0:
            print(f"  {n}/{len(lbls)}  写出 {n_ok}", flush=True)

    print(f"\n完成: 写出 {n_ok}  无 id {n_miss}  读取失败 {n_err}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
