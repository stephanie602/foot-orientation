#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Bottom-Center 的关键点顺序统一成「脚跟在首位」，与其余 8 个类一致。

背景：该类的骨架模板里 Heel_Center 排在最后（796/808 的样本脚跟在第 2 位），
      其余 8 个类都是脚跟在第 0 位。导出 YOLO-pose 时按槽位顺序写，
      于是 Bottom-Center 的关键点语义与其它类相反。

操作：纯循环移位 [趾A, 趾B, 跟] → [跟, 趾A, 趾B]。
      两个趾尖的相对顺序保持不变，坐标和类别都不改。

判定脚跟：三点中距离最近的一对是两个趾尖，剩下那个是脚跟。
          （正常样本趾-趾间距 0.11，趾-跟间距 0.29，两者相差 2.6 倍）
"""
from __future__ import annotations

import argparse
import csv
import time
from itertools import combinations
from pathlib import Path

import numpy as np
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
TARGET = "Bottom-Center"


def heel_index(pts):
    prs = list(combinations(range(3), 2))
    d = [np.linalg.norm(pts[i] - pts[j]) for i, j in prs]
    i, j = prs[int(np.argmin(d))]
    return ({0, 1, 2} - {i, j}).pop(), i, j


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="normalize_log.csv")
    a = ap.parse_args()

    rf = _rf_client()
    pj = rf.workspace("mago-ai").project("keypoint-categorization-foot-3")

    # 枚举该类全部图片
    ids, off = [], 0
    while True:
        try:
            res = pj.search(limit=200, offset=off, class_name=TARGET, fields=["id", "name"])
        except Exception:
            time.sleep(3); continue
        if not res:
            break
        ids += [(x["id"], x["name"]) for x in res]
        off += len(res)
        if len(res) < 200:
            break
    print(f"{TARGET} 共 {len(ids)} 张")
    if a.limit:
        ids = ids[: a.limit]

    tmp = Path("_norm.txt")
    n_ok = n_skip = n_fail = 0
    log = []
    def with_retry(fn, tries=4):
        last = None
        for t in range(tries):
            try:
                return fn()
            except Exception as e:
                last = e
                time.sleep(2 * (t + 1))
        raise last

    for n, (iid, name) in enumerate(ids, 1):
        try:
            ann = with_retry(lambda: pj.image(iid)).get("annotation") or {}
            boxes = ann.get("boxes") or []
            if not boxes or boxes[0].get("label") != TARGET:
                n_skip += 1; continue
            b = boxes[0]
            W, H = ann["width"], ann["height"]
            kps = sorted(b.get("keypoints", []), key=lambda k: k["id"])
            real = [k for k in kps if not (float(k["x"]) == 0 and float(k["y"]) == 0)]
            if len(real) != 3:
                n_skip += 1
                log.append(dict(image_id=iid, name=name, action="skip",
                                note=f"真实关键点 {len(real)} 个"))
                continue
            pts = [np.array([float(k["x"]), float(k["y"])]) for k in real]
            h, i, j = heel_index(pts)
            if h == 0:
                n_skip += 1
                log.append(dict(image_id=iid, name=name, action="already_ok", note=""))
                continue
            order = [pts[h], pts[i], pts[j]]      # 跟, 趾A, 趾B（趾尖相对顺序不变）
            if a.dry_run:
                n_ok += 1
                if n <= 5:
                    print(f"  {name[:38]}  脚跟 第{h}位 → 第0位")
                continue
            cx, cy = float(b["x"]) / W, float(b["y"]) / H
            bw, bh = float(b["width"]) / W, float(b["height"]) / H
            row = ["0", f"{cx:.10f}", f"{cy:.10f}", f"{bw:.10f}", f"{bh:.10f}"]
            for p in order:
                row += [f"{p[0]/W:.10f}", f"{p[1]/H:.10f}", "2"]
            tmp.write_text(" ".join(row) + "\n")
            with_retry(lambda: pj.save_annotation(
                annotation_path=str(tmp), annotation_labelmap=LABELMAP,
                image_id=iid, annotation_overwrite=True))
            # 读回验证
            a2 = with_retry(lambda: pj.image(iid))["annotation"]["boxes"][0]
            k2 = [k for k in sorted(a2["keypoints"], key=lambda z: z["id"])
                  if not (float(k["x"]) == 0 and float(k["y"]) == 0)]
            p2 = [np.array([float(k["x"]), float(k["y"])]) for k in k2]
            good = (a2["label"] == TARGET and len(k2) == 3 and heel_index(p2)[0] == 0
                    and all(np.linalg.norm(p2[t] - order[t]) < 3 for t in range(3)))
            if good:
                n_ok += 1
                log.append(dict(image_id=iid, name=name, action="fixed", note=""))
            else:
                n_fail += 1
                log.append(dict(image_id=iid, name=name, action="FAIL",
                                note=f"读回 label={a2['label']} kp={len(k2)}"))
                print(f"  ❌ {name[:38]}", flush=True)
        except Exception as e:
            n_fail += 1
            log.append(dict(image_id=iid, name=name, action="ERROR",
                            note=f"{type(e).__name__}: {str(e)[:100]}"))
        if n % 100 == 0:
            print(f"  {n}/{len(ids)}  已修 {n_ok} 跳过 {n_skip} 失败 {n_fail}", flush=True)

    tmp.unlink(missing_ok=True)
    print(f"\n完成: 修正 {n_ok}  跳过 {n_skip}  失败 {n_fail}")
    if log:
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0]))
            w.writeheader(); w.writerows(log)
        print(f"日志 → {a.out}")


if __name__ == "__main__":
    main()
