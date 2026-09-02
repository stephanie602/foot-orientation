#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把涉及 Center-Lateral / Center-Medial 的框还原为原始人工标注。

范围：只要一个框的「原标注」或「模型新标注」是 CL/CM，就恢复成原始人工标注。
      结果是这两个类完全回到人工判断，模型不再对内外侧做任何修改。

依据：模型在横轴（内/外侧）上与人工标注一致率 0.94，没有增量信息；
      纵轴（上/中/下）一致率仅 0.70–0.82，改动才有价值。

关键点顺序保持当前状态（脚跟在首位），不回退 —— 那是独立的规范化成果。
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
CID = {c: i for i, c in enumerate(CLASSES9)}
LABELMAP = {str(i): c for i, c in enumerate(CLASSES9)}
CLCM = {"Center-Lateral", "Center-Medial"}


def kp_order(k):
    prs = list(combinations(range(3), 2))
    d = [np.linalg.norm(k[i][:2] - k[j][:2]) for i, j in prs]
    i, j = prs[int(np.argmin(d))]
    h = ({0, 1, 2} - {i, j}).pop()
    return [k[h], k[i], k[j]]


def parse(line):
    p = line.split()
    if len(p) < 14:
        return None
    return (int(p[0]), [float(x) for x in p[1:5]],
            [np.array([float(p[5 + 3 * i]), float(p[6 + 3 * i]), float(p[7 + 3 * i])])
             for i in range(3)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="v19", help="原始人工标注所在目录")
    ap.add_argument("--changes", required=True, help="relabel_<split>.csv")
    ap.add_argument("--split", required=True)
    ap.add_argument("--map", default="rf_name2id.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import json
    import re
    idx = json.load(open(a.map))
    rows = [r for r in csv.DictReader(open(a.changes, encoding="utf-8-sig"))
            if r["old"] in CLCM or r["new"] in CLCM]
    print(f"{a.split}: 需还原 {len(rows)} 个框")
    for r in rows[:6]:
        print(f"   {r['old']:16s} → {r['new']:16s}  还原回 {r['old']}")
    if a.dry_run:
        print("\n(dry-run，未写入)")
        return

    rf = _rf_client()
    pj = rf.workspace("mago-ai").project("keypoint-categorization-foot-3")
    tmp = Path("_revert.txt")

    def retry(fn, tries=4):
        last = None
        for t in range(tries):
            try:
                return fn()
            except Exception as e:
                last = e
                time.sleep(2 * (t + 1))
        raise last

    ok = fail = 0
    log = []
    for n, r in enumerate(rows, 1):
        stem = r["file"].replace(".txt", "")
        name = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", stem) + ".jpg"
        iid = idx.get(name)
        if not iid:
            fail += 1
            log.append(dict(file=r["file"], image_id="", target=r["old"],
                            ok=0, note="找不到 image_id"))
            continue
        # 原始人工标注（v19 导出，未经模型改写）
        src_lbl = Path(a.src) / a.split / "labels" / r["file"]
        got = parse(src_lbl.read_text().splitlines()[int(r["line"])])
        if not got:
            fail += 1
            continue
        cls, box, k = got
        target = CLASSES9[cls]
        if target != r["old"]:
            fail += 1
            log.append(dict(file=r["file"], image_id=iid, target=target, ok=0,
                            note=f"原始标注 {target} 与记录的 {r['old']} 不符"))
            continue
        k = kp_order(k)                      # 关键点顺序保持规范化状态
        out = [str(cls)] + [f"{v:.17g}" for v in box]
        for p in k:
            out += [f"{p[0]:.17g}", f"{p[1]:.17g}", f"{p[2]:.17g}"]
        tmp.write_text(" ".join(out) + "\n")
        try:
            retry(lambda: pj.save_annotation(annotation_path=str(tmp),
                                             annotation_labelmap=LABELMAP,
                                             image_id=iid, annotation_overwrite=True))
            b = retry(lambda: pj.image(iid))["annotation"]["boxes"][0]
            good = b["label"] == target
            ok, fail = (ok + 1, fail) if good else (ok, fail + 1)
            log.append(dict(file=r["file"], image_id=iid, target=target,
                            ok=int(good), note="" if good else f"读回 {b['label']}"))
            if not good:
                print(f"  ❌ {name[:38]} 读回 {b['label']}", flush=True)
        except Exception as e:
            fail += 1
            log.append(dict(file=r["file"], image_id=iid, target=target, ok=0,
                            note=f"{type(e).__name__}: {str(e)[:90]}"))
        if n % 20 == 0:
            print(f"  {n}/{len(rows)}  成功 {ok} 失败 {fail}", flush=True)

    tmp.unlink(missing_ok=True)
    print(f"\n完成: 还原 {ok}  失败 {fail}")
    outp = a.out or f"revert_{a.split}_log.csv"
    if log:
        with open(outp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0]))
            w.writeheader(); w.writerows(log)
        print(f"日志 → {outp}")


if __name__ == "__main__":
    main()
