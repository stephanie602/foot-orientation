#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 YOLO-pose 数据集切成 ImageFolder 分类版式。

    foot_classification/test/images/xxx.jpg  +  labels/xxx.txt
        ↓
    foot_classification_cls/test/Top-Center/xxx_0.jpg

裁剪配方与 pipeline.py / eval_*.py 一致：bbox 外扩 pad → 裁到图内 → 灰边补正方形。
文件名后缀 _0 _1 … 是该图内的框序号（本数据集每图恰好 1 个框）。

用法：
    python make_cls_dataset.py --dry-run                    # 只统计和估算体积
    python make_cls_dataset.py --splits test                # 先切一个 split
    python make_cls_dataset.py                              # 全切
    python make_cls_dataset.py --classes 5 --max-size 448   # 只要模型的5类、缩到448省空间
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from foot_geom import (CLASSES9, CLASSES5, C5, N_FEAT, IMAGE_SIZE, CROP_PAD,
                       parse_label_line, order_keypoints, geom, crop_square, scores)


PAD_COLOR = (114, 114, 114)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="EfficientNetV2/foot_classification",
                   help="YOLO-pose 数据集根目录（下含 train/valid/test 的 images+labels）")
    p.add_argument("--out", default="foot_classification_cls", help="输出根目录")
    p.add_argument("--splits", default="train,valid,test")
    p.add_argument("--classes", choices=["9", "5"], default="9",
                   help="9=全部类别；5=只保留 best_model.pth 认识的那 5 类")
    p.add_argument("--pad", type=float, default=0.2, help="bbox 外扩比例")
    p.add_argument("--max-size", type=int, default=0,
                   help="裁剪图最长边上限，0=不缩放。设 448 可大幅省磁盘")
    p.add_argument("--quality", type=int, default=95, help="JPEG 质量")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    p.add_argument("--dry-run", action="store_true", help="不写文件，只统计并估算体积")
    p.add_argument("--manifest", default="cls_manifest.csv")
    return p.parse_args()


def read_label(path: Path) -> list[tuple[int, list[float]]]:
    """读出 (类别id, [cx,cy,bw,bh])。

    只要求 5 个字段——裁剪不需要关键点。返回顺序即文件行序，裁剪图名里的 `_i`
    就是这个下标，manifest 的 box 列也是它，三者必须一致。
    """
    rows = []
    for l in path.read_text().splitlines():
        f = l.split()
        if len(f) >= 5:
            rows.append((int(f[0]), [float(x) for x in f[1:5]]))
    return rows


def main() -> None:
    a = parse_args()
    src = Path(a.src)
    out = Path(a.out)
    keep = set(CLASSES9 if a.classes == "9" else CLASSES5)

    rows = []
    n_written = n_skip_cls = n_skip_exist = n_bad_img = n_bad_crop = n_clipped = 0
    bytes_out = 0

    for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
        ldir, idir = src / split / "labels", src / split / "images"
        if not ldir.is_dir():
            print(f"⚠️  跳过 {split}：找不到 {ldir}")
            continue
        labels = sorted(ldir.glob("*.txt"))
        print(f"\n[{split}] {len(labels)} 个标签文件", flush=True)

        for n, lb in enumerate(labels, 1):
            boxes = read_label(lb)
            if not boxes:
                continue
            img = None
            for i, (cls, box) in enumerate(boxes):
                name = CLASSES9[cls] if 0 <= cls < len(CLASSES9) else None
                if name is None or name not in keep:
                    n_skip_cls += 1
                    continue
                dst = out / split / name / f"{lb.stem}_{i}.jpg"
                if dst.exists() and not a.overwrite:
                    n_skip_exist += 1
                    continue
                if img is None:
                    ip = idir / f"{lb.stem}.jpg"
                    img = cv2.imread(str(ip))
                    if img is None:
                        n_bad_img += 1
                        break
                c, clipped = crop_square(img, box, a.pad)
                if c is None:
                    n_bad_crop += 1
                    continue
                if a.max_size and max(c.shape[:2]) > a.max_size:
                    c = cv2.resize(c, (a.max_size, a.max_size), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", c, [int(cv2.IMWRITE_JPEG_QUALITY), a.quality])
                if not ok:
                    n_bad_crop += 1
                    continue
                bytes_out += len(buf)
                n_clipped += int(clipped)
                if not a.dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(buf.tobytes())
                rows.append(dict(split=split, cls=name, box=i, clipped=int(clipped),
                                 src=str(idir / f"{lb.stem}.jpg"), dst=str(dst)))
                n_written += 1
            if n % 2000 == 0:
                print(f"  {n}/{len(labels)}  已产出 {n_written}", flush=True)

    print(f"\n{'（dry-run，未写盘）' if a.dry_run else '完成'}")
    print(f"  产出裁剪图      {n_written}")
    print(f"  体积            {bytes_out/1e9:.2f} GB")
    print(f"  类别不在范围内  {n_skip_cls}")
    print(f"  已存在跳过      {n_skip_exist}")
    print(f"  图像读取失败    {n_bad_img}")
    print(f"  裁剪失败        {n_bad_crop}")
    print(f"  触到图像边界    {n_clipped}（这些图的灰边不对称，属正常）")

    free = shutil.disk_usage(Path.cwd()).free
    print(f"  当前可用磁盘    {free/1e9:.2f} GB")
    if bytes_out > free * 0.9:
        print("  ⚠️  空间可能不够，建议加 --max-size 448")

    if rows:
        from collections import Counter
        print("\n每个 split 的类别分布:")
        for split in sorted({r["split"] for r in rows}):
            cnt = Counter(r["cls"] for r in rows if r["split"] == split)
            print(f"  [{split}] 共 {sum(cnt.values())}")
            for c in CLASSES9:
                if cnt[c]:
                    print(f"      {c:16s} {cnt[c]:5d}")
        if not a.dry_run:
            with open(a.manifest, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"\n清单 → {a.manifest}")
            print(f"输出 → {out}/<split>/<类别>/<图名>_<框号>.jpg")


if __name__ == "__main__":
    main()
