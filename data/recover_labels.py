"""Recover ``labels.csv`` + splits from a dataset directory whose images
were generated but whose per-shard records JSONL files were never written
(e.g. ``data.generate --shard-only`` was killed mid-run by an OS error).

Generation is fully deterministic given a seed, so the *labels* and
*scenario* for each image can be recomputed from the same seed without
re-rendering the (slow) image and mask.  Only ``sample_scenario`` and
``derive_labels`` are run here — both are millisecond-level — so this
script processes hundreds of thousands of images in a few minutes.

Outputs ``labels.csv``, ``finding_definitions.json``, and
``splits/{train,val,test}.txt``. Annotations and reports are *not*
recovered (they require re-rendering); the linear/end-to-end classifier
training pipelines do not need them.

Usage:
    python -m data.recover_labels --out datasets/n750000 --seed 42
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from data.findings import ALL_LABELS, derive_labels
from data.generate import write_definitions, write_splits
from data.scenarios import sample_scenario


_NAME_RE = re.compile(r"^img_(\d+)$")


def _record_for(args):
    idx, seed, image_id = args
    rng = np.random.default_rng(seed)
    scenario, state = sample_scenario(rng)
    labels = derive_labels(state)
    return {"image_id": image_id, "scenario": scenario, "labels": labels}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True,
                        help="dataset directory (must contain images/)")
    parser.add_argument("--seed", type=int, default=42,
                        help="base seed used during generation; per-image "
                              "seed is base + idx (matches data.generate)")
    parser.add_argument("--workers", type=int,
                        default=max(1, (32)))
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio",   type=float, default=0.10)
    parser.add_argument("--split-seed",  type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    if not images_dir.exists():
        print(f"images/ not found at {images_dir}", file=sys.stderr)
        return 1

    work: list[tuple[int, int, str]] = []
    for p in images_dir.iterdir():
        if p.suffix != ".png":
            continue
        m = _NAME_RE.match(p.stem)
        if not m:
            continue
        idx = int(m.group(1))
        work.append((idx, args.seed + idx, p.stem))
    work.sort()
    if not work:
        print(f"no img_<digits>.png files under {images_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(work)} images. Re-deriving labels with {args.workers} workers…")

    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_record_for, w) for w in work]
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if i % max(1, len(work) // 20) == 0:
                print(f"  {i}/{len(work)}")
    records.sort(key=lambda r: r["image_id"])

    print(f"Writing labels.csv ({len(records)} rows)")
    header = ["image_id", "scenario"] + ALL_LABELS
    with (out_dir / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in records:
            row = [r["image_id"], r["scenario"]] + [
                int(r["labels"].get(k, 0)) for k in ALL_LABELS
            ]
            w.writerow(row)

    write_definitions(out_dir)
    test_ratio = max(0.0, 1.0 - args.train_ratio - args.val_ratio)
    sizes = write_splits(records, out_dir,
                         (args.train_ratio, args.val_ratio, test_ratio),
                         args.split_seed)
    print(f"Splits: {sizes}")
    print(f"OK — labels.csv + splits/ recovered for {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
