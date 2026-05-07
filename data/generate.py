"""CLI for the v3 dataset (clinical taxonomy: atomic + computed +
syndrome labels). Writes:

  output_v3/
    images/img_00001.png ...
    masks/img_00001.npz             — soft segmentation masks per finding
                                       (uint8 0-255, named by finding label)
    labels.csv                      — image_id, scenario, atomic+computed+syndrome cols
    annotations.json                — per-image list of finding metadata,
                                       each entry references its mask key
    reports.json                    — per-image free-text radiology report
                                       (training target) + legacy fields
    finding_definitions.json        — taxonomy + mask format documentation
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from data.findings import ALL_LABELS, LABEL_GROUPS, derive_labels
from data.render import render_state
from data.reports import compose_report
from data.scenarios import sample_scenario


def _save_masks(path: Path, masks: dict[str, np.ndarray]) -> None:
    """Persist soft masks for one image as a compressed NPZ of uint8 arrays
    (values 0-255, divide by 255 on load to recover [0,1] floats)."""
    payload = {label: (np.clip(m, 0.0, 1.0) * 255.0).astype(np.uint8)
               for label, m in masks.items()}
    np.savez_compressed(path, **payload)


def _generate_one(args):
    idx, seed, out_dir, width = args
    image_id = f"img_{idx:0{width}d}"
    rng = np.random.default_rng(seed)
    scenario, state = sample_scenario(rng)
    img, annotations, masks = render_state(state, rng)
    report = compose_report(state, rng)
    img_path = Path(out_dir) / "images" / f"{image_id}.png"
    Image.fromarray(img, mode="L").save(img_path, optimize=True)
    mask_path = Path(out_dir) / "masks" / f"{image_id}.npz"
    _save_masks(mask_path, masks)
    labels = derive_labels(state)
    return {
        "image_id": image_id,
        "scenario": scenario,
        "labels": labels,
        "annotations": annotations,
        "report": report,
        "mask_keys": sorted(masks.keys()),
    }


def write_definitions(out_dir):
    payload = {
        "label_groups": LABEL_GROUPS,
        "all_labels": ALL_LABELS,
        "syndrome_definitions": {
            "chf_syndrome": "cardiomegaly>=2 AND bilateral_effusion AND (kerley_b OR pulm_venous_redistribution)",
            "uip_pattern": "honeycombing AND reticular_pattern AND lower-zone bias",
            "sarcoidosis_pattern": "hilar_adenopathy AND reticular_pattern (upper) AND >=3 small nodules",
            "tb_pattern": "any cavitary nodule AND apical/upper bias",
            "mesothelioma_pattern": "ipsilateral effusion + pleural thickening + volume loss",
            "tension_pneumothorax": "pneumothorax + contralateral volume loss",
            "metastatic_pattern": ">=3 nodules with size variability >=4 px",
            "miliary_pattern": ">=80 micronodules diffusely",
            "solitary_pulmonary_nodule": "exactly 1 nodule sized 4-30 px (Fleischner-relevant)",
            "lobar_pneumonia_pattern": "consolidation>=lobar AND air bronchogram",
        },
        "size_to_mm_estimate": "1 px ≈ 1.5 mm in the 320×320 image (heuristic only — synthetic data)",
        "masks_format": {
            "container": "one .npz per image at masks/<image_id>.npz",
            "dtype": "uint8 in [0, 255]; load with (arr.astype('float32') / 255.0)",
            "shape": "(320, 320) per finding, soft (gaussian-edged) segmentation",
            "keys": "named by finding label — e.g. 'cardiomegaly', 'effusion_left', "
                    "'consolidation', 'nodules' (multiple instances merged), "
                    "'micronodules' (cluster), etc. Only present labels are stored.",
            "semantics": "per-finding contribution to the canvas, max-blended across "
                          "instances of the same type, then normalized to [0,1] by the "
                          "per-finding peak intensity (so faint findings still reach 1.0 "
                          "at their core).",
        },
        "report_format": {
            "container": "reports.json — {image_id: {text, findings, impression}}",
            "text": "Free-text report imitating real radiology phrasing — primary "
                     "training target. Mixes structured (TECHNIQUE / CLINICAL "
                     "INDICATION / COMPARISON / FINDINGS / IMPRESSION) and prose styles, "
                     "with anatomical sub-headers in some samples.",
            "findings": "Plain prose body of just the FINDINGS sentences (legacy).",
            "impression": "Plain prose IMPRESSION line(s) (legacy).",
        },
    }
    with (Path(out_dir) / "finding_definitions.json").open("w") as f:
        json.dump(payload, f, indent=2)


def _to_jsonable(o):
    """Recursively coerce numpy scalars / arrays / tuples into JSON-safe types."""
    import numpy as np
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o


def write_splits(records, out_dir: Path, ratios: tuple[float, float, float],
                  seed: int) -> dict[str, int]:
    """Stratified split by scenario. Writes splits/{train,val,test}.txt with
    one image_id per line. Returns split-size dict."""
    train_r, val_r, _ = ratios
    by_scenario: dict[str, list[str]] = {}
    for r in records:
        by_scenario.setdefault(r["scenario"], []).append(r["image_id"])
    rng = np.random.default_rng(seed)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for scenario, ids in by_scenario.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train:n_train + n_val])
        splits["test"].extend(ids[n_train + n_val:])
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for name, ids in splits.items():
        ids.sort()
        with (splits_dir / f"{name}.txt").open("w") as f:
            f.write("\n".join(ids) + ("\n" if ids else ""))
    return {k: len(v) for k, v in splits.items()}


def aggregate(records, out_dir):
    records = sorted(records, key=lambda r: r["image_id"])
    header = ["image_id", "scenario"] + ALL_LABELS
    with (Path(out_dir) / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in records:
            row = [r["image_id"], r["scenario"]] + [
                int(r["labels"].get(k, 0)) for k in ALL_LABELS
            ]
            w.writerow(row)
    with (Path(out_dir) / "annotations.json").open("w") as f:
        json.dump(_to_jsonable({
            r["image_id"]: {"findings": r["annotations"],
                              "mask_keys": r["mask_keys"]}
            for r in records
        }), f, indent=1)
    with (Path(out_dir) / "reports.json").open("w") as f:
        json.dump({r["image_id"]: r["report"] for r in records}, f, indent=1)


def validate(out_dir):
    import pandas as pd
    df = pd.read_csv(Path(out_dir) / "labels.csv")
    print(f"\nLoaded {len(df)} samples.")
    print("\n=== Scenario distribution ===")
    print(df["scenario"].value_counts())
    print("\n=== Atomic finding prevalences ===")
    for c in LABEL_GROUPS["atomic"]:
        if df[c].dtype.kind in "iu" and df[c].max() <= 1:
            print(f"  {c:<40} {df[c].mean():.3f}")
    print("\n=== Computed labels (binary) ===")
    for c in LABEL_GROUPS["computed"]:
        if df[c].dtype.kind in "iu" and df[c].max() <= 1:
            print(f"  {c:<40} {df[c].mean():.3f}")
    print("\n=== Syndrome prevalences ===")
    for c in LABEL_GROUPS["syndromes"]:
        print(f"  {c:<40} {df[c].mean():.3f}")
    print("\n=== Selected size/count distributions ===")
    print("  nodule_count distribution:")
    print("   ", df["nodule_count"].value_counts().sort_index().head(15).to_dict())
    print("  micronodule_count quantiles:",
          df["micronodule_count"].quantile([0.1, 0.5, 0.9, 0.99]).to_dict())

    masks_dir = Path(out_dir) / "masks"
    if masks_dir.exists():
        sample = next(masks_dir.glob("*.npz"), None)
        if sample is not None:
            with np.load(sample) as data:
                print(f"\n=== Sample mask file: {sample.name} ===")
                for k in data.files:
                    arr = data[k]
                    print(f"  {k:<32} shape={arr.shape} dtype={arr.dtype} "
                          f"min={arr.min()} max={arr.max()} "
                          f"coverage={(arr > 25).mean():.3f}")

    reports_path = Path(out_dir) / "reports.json"
    if reports_path.exists():
        with reports_path.open() as f:
            reports = json.load(f)
        sample_id = next(iter(reports))
        print(f"\n=== Sample report: {sample_id} ===")
        print(reports[sample_id]["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="output_v3")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--validate-only", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=1,
                        help="first 1-based index produced by this run "
                              "(use with --shard-only for multi-pod gen)")
    parser.add_argument("--total-n", type=int, default=None,
                        help="full dataset size — controls image-id zero-padding "
                              "so different shards produce identical width ids. "
                              "Defaults to --n.")
    parser.add_argument("--shard-only", action="store_true",
                        help="Produce only images, masks, and a per-shard "
                              "_shards/records_<start>_<end>.jsonl. Skip "
                              "labels.csv / annotations.json / reports.json / "
                              "splits — to be assembled by data.merge_shards.")
    args = parser.parse_args()

    if args.validate_only:
        validate(args.validate_only)
        return 0

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    if args.shard_only:
        (out_dir / "_shards").mkdir(parents=True, exist_ok=True)
    write_definitions(out_dir)

    total_n = args.total_n if args.total_n is not None else args.n
    max_idx = total_n  # 1-based, so the largest index is total_n
    width = max(5, len(str(max_idx)))
    end_idx = args.start_idx + args.n - 1
    print(f"Generating {args.n} v3 samples [idx {args.start_idx}..{end_idx}] "
          f"(width={width}) into {out_dir}"
          f"{' (shard-only)' if args.shard_only else ''}")
    work = [(args.start_idx + i, args.seed + (args.start_idx + i),
             str(out_dir), width) for i in range(args.n)]
    records = []
    if args.workers <= 1:
        for w in work:
            records.append(_generate_one(w))
            if len(records) % max(1, args.n // 20) == 0:
                print(f"  {len(records)} / {args.n}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_generate_one, w) for w in work]
            for i, fut in enumerate(as_completed(futures), 1):
                records.append(fut.result())
                if i % max(1, args.n // 20) == 0:
                    print(f"  {i} / {args.n}")
    if args.shard_only:
        records.sort(key=lambda r: r["image_id"])
        records_path = (out_dir / "_shards"
                        / f"records_{args.start_idx:08d}_{end_idx:08d}.jsonl")
        with records_path.open("w") as f:
            for r in records:
                f.write(json.dumps(_to_jsonable({
                    "image_id":    r["image_id"],
                    "scenario":    r["scenario"],
                    "labels":      r["labels"],
                    "annotations": r["annotations"],
                    "report":      r["report"],
                    "mask_keys":   r["mask_keys"],
                })) + "\n")
        print(f"Wrote shard records to {records_path}  ({len(records)} records)")
        return 0

    print("Aggregating outputs...")
    aggregate(records, out_dir)
    test_ratio = max(0.0, 1.0 - args.train_ratio - args.val_ratio)
    print(f"Writing stratified splits (train/val/test = "
          f"{args.train_ratio:.2f}/{args.val_ratio:.2f}/{test_ratio:.2f})...")
    sizes = write_splits(records, out_dir,
                         (args.train_ratio, args.val_ratio, test_ratio),
                         args.split_seed)
    print(f"  splits: {sizes}")
    print("Validating...")
    validate(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
