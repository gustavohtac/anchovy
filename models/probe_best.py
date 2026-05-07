"""Best linear probe — per-label hyperparameter sweep, pure torch, GPU.

For each variant in {cls, dense, both} and each binary target, train a
multi-label nn.Linear head with weighted BCE; sweep over (weight_decay,
class_weight) combinations and select the best combo per-label by val
AUC. All training and metric computation is done in torch/numpy — no
sklearn dependency.

Reads cached features produced by ``models.probe`` (under
``<root>/features/<backbone>/``) — does not re-run the backbone.

Run:
    python -m models.probe_best --root output_v3 --backbone dinov2_vitb14
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.dataset import ATOMIC_COLS, SYNDROME_COLS


WD_GRID = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
CLASS_WEIGHTS = (None, "balanced")
EPOCHS_DEFAULT = 800
LR_DEFAULT = 1e-2


# ---------- Data plumbing ----------


def load_feats(out_dir: str, backbone: str) -> dict:
    cache = Path(out_dir) / "features" / backbone
    feats: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        feats[split] = {
            "cls":       np.load(cache / f"{split}_cls.npy"),
            "dense":     np.load(cache / f"{split}_dense.npy"),
            "atomic":    np.load(cache / f"{split}_atomic.npy"),
            "syndromes": np.load(cache / f"{split}_syndromes.npy"),
        }
    return feats


def make_X(feats: dict, variant: str) -> np.ndarray:
    if variant == "cls":   return feats["cls"]
    if variant == "dense": return feats["dense"]
    if variant == "both":  return np.concatenate([feats["cls"], feats["dense"]], axis=1)
    raise ValueError(variant)


# ---------- Metrics (no sklearn) ----------


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC with tie averaging."""
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    n = len(y)
    n_pos = int(y.sum()); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based mean of contiguous ranks
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = ranks[y == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """sklearn-equivalent step-function integral of PR curve."""
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1.0)
    return float((precision * y_sorted).sum() / n_pos)


# ---------- Probe training ----------


def _standardize(X_tr_np: np.ndarray, X_va_np: np.ndarray, X_te_np: np.ndarray):
    mu = X_tr_np.mean(0, keepdims=True)
    sd = X_tr_np.std(0, keepdims=True)
    sd = np.maximum(sd, 1e-6)
    return ((X_tr_np - mu) / sd).astype(np.float32), \
           ((X_va_np - mu) / sd).astype(np.float32), \
           ((X_te_np - mu) / sd).astype(np.float32)


def train_one_combo(X_tr: torch.Tensor, y_tr: torch.Tensor,
                     X_va: torch.Tensor, X_te: torch.Tensor,
                     weight_decay: float, class_weight: str | None,
                     epochs: int, lr: float, device: str):
    """Train one multi-label nn.Linear head with one (wd, cw) combo on GPU.
    Returns (val_logits_np, test_logits_np)."""
    in_dim = X_tr.shape[1]
    n_labels = y_tr.shape[1]
    head = nn.Linear(in_dim, n_labels).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    if class_weight == "balanced":
        pos = y_tr.sum(dim=0)
        neg = y_tr.shape[0] - pos
        pos_weight = neg / pos.clamp(min=1.0)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        bce = nn.BCEWithLogitsLoss()
    head.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = bce(head(X_tr), y_tr)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        val_logits = head(X_va).float().cpu().numpy()
        test_logits = head(X_te).float().cpu().numpy()
    return val_logits, test_logits


def fit_per_target(X_tr_np, y_tr_np, X_va_np, y_va_np, X_te_np, y_te_np,
                    label_names, device, epochs, lr):
    """Sweep all (wd, cw) combos training all labels jointly; per-label,
    pick the combo with the best val AUC. Returns per-label metrics dict
    plus macro AUC / AP on test."""
    X_tr_n, X_va_n, X_te_n = _standardize(X_tr_np, X_va_np, X_te_np)
    X_tr = torch.from_numpy(X_tr_n).to(device)
    y_tr = torch.from_numpy(y_tr_np).to(device)
    X_va = torch.from_numpy(X_va_n).to(device)
    X_te = torch.from_numpy(X_te_n).to(device)

    combos = [(wd, cw) for wd in WD_GRID for cw in CLASS_WEIGHTS]
    val_per_combo: list[np.ndarray] = []
    test_per_combo: list[np.ndarray] = []
    for k, (wd, cw) in enumerate(combos):
        t0 = time.time()
        v_l, t_l = train_one_combo(X_tr, y_tr, X_va, X_te, wd, cw,
                                    epochs, lr, device)
        val_per_combo.append(v_l)
        test_per_combo.append(t_l)
        print(f"    combo {k+1:>2}/{len(combos)}  wd={wd:<7g}  cw={str(cw):<9}  "
              f"({time.time()-t0:.2f}s)", flush=True)

    out: dict[str, dict] = {}
    aucs, aps = [], []
    for i, name in enumerate(label_names):
        prev_tr = float(y_tr_np[:, i].mean())
        prev_va = float(y_va_np[:, i].mean())
        prev_te = float(y_te_np[:, i].mean())
        # Pick best combo by val AUC (skip combos where val degenerate)
        if y_va_np[:, i].sum() == 0 or y_va_np[:, i].sum() == len(y_va_np):
            best_idx = 0; best_val_auc = float("nan")
        else:
            best_idx, best_val_auc = 0, -1.0
            for c in range(len(combos)):
                auc = roc_auc(y_va_np[:, i], val_per_combo[c][:, i])
                if auc > best_val_auc:
                    best_val_auc, best_idx = auc, c
        test_scores = test_per_combo[best_idx][:, i]
        if 0 < y_te_np[:, i].sum() < len(y_te_np):
            tauc = roc_auc(y_te_np[:, i], test_scores)
            tap = average_precision(y_te_np[:, i], test_scores)
            aucs.append(tauc); aps.append(tap)
        else:
            tauc = tap = None
        wd, cw = combos[best_idx]
        out[name] = {
            "auc": tauc, "ap": tap,
            "val_auc": None if np.isnan(best_val_auc) else float(best_val_auc),
            "best_weight_decay": float(wd),
            "best_class_weight": cw,
            "prev_train": prev_tr,
            "prev_val":   prev_va,
            "prev_test":  prev_te,
        }
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    macro_ap  = float(np.mean(aps))  if aps  else float("nan")
    return out, macro_auc, macro_ap


# ---------- Reporting ----------


def write_markdown(detailed: dict, target_groups, out_path: Path) -> None:
    rows = [
        f"# Linear probe — per-label detailed results",
        "",
        f"Backbone: `{detailed['backbone']}`  ·  "
        f"input: {detailed.get('image_size', 224)}×{detailed.get('image_size', 224)}  ·  "
        f"L2-regularized BCE multi-label logistic regression in torch (GPU); "
        f"per-label sweep over weight_decay × class_weight, model selected on val AUC.",
        "",
        "Sweep grid:",
        f"- `weight_decay` ∈ {{ {', '.join(str(w) for w in WD_GRID)} }}",
        f"- `class_weight` ∈ {{ {', '.join(str(c) for c in CLASS_WEIGHTS)} }}",
        f"- epochs = {detailed.get('epochs', EPOCHS_DEFAULT)}, lr = {detailed.get('lr', LR_DEFAULT)}, "
        f"optimizer = AdamW, full-batch.",
        "",
    ]
    for tgt_name, tgt_cols in target_groups:
        rows.append(f"## {tgt_name}")
        rows.append("")
        rows.append(
            "| label | prev (test) | "
            "cls AUC | cls AP | dense AUC | dense AP | both AUC | both AP |"
        )
        rows.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for label in tgt_cols:
            cls_m = detailed["variants"]["cls"][tgt_name]["per_label"][label]
            r = [f"`{label}`", f"{cls_m['prev_test']:.3f}"]
            for v in ("cls", "dense", "both"):
                m = detailed["variants"][v][tgt_name]["per_label"][label]
                r.append("—" if m["auc"] is None else f"{m['auc']:.3f}")
                r.append("—" if m["ap"]  is None else f"{m['ap']:.3f}")
            rows.append("| " + " | ".join(r) + " |")
        # Macro row
        r = ["**macro**", "—"]
        for v in ("cls", "dense", "both"):
            d = detailed["variants"][v][tgt_name]
            r.append(f"**{d['macro_auc']:.3f}**")
            r.append(f"**{d['macro_ap']:.3f}**")
        rows.append("| " + " | ".join(r) + " |")
        rows.append("")

        # Hyperparameter table per label (which combo each variant picked)
        rows.append(f"### Selected hyperparameters ({tgt_name})")
        rows.append("")
        rows.append("| label | cls (wd, cw) | dense (wd, cw) | both (wd, cw) |")
        rows.append("|---|---|---|---|")
        for label in tgt_cols:
            r = [f"`{label}`"]
            for v in ("cls", "dense", "both"):
                m = detailed["variants"][v][tgt_name]["per_label"][label]
                wd = m["best_weight_decay"]; cw = m["best_class_weight"]
                r.append(f"{wd:g}, {cw}")
            rows.append("| " + " | ".join(r) + " |")
        rows.append("")
    out_path.write_text("\n".join(rows))


# ---------- Orchestration ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output_v3")
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--image-size", type=int, default=224,
                        help="recorded in the report (does not re-extract)")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--out-suffix", default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, backbone={args.backbone}, "
          f"epochs={args.epochs}, lr={args.lr}", flush=True)

    feats = load_feats(args.root, args.backbone)
    target_groups = [("atomic", ATOMIC_COLS), ("syndromes", SYNDROME_COLS)]

    detailed: dict = {
        "backbone": args.backbone,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay_grid": list(WD_GRID),
        "class_weights": [str(c) for c in CLASS_WEIGHTS],
        "variants": {},
    }

    for variant in ("cls", "dense", "both"):
        X_tr = make_X(feats["train"], variant)
        X_va = make_X(feats["val"], variant)
        X_te = make_X(feats["test"], variant)
        d_variant = {"feature_dim": int(X_tr.shape[1])}
        for tgt_name, tgt_cols in target_groups:
            print(f"\n=== {variant} / {tgt_name} (dim={X_tr.shape[1]}) ===", flush=True)
            t0 = time.time()
            per_label, mauc, map_ = fit_per_target(
                X_tr, feats["train"][tgt_name],
                X_va, feats["val"][tgt_name],
                X_te, feats["test"][tgt_name],
                tgt_cols, device, args.epochs, args.lr,
            )
            d_variant[tgt_name] = {
                "macro_auc": mauc, "macro_ap": map_, "per_label": per_label,
            }
            print(f"  -> macro AUC={mauc:.3f}   macro AP={map_:.3f}   "
                  f"({time.time()-t0:.1f}s)", flush=True)
            for name in tgt_cols:
                m = per_label[name]
                auc_s = "—" if m["auc"] is None else f"{m['auc']:.3f}"
                ap_s = "—" if m["ap"] is None else f"{m['ap']:.3f}"
                print(f"    {name:<35} prev_te={m['prev_test']:.3f}  "
                      f"auc={auc_s}  ap={ap_s}  "
                      f"wd={m['best_weight_decay']:g}  cw={m['best_class_weight']}")
        detailed["variants"][variant] = d_variant

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    json_path = Path(args.root) / f"probe_best_{args.backbone}{suffix}.json"
    md_path   = Path(args.root) / f"probe_best_{args.backbone}{suffix}.md"
    with json_path.open("w") as f:
        json.dump(detailed, f, indent=2)
    write_markdown(detailed, target_groups, md_path)

    print(f"\nWrote: {json_path}\n       {md_path}")
    print("\n" + "=" * 60)
    print(f"{'variant/target':<22} {'dim':>5} {'macro AUC':>10} {'macro AP':>10}")
    print("-" * 60)
    for v in ("cls", "dense", "both"):
        for tgt in ("atomic", "syndromes"):
            d = detailed["variants"][v][tgt]
            print(f"{v + '/' + tgt:<22} {detailed['variants'][v]['feature_dim']:>5} "
                  f"{d['macro_auc']:>10.3f} {d['macro_ap']:>10.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
