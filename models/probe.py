"""Linear probe on DINOv2 features for the v3 dataset.

Pipeline (per backbone):
  1. Forward all images once through a frozen DINOv2 ViT, capture both
     the CLS token and the mean-pooled patch tokens. Cache both as .npy
     under <out>/features/<backbone>/.
  2. For each (variant ∈ {cls, dense, both}) × (target ∈ {atomic, syndromes}),
     train a single nn.Linear head with BCEWithLogitsLoss for a fixed number
     of epochs, selecting the epoch with the best macro-AUC on the val
     split. Report per-label AUC, AP, and prevalence on the test split.

The feature cache makes re-running probes (e.g. with different LR or
weight decay) effectively free after the first extraction pass.

Run:
    python -m models.probe --root output_v3 --backbone dinov2_vitb14
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.dataset import (ATOMIC_COLS, SYNDROME_COLS, RadiographDataset)


# ---------- Feature extraction ----------


@torch.no_grad()
def _extract_split(model, root, split, image_size, batch_size, device):
    ds = RadiographDataset(root, split, image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    cls_list, dense_list = [], []
    for imgs, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model.forward_features(imgs)
        cls_list.append(out["x_norm_clstoken"].float().cpu().numpy())
        dense_list.append(out["x_norm_patchtokens"].mean(dim=1).float().cpu().numpy())
    cls = np.concatenate(cls_list, axis=0)
    dense = np.concatenate(dense_list, axis=0)
    return cls, dense, ds.atomic, ds.syndromes


def get_or_extract(args, model, device):
    cache_dir = Path(args.out) / "features" / args.backbone
    cache_dir.mkdir(parents=True, exist_ok=True)
    feats: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        paths = {
            "cls":       cache_dir / f"{split}_cls.npy",
            "dense":     cache_dir / f"{split}_dense.npy",
            "atomic":    cache_dir / f"{split}_atomic.npy",
            "syndromes": cache_dir / f"{split}_syndromes.npy",
        }
        if all(p.exists() for p in paths.values()):
            feats[split] = {k: np.load(p) for k, p in paths.items()}
            print(f"[cache] {split}: cls={feats[split]['cls'].shape} "
                  f"dense={feats[split]['dense'].shape}", flush=True)
        else:
            print(f"[extract] {split}…", flush=True)
            t0 = time.time()
            cls, dense, ya, ys = _extract_split(
                model, args.root, split, args.image_size, args.batch_size, device,
            )
            np.save(paths["cls"], cls)
            np.save(paths["dense"], dense)
            np.save(paths["atomic"], ya)
            np.save(paths["syndromes"], ys)
            feats[split] = {"cls": cls, "dense": dense,
                            "atomic": ya, "syndromes": ys}
            print(f"[extract] {split} done: cls={cls.shape} dense={dense.shape} "
                  f"in {time.time()-t0:.1f}s", flush=True)
    return feats


# ---------- Linear probe ----------


def _make_X(feats: dict, variant: str) -> np.ndarray:
    if variant == "cls":   return feats["cls"]
    if variant == "dense": return feats["dense"]
    if variant == "both":  return np.concatenate([feats["cls"], feats["dense"]], axis=1)
    raise ValueError(variant)


def _macro_auc(y_true: np.ndarray, logits: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    aucs = []
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if col.sum() == 0 or col.sum() == len(col): continue
        aucs.append(roc_auc_score(col, logits[:, i]))
    return float(np.mean(aucs)) if aucs else 0.5


def _per_label_metrics(y_true, logits, label_names):
    from sklearn.metrics import roc_auc_score, average_precision_score
    out: dict[str, dict] = {}
    aucs, aps = [], []
    probs = 1.0 / (1.0 + np.exp(-logits))
    for i, name in enumerate(label_names):
        col = y_true[:, i]
        prev = float(col.mean())
        if col.sum() == 0 or col.sum() == len(col):
            out[name] = {"auc": None, "ap": None, "prevalence": prev}
        else:
            auc = float(roc_auc_score(col, probs[:, i]))
            ap = float(average_precision_score(col, probs[:, i]))
            out[name] = {"auc": auc, "ap": ap, "prevalence": prev}
            aucs.append(auc); aps.append(ap)
    return out, float(np.mean(aucs)), float(np.mean(aps))


def train_probe(X_train, y_train, X_val, y_val, X_test,
                 device, epochs=200, lr=1e-2, weight_decay=1e-4):
    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv_np = y_val
    Xs = torch.from_numpy(X_test).to(device)

    head = nn.Linear(Xt.shape[1], yt.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    best_val_auc, best_state = -1.0, None
    for epoch in range(epochs):
        head.train()
        opt.zero_grad()
        loss = bce(head(Xt), yt)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(Xv).cpu().numpy()
        val_auc = _macro_auc(yv_np, val_logits)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_logits = head(Xs).cpu().numpy()
    return test_logits, best_val_auc


# ---------- Orchestration ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output_v3",
                        help="dataset root (must contain images/, labels.csv, splits/)")
    parser.add_argument("--out", default="output_v3",
                        help="where to write features/ and probe_results.json")
    parser.add_argument("--backbone", default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14",
                                  "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, backbone={args.backbone}, image_size={args.image_size}",
          flush=True)

    # Load DINOv2 backbone (frozen)
    model = torch.hub.load("facebookresearch/dinov2", args.backbone)
    model = model.to(device).eval()

    # 1. Cache features
    feats = get_or_extract(args, model, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Probes for every (variant × target) pair
    target_groups = {"atomic": ATOMIC_COLS, "syndromes": SYNDROME_COLS}
    results = {
        "backbone": args.backbone,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "experiments": {},
    }

    for variant in ("cls", "dense", "both"):
        for tgt_name, tgt_cols in target_groups.items():
            tag = f"{variant}__{tgt_name}"
            print(f"\n=== {tag} ===", flush=True)
            t0 = time.time()
            X_tr = _make_X(feats["train"], variant)
            X_va = _make_X(feats["val"], variant)
            X_te = _make_X(feats["test"], variant)
            y_tr = feats["train"][tgt_name]
            y_va = feats["val"][tgt_name]
            y_te = feats["test"][tgt_name]
            test_logits, best_val = train_probe(
                X_tr, y_tr, X_va, y_va, X_te, device,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            )
            per_label, mauc, map_ = _per_label_metrics(y_te, test_logits, tgt_cols)
            print(f"  feature_dim={X_tr.shape[1]}  "
                  f"best_val_macro_auc={best_val:.3f}  "
                  f"test macro_auc={mauc:.3f} macro_ap={map_:.3f}  "
                  f"({time.time()-t0:.1f}s)",
                  flush=True)
            for name, m in per_label.items():
                if m["auc"] is None:
                    print(f"    {name:<35} prev={m['prevalence']:.3f}  (degenerate)")
                else:
                    print(f"    {name:<35} prev={m['prevalence']:.3f}  "
                          f"auc={m['auc']:.3f}  ap={m['ap']:.3f}")
            results["experiments"][tag] = {
                "feature_dim": int(X_tr.shape[1]),
                "best_val_macro_auc": float(best_val),
                "test_macro_auc": float(mauc),
                "test_macro_ap": float(map_),
                "per_label": per_label,
            }

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'experiment':<25} {'dim':>5} {'val_auc':>8} {'test_auc':>9} {'test_ap':>8}")
    print("-" * 60)
    for tag, r in results["experiments"].items():
        print(f"{tag:<25} {r['feature_dim']:>5} {r['best_val_macro_auc']:>8.3f} "
              f"{r['test_macro_auc']:>9.3f} {r['test_macro_ap']:>8.3f}")
    print("=" * 60)

    out_path = Path(args.out) / f"probe_results_{args.backbone}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
