"""Train TUNA-mini for one (variant, mode) experiment.

Variants (input front-end):
    pixel    — direct patch embedding
    encoder  — frozen DINOv2 features
    vae      — frozen SD VAE latents (requires diffusers)

Modes:
    cls       — classification head only (BCE on 25 multi-label targets)
    cls+diff  — classification head + auxiliary mask-conditioned flow-matching
                head. Both losses are computed in a single forward pass.

Usage:
    python -m models.train_tuna --variant pixel    --mode cls
    python -m models.train_tuna --variant pixel    --mode cls+diff
    python -m models.train_tuna --variant encoder  --mode cls+diff
    python -m models.train_tuna --variant vae      --mode cls+diff

Reports per-epoch val macro AUC for atomic and syndrome label groups,
saves best checkpoint by atomic-AUC, and writes test-set per-label AUC/AP
to <root>/tuna_results_<variant>_<mode>.json.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

from models.dataset import ATOMIC_COLS, MASK_LABELS, SYNDROME_COLS
from models.tuna_mini import TunaMini


def make_tuna_transform(image_size: int = 224):
    """Grayscale [-1, 1], single channel — variant-agnostic."""
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert("L")),
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])


class TunaDataset(Dataset):
    """Same image+labels semantics as RadiographDataset but with the
    grayscale [-1, 1] transform appropriate for TUNA-mini variants.

    When ``load_masks`` is True the per-image soft segmentation NPZs are
    loaded too: ``__getitem__`` returns a third tensor of shape
    ``[len(MASK_LABELS), 320, 320]`` in ``[0, 1]``, with zeros for any
    finding label not present in the per-image NPZ. Used by
    ``mode='cls+mask'`` in ``train_tuna``; otherwise off to save memory.
    """

    def __init__(self, root: str, split: str, image_size: int = 224,
                 load_masks: bool = False):
        self.root = Path(root)
        with (self.root / "splits" / f"{split}.txt").open() as f:
            self.ids = [ln.strip() for ln in f if ln.strip()]
        df = pd.read_csv(self.root / "labels.csv").set_index("image_id").loc[self.ids]
        self.atomic = df[ATOMIC_COLS].values.astype(np.float32)
        self.syndromes = df[SYNDROME_COLS].values.astype(np.float32)
        self.tx = make_tuna_transform(image_size)
        self.load_masks = load_masks

    def __len__(self): return len(self.ids)

    def _load_mask(self, img_id: str) -> torch.Tensor:
        path = self.root / "masks" / f"{img_id}.npz"
        out = np.zeros((len(MASK_LABELS), 320, 320), dtype=np.float32)
        if path.exists():
            with np.load(path) as data:
                for j, label in enumerate(MASK_LABELS):
                    if label in data.files:
                        out[j] = data[label].astype(np.float32) / 255.0
        return torch.from_numpy(out)

    def __getitem__(self, i):
        img_id = self.ids[i]
        img = Image.open(self.root / "images" / f"{img_id}.png")
        labels = np.concatenate([self.atomic[i], self.syndromes[i]])
        x = self.tx(img)
        y = torch.from_numpy(labels)
        if not self.load_masks:
            return x, y
        return x, y, self._load_mask(img_id)


# ---------- metrics (no sklearn) ----------


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = ranks[y == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted); fp = np.cumsum(1.0 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1.0)
    return float((precision * y_sorted).sum() / n_pos)


def macro_metrics(y_true: np.ndarray, scores: np.ndarray, label_names) -> dict:
    aucs, aps, per_label = [], [], {}
    for i, name in enumerate(label_names):
        col = y_true[:, i]
        if 0 < col.sum() < len(col):
            a = roc_auc(col, scores[:, i]); p = average_precision(col, scores[:, i])
            aucs.append(a); aps.append(p)
            per_label[name] = {"auc": a, "ap": p, "prevalence": float(col.mean())}
        else:
            per_label[name] = {"auc": None, "ap": None, "prevalence": float(col.mean())}
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "macro_ap": float(np.mean(aps)) if aps else float("nan"),
        "per_label": per_label,
    }


# ---------- training ----------


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str,
             bf16: bool = True,
             eval_mode: str = "cls") -> tuple[np.ndarray, np.ndarray]:
    """Run the classifier over a loader and return (labels, logits).

    ``eval_mode`` should match how the model was *trained* so the input
    distribution at eval matches what it saw during training:
        - "cls" or "cls+diff" → backbone reads image tokens only.
        - "cls+mask" → backbone reads image + (zero mask noised at t=1) +
          t-embedding (the same conditioning slice the cls+mask trainer
          saw at t=1).
    Without this routing, a cls+mask-trained model evaluated under "cls"
    sees a totally different token distribution and the cls head returns
    garbage.
    """
    model.eval()
    all_logits, all_y = [], []
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                   enabled=bf16 and device == "cuda")
    fwd_mode = "cls+mask" if eval_mode == "cls+mask" else "cls"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(x, mode=fwd_mode)["cls_logits"]
        all_logits.append(logits.float().cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_y), np.concatenate(all_logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output_v3")
    parser.add_argument("--variant", choices=["pixel", "encoder", "vae"], required=True)
    parser.add_argument("--mode", choices=["cls", "cls+diff", "cls+mask"], required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--diff-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                        help="forward + backward in bfloat16 autocast (default on)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds torch + numpy. Same seed should reproduce a run.")
    parser.add_argument("--results-dir", default=None,
                        help="where to write the result JSON. Defaults to --root.")
    parser.add_argument("--out-suffix", default="",
                        help="filename for result JSON (without .json). "
                             "If empty, falls back to tuna_<variant>_<mode>.json.")
    parser.add_argument("--track-test", type=int, default=0,
                        help="Also evaluate the test set every N epochs and "
                              "log it in the per-epoch history (0 = only final).")
    parser.add_argument("--sample-mask-every", type=int, default=0,
                        help="In mode='cls+mask', generate predicted masks "
                              "for a fixed batch of test images every N epochs "
                              "and save them as NPZ snapshots, so a separate "
                              "script can plot mask convergence over training. "
                              "0 = off.")
    parser.add_argument("--sample-mask-n", type=int, default=8,
                        help="number of test images to sample masks for")
    parser.add_argument("--sample-mask-steps", type=int, default=20,
                        help="Euler steps for the mask sampler")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_names = ATOMIC_COLS + SYNDROME_COLS
    n_atomic = len(ATOMIC_COLS)

    print(f"== TUNA-mini ==  variant={args.variant}  mode={args.mode}  "
          f"image={args.image_size}  device={device}", flush=True)

    # Datasets — only train needs the per-image masks (val/test forward
    # uses mode='cls' so masks aren't read).
    needs_masks = args.mode == "cls+mask"
    train_ds = TunaDataset(args.root, "train", image_size=args.image_size,
                            load_masks=needs_masks)
    val_ds   = TunaDataset(args.root, "val",   image_size=args.image_size)
    test_ds  = TunaDataset(args.root, "test",  image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=args.num_workers > 0)

    # Model
    model = TunaMini(
        variant=args.variant, embed_dim=args.embed_dim,
        depth=args.depth, n_heads=args.n_heads,
        n_classes=len(label_names),
        image_size=args.image_size, patch_size=args.patch_size,
    ).to(device)
    n_train_params = model.num_trainable_params()
    print(f"trainable params: {n_train_params/1e6:.1f}M  "
          f"(N={model.n_patches} tokens, token_dim={model.token_dim})", flush=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    bce = nn.BCEWithLogitsLoss()

    n_steps = len(train_loader) * args.epochs
    warmup = max(1, n_steps // 20)

    history = []
    best_val_atomic_auc = -1.0
    best_state = None
    step = 0
    t0 = time.time()
    use_bf16 = args.bf16 and device == "cuda"
    has_aux = args.mode in ("cls+diff", "cls+mask")

    # Set up the fixed batch of test images we'll sample masks for at the
    # snapshot epochs. Loaded once so the same images appear in every
    # snapshot, making the convergence visual readable.
    do_mask_sampling = (args.mode == "cls+mask" and args.sample_mask_every > 0)
    if do_mask_sampling:
        sample_ids = test_ds.ids[: args.sample_mask_n]
        sample_imgs = torch.stack(
            [test_ds.tx(Image.open(Path(args.root) / "images" / f"{i}.png"))
             for i in sample_ids]
        ).to(device)
        # Stash the GT masks once
        sample_gt = torch.stack(
            [TunaDataset(args.root, "test", image_size=args.image_size,
                         load_masks=True)._load_mask(i)
             for i in sample_ids]
        )
        snap_dir = Path(args.results_dir or args.root) / f"{args.out_suffix or 'mask_snaps'}_masks"
        snap_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            snap_dir / "_inputs.npz",
            ids=np.array(sample_ids),
            images=(sample_imgs.cpu().numpy() * 0.5 + 0.5),  # [-1,1] → [0,1]
            gt_masks=sample_gt.cpu().numpy(),
        )
        print(f"mask sampling enabled: {len(sample_ids)} test images, "
              f"every {args.sample_mask_every} epochs → {snap_dir}",
              flush=True)
    for epoch in range(args.epochs):
        model.train()
        run_cls, run_flow, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            if needs_masks:
                x, y, m = batch
                m = m.to(device, non_blocking=True)
            else:
                x, y = batch
                m = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            for g in optim.param_groups:
                g["lr"] = cosine_lr(step, n_steps, args.lr, warmup)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                out = model(x, mode=args.mode, mask_ratio=args.mask_ratio,
                            mask_target=m)
                cls_loss = bce(out["cls_logits"], y)
                loss = cls_loss
                if has_aux:
                    loss = cls_loss + args.diff_weight * out["flow_loss"]
                    run_flow += out["flow_loss"].item()
                run_cls += cls_loss.item()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            n_batches += 1
            step += 1

        y_val, logits_val = evaluate(model, val_loader, device, bf16=use_bf16, eval_mode=args.mode)
        val_atomic = macro_metrics(y_val[:, :n_atomic], logits_val[:, :n_atomic], ATOMIC_COLS)
        val_synd   = macro_metrics(y_val[:, n_atomic:], logits_val[:, n_atomic:], SYNDROME_COLS)

        elapsed = time.time() - t0
        # Optional periodic test eval — for plotting train/val/test curves
        # without waiting for the run to finish.
        epoch_test_atomic = epoch_test_synd = None
        if args.track_test > 0 and ((epoch + 1) % args.track_test == 0 or
                                     (epoch + 1) == args.epochs):
            y_te, logits_te = evaluate(model, test_loader, device, bf16=use_bf16, eval_mode=args.mode)
            epoch_test_atomic = macro_metrics(
                y_te[:, :n_atomic], logits_te[:, :n_atomic], ATOMIC_COLS)
            epoch_test_synd = macro_metrics(
                y_te[:, n_atomic:], logits_te[:, n_atomic:], SYNDROME_COLS)
        line = (f"epoch {epoch+1:>3}/{args.epochs}  "
                f"cls={run_cls/n_batches:.4f}  "
                f"flow={(run_flow/n_batches if has_aux else 0):.4f}  "
                f"val: atomic AUC={val_atomic['macro_auc']:.3f}/{val_atomic['macro_ap']:.3f}  "
                f"syndr  AUC={val_synd['macro_auc']:.3f}/{val_synd['macro_ap']:.3f}  ")
        if epoch_test_atomic is not None:
            line += (f" test atomic AUC={epoch_test_atomic['macro_auc']:.3f}/"
                     f"{epoch_test_atomic['macro_ap']:.3f}  ")
        line += f"lr={optim.param_groups[0]['lr']:.2e}  ({elapsed/60:.1f}min)"
        print(line, flush=True)
        history.append({
            "epoch": epoch + 1,
            "cls_loss": run_cls / n_batches,
            "flow_loss": run_flow / n_batches if has_aux else None,
            "val_atomic": {k: val_atomic[k] for k in ("macro_auc", "macro_ap")},
            "val_syndromes": {k: val_synd[k] for k in ("macro_auc", "macro_ap")},
            "test_atomic": ({k: epoch_test_atomic[k] for k in ("macro_auc", "macro_ap")}
                             if epoch_test_atomic is not None else None),
            "test_syndromes": ({k: epoch_test_synd[k] for k in ("macro_auc", "macro_ap")}
                                if epoch_test_synd is not None else None),
        })
        if val_atomic["macro_auc"] > best_val_atomic_auc:
            best_val_atomic_auc = val_atomic["macro_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Periodic mask snapshot — runs the rectified-flow sampler on the
        # fixed test batch and dumps the predicted masks alongside the GT.
        if do_mask_sampling and ((epoch + 1) % args.sample_mask_every == 0 or
                                  (epoch + 1) == args.epochs):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_bf16):
                pred = model.sample_mask(sample_imgs,
                                          n_steps=args.sample_mask_steps)
            np.savez_compressed(
                snap_dir / f"epoch_{epoch+1:03d}.npz",
                pred=pred.float().cpu().numpy(),
            )

    # ---- final test eval with best checkpoint ----
    model.load_state_dict(best_state)
    y_te, logits_te = evaluate(model, test_loader, device, bf16=use_bf16, eval_mode=args.mode)
    test_atomic = macro_metrics(y_te[:, :n_atomic], logits_te[:, :n_atomic], ATOMIC_COLS)
    test_synd   = macro_metrics(y_te[:, n_atomic:], logits_te[:, n_atomic:], SYNDROME_COLS)

    print("\n== test (best by val atomic AUC) ==")
    print(f"atomic   macro AUC={test_atomic['macro_auc']:.4f}  AP={test_atomic['macro_ap']:.4f}")
    print(f"syndrome macro AUC={test_synd['macro_auc']:.4f}  AP={test_synd['macro_ap']:.4f}")

    # Per-label breakdown
    for grp_name, grp in [("atomic", test_atomic), ("syndromes", test_synd)]:
        print(f"\n  -- {grp_name} --")
        for name, m in grp["per_label"].items():
            if m["auc"] is None:
                print(f"    {name:<35} prev={m['prevalence']:.3f}  (degenerate)")
            else:
                print(f"    {name:<35} prev={m['prevalence']:.3f}  "
                      f"auc={m['auc']:.3f}  ap={m['ap']:.3f}")

    out_dir = Path(args.results_dir) if args.results_dir else Path(args.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_suffix:
        out_path = out_dir / f"{args.out_suffix}.json"
    else:
        mode_tag = args.mode.replace("+", "_")
        out_path = out_dir / f"tuna_{args.variant}_{mode_tag}.json"
    with out_path.open("w") as f:
        json.dump({
            "variant": args.variant,
            "mode": args.mode,
            "image_size": args.image_size,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "n_heads": args.n_heads,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "mask_ratio": args.mask_ratio,
            "diff_weight": args.diff_weight,
            "trainable_params": n_train_params,
            "n_patches": int(model.n_patches),
            "token_dim": int(model.token_dim),
            "history": history,
            "best_val_atomic_macro_auc": best_val_atomic_auc,
            "test_atomic": test_atomic,
            "test_syndromes": test_synd,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
