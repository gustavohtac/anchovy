# `models/` — architectures, training, linear probes

## Three things live here

1. **[`tuna_mini.py`](tuna_mini.py)** — a reduced reproduction of the [TUNA-2](https://github.com/facebookresearch/tuna-2) architectural family. One transformer backbone, three swappable input front-ends:
    - `pixel`   — direct patch embedding of the raw grayscale image (TUNA-2)
    - `encoder` — frozen DINOv2 ViT-S/14 features projected into the backbone (TUNA-R)
    - `vae`     — frozen Stable Diffusion VAE latents patchified (Tuna)
   The backbone has two heads:
    - `cls_head`  : multi-label classification (15 atomic + 10 syndrome targets = 25)
    - `diff_head` : per-token rectified-flow velocity prediction at masked positions
2. **[`train_tuna.py`](train_tuna.py)** — single-experiment training loop. Modes:
    - `cls`      : classification only (BCEWithLogitsLoss).
    - `cls+diff` : both heads in a single forward pass — 50 % of patches get noised in their native space (pixel patches / DINOv2 features / VAE latents) and the diff head predicts the velocity `ε − x_clean` only on those positions, while the cls head reads the CLS token. Loss is `BCE + diff_weight · flow_loss`.
   Both modes use AdamW with a cosine LR schedule and bf16 autocast on CUDA.
3. **[`probe.py`](probe.py) and [`probe_best.py`](probe_best.py)** — linear-probe baselines on cached DINOv2 features. `probe.py` trains one multi-label linear head per variant (cls / dense / both); `probe_best.py` does a per-label hyperparameter sweep over `weight_decay × class_weight` to push each label to its strongest linear ceiling. Both write a per-label table of AUC/AP. Pure torch — no sklearn dependency.

## How the dual-head training works

Given an input image, `forward(x, mode="cls+diff")` does the following in **one pass**:

```
native      = encode(x)                       # variant-specific tokenization, no grad in encoder/vae
mask        = random subset of N patches      # 50 % by default
t           = sample U(0, 1)                  # one noise level per sample
xt          = (1 - t) · native  +  t · noise  # rectified flow at masked positions
native_in   = where(mask, xt, native)         # unmasked positions stay clean
tokens      = TokenProj(native_in)
tokens     += t-embedding · 1[masked]         # so the backbone knows the noise level
seq         = [CLS] + tokens + pos_embed
seq         = Transformer(seq)
cls_logits  = cls_head(seq[CLS])              #  -> BCE loss (multi-label)
v_pred      = diff_head(seq[1:])              #  -> MSE on (noise − native), only on masked positions
```

The auxiliary objective conditions the backbone to recover the input distribution from partial context — a self-supervised loss that, in principle, regularizes the shared representations toward more useful features. The `cls` vs `cls+diff` ablation (each variant trained both ways) tests whether that helps classification convergence in the small-data regime.

## Linear-probe baselines

Computed on cached DINOv2-ViT-B/14 features at 224 × 224 (val/test of `datasets/n10000/`):

| variant | target group | macro AUC | macro AP |
|---|---|---:|---:|
| `cls` (CLS token, 768 d) | atomic | 0.975 | 0.908 |
| | syndromes | 0.976 | 0.896 |
| `dense` (avg patch tokens, 768 d) | atomic | 0.981 | 0.924 |
| | syndromes | 0.975 | 0.902 |
| `both` (cls ⊕ dense, 1536 d) | atomic | **0.982** | **0.936** |
| | syndromes | 0.973 | 0.901 |

Per-label breakdowns are written to `docs/results/` (gitignored — local only) by `models.probe_best` and the run aggregator; the snapshot tables are inlined in the top-level `README.md`.

The hardest labels for a frozen-feature linear probe — `solitary_pulmonary_nodule` (AUC 0.81 / AP 0.38), `honeycombing` (0.88 / 0.68), `pleural_thickening` (0.96 / 0.74) — are exactly the ones that depend on *small, focal, spatially-localized* features that get diluted by the 320 → 224 resize and by global pooling. Whether end-to-end training on enough samples can recover them is the question the scaling sweep is designed to answer.

## CLIs

Train one experiment manually:

```bash
python -m models.train_tuna \
    --root /workspace/pixel-vs-prior/datasets/n10000 \
    --variant pixel --mode cls+diff \
    --epochs 30 --batch-size 256 \
    --results-dir /workspace/pixel-vs-prior/experiments/results \
    --out-suffix n0010000_pixel_cls_diff_s0
```

Run a linear probe manually (must have features cached by `models.probe`):

```bash
python -m models.probe       --root datasets/n10000 --backbone dinov2_vitb14   # extracts features once
python -m models.probe_best  --root datasets/n10000 --backbone dinov2_vitb14   # per-label sweep
```

Most of the time the orchestration in [`experiments/`](../experiments/) is what invokes `train_tuna.py`; running manually is for debugging.
