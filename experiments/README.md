# Experiments — scaling sweep across TUNA-mini variants

## Question

Do pixel-patch front-ends (TUNA-2 style) overtake encoder- and VAE-based front-ends as the training set grows? And does a self-supervised mask-conditioned diffusion auxiliary objective (`cls+diff` vs `cls`) accelerate classification convergence in the small-data regime?

## Sweep

The full cross-product, defined declaratively in [`config.py`](config.py):

| axis | values |
|---|---|
| data regime (training samples) | `1_000`, `10_000`, `100_000`, `600_000` |
| variant (input front-end)      | `pixel`, `encoder`, `vae` |
| mode                           | `cls`, `cls+diff` |
| seed                           | `0` (extend to `(0, 1, 2)` for 3-rep variance) |

That is 4 × 3 × 2 × 1 = **24 experiments** at single-seed; 72 at three seeds. Adding a regime or a seed only requires editing `config.py` and re-running `python -m experiments.manifest init` — the queue picks up just the new ids.

Held constant across the sweep:

- backbone (depth=12, dim=768, 12 heads, ~87 M trainable params)
- val/test split (deterministic from `data/generate.py --split-seed 0`)
- optimizer (AdamW, cosine schedule, weight decay 0.05, grad-clip 1.0)
- input pipeline (224×224 grayscale, `[-1, 1]`)
- bf16 autocast

Per-regime epochs are sublinear in `N`: `{1k: 50, 10k: 30, 100k: 10, 600k: 4}` so total samples seen stays within an order of magnitude across regimes.

## Queue mechanics

The queue is a directory state machine on the shared `/workspace` filesystem (MooseFS, mounted on every pod in the same RunPod region):

```
experiments/
    queue/    # pending tasks
    running/  # currently claimed by some pod
    done/     # completed (rc == 0)
    failed/   # exited non-zero
    logs/     # per-experiment training logs
    results/  # per-experiment final JSON
```

State transitions happen via `os.rename`, which is atomic on POSIX and on MooseFS — multiple pods racing for the same task file see exactly one winner. Losers fall through to the next candidate.

## Running across multiple pods

```bash
# 1. From the laptop, sync code to each pod (one-time per code change):
for pod in $(grep -v '^#' pods.txt); do
    ./scripts/sync_to_pod.sh $pod
done

# 2. On each pod (one-time setup), install Python deps:
ssh ... "bash /workspace/pixel-vs-prior/scripts/pod_setup.sh"

# 3. On any one pod, generate the datasets you want to sweep
#    (downstream pods reuse the same files via the shared volume):
ssh ... "bash /workspace/pixel-vs-prior/scripts/prep_data.sh 1000 10000 100000"

# 4. On any one pod, initialize the queue (idempotent):
ssh ... "cd /workspace/pixel-vs-prior && python3 -m experiments.manifest init"

# 5. On EACH pod, kick off the worker loop in the background:
ssh ... "nohup bash /workspace/pixel-vs-prior/scripts/pod_loop.sh \
            > /workspace/pixel-vs-prior/pod_\$(hostname).log 2>&1 &"

# 6. Check progress at any time:
ssh ... "cd /workspace/pixel-vs-prior && python3 -m experiments.manifest status"

# 7. Once enough experiments are done, aggregate the results:
ssh ... "cd /workspace/pixel-vs-prior && python3 -m experiments.aggregate"
```

Pods finish in any order. If a pod dies mid-run, its task stays under `running/` — `python -m experiments.manifest status` lists it; move it back to `queue/` manually if a re-attempt is desired.

## Identifiers

Each experiment id encodes every dimension that affects the result:

```
n0010000_pixel_cls_diff_s0
└─size─┘ └var┘ └─mode──┘└─seed
```

so the result file `experiments/results/n0010000_pixel_cls_diff_s0.json` is self-describing. Changing any field in [`config.py`](config.py) yields a new id; old results are preserved.

## Reading the results

`python -m experiments.aggregate` writes `results.csv` (machine-readable) and `results.md` (one table per data regime, variant × mode rows, AUC and AP for atomic findings and clinical syndromes). The markdown is the primary view for spotting scaling trends across regimes.
