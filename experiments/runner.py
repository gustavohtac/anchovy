"""Pod-side worker: claim experiments from the shared queue and run them.

Each pod runs::

    python -m experiments.runner --root /workspace/pixel-vs-prior

which loops:

    1. atomically claim the oldest pending experiment by renaming
       ``queue/<id>.json`` to ``running/<id>.json``;
    2. invoke :mod:`models.train_tuna` as a subprocess with the claimed
       hyperparameters, tee'ing stdout/stderr to ``logs/<id>.log`` and
       writing the final metrics to ``results/<id>.json``;
    3. on exit, atomically move the running file to ``done/`` (rc==0)
       or ``failed/`` (otherwise);
    4. repeat until the queue is empty.

Concurrency safety relies on POSIX ``os.rename`` being atomic on the
shared filesystem (MooseFS satisfies this).  Multiple pods racing on the
same queue will see exactly one winner per file; losers fall through to
the next candidate.
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .manifest import state_dirs


def claim_one(root: Path,
               allow_variants: set[str] | None = None) -> tuple[Path, dict] | tuple[None, None]:
    """Try to claim the oldest pending experiment whose dataset is ready
    and whose variant is in ``allow_variants`` (or ``None`` = all).

    Tasks whose dataset isn't materialized yet are silently skipped so
    pods can drain runnable work while a separate worker is still
    generating a larger dataset.  Tasks whose variant is filtered out by
    the local pod's ``--allow-variants`` are likewise skipped — this is
    used to keep ``vae`` tasks on the pods whose torch/diffusers install
    can actually load the SD-VAE.

    Returns (None, None) when no runnable task is present *for this pod*.
    """
    dirs = state_dirs(root)
    if not dirs["queue"].exists():
        return None, None
    candidates = sorted(dirs["queue"].glob("*.json"))
    for src in candidates:
        try:
            peek = json.loads(src.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if allow_variants is not None and peek["variant"] not in allow_variants:
            continue
        data_root = root / "datasets" / f"n{peek['data_size']}"
        if not (data_root / "labels.csv").exists():
            continue  # dataset not yet ready; leave for later
        dst = dirs["running"] / src.name
        try:
            os.rename(src, dst)
        except FileNotFoundError:
            continue  # another pod claimed it first
        data = json.loads(dst.read_text())
        data["_claimed_by"] = f"{socket.gethostname()}:{os.getpid()}"
        data["_claimed_at"] = time.time()
        dst.write_text(json.dumps(data, indent=2) + "\n")
        return dst, data
    return None, None


def finish(running_path: Path, success: bool) -> Path:
    final = running_path.parent.parent / ("done" if success else "failed") / running_path.name
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(running_path, final)
    return final


def run_train(exp: dict, repo_root: Path) -> int:
    """Spawn ``models.train_tuna`` for one claimed experiment.

    Returns the subprocess exit code.  Logs are appended to
    ``experiments/logs/<id>.log``; the result JSON lands in
    ``experiments/results/<id>.json``.
    """
    eid = exp["id"]
    data_root = repo_root / "datasets" / f"n{exp['data_size']}"
    if not (data_root / "labels.csv").exists():
        print(f"  ERROR: dataset not ready at {data_root}", flush=True)
        return 2

    log_path = repo_root / "experiments" / "logs" / f"{eid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_root / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "models.train_tuna",
        "--root",         str(data_root),
        "--results-dir",  str(results_dir),
        "--variant",      exp["variant"],
        "--mode",         exp["mode"],
        "--epochs",       str(exp["epochs"]),
        "--batch-size",   str(exp["batch_size"]),
        "--lr",           str(exp["lr"]),
        "--weight-decay", str(exp["weight_decay"]),
        "--embed-dim",    str(exp["embed_dim"]),
        "--depth",        str(exp["depth"]),
        "--n-heads",      str(exp["n_heads"]),
        "--image-size",   str(exp["image_size"]),
        "--patch-size",   str(exp["patch_size"]),
        "--mask-ratio",   str(exp["mask_ratio"]),
        "--diff-weight",  str(exp["diff_weight"]),
        "--seed",         str(exp["seed"]),
        "--out-suffix",   eid,
        # Always track per-epoch test metrics so plot_training_curves.py
        # can render the train/val/test trajectory of any run.  No-op
        # cost is one extra forward over the test split per epoch.
        "--track-test",   str(exp.get("track_test", 1)),
        # And in cls+mask mode, snapshot the predicted masks every K
        # epochs so plot_mask_convergence.py can show how the rectified-
        # flow sampler's output sharpens as training progresses. Off by
        # default for cls / cls+diff (which have no mask head to sample).
        "--sample-mask-every", str(
            exp.get("sample_mask_every",
                     5 if exp["mode"] == "cls+mask" else 0)),
    ]
    with log_path.open("w") as f:
        f.write("# " + " ".join(cmd) + "\n\n")
    with log_path.open("a") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=str(repo_root))
    return proc.returncode


def loop(root: Path, max_runs: int | None = None,
         poll_seconds: float = 0.0,
         allow_variants: set[str] | None = None) -> None:
    """Drain the queue. If ``poll_seconds`` > 0, wait that long when no
    runnable task is present *for this pod* and re-check (useful when
    other pods are slowly enqueuing more work, or when a dataset is
    still being generated). 0 means exit immediately on empty."""
    if allow_variants:
        print(f"variant filter: only claiming {sorted(allow_variants)}", flush=True)
    completed = 0
    while max_runs is None or completed < max_runs:
        path, data = claim_one(root, allow_variants=allow_variants)
        if path is None:
            if poll_seconds > 0:
                time.sleep(poll_seconds); continue
            print("queue empty — exiting", flush=True)
            return
        eid = data["id"]
        print(f"[claim] {eid}", flush=True)
        t0 = time.time()
        rc = run_train(data, root)
        ok = (rc == 0)
        finish(path, ok)
        elapsed = time.time() - t0
        print(f"[{('done' if ok else 'failed')}] {eid}  "
              f"rc={rc}  {elapsed/60:.1f}min", flush=True)
        completed += 1


def main():
    parser = argparse.ArgumentParser(description="Run claimed experiments in a loop.")
    parser.add_argument("--root", default="/workspace/pixel-vs-prior",
                        help="repo root containing datasets/ and experiments/")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="exit after this many runs (default: drain queue)")
    parser.add_argument("--poll-seconds", type=float, default=0.0,
                        help="seconds to wait when queue is empty (0=exit)")
    parser.add_argument("--allow-variants", default="all",
                        help="comma-separated list of variants this pod will "
                              "claim (e.g. 'pixel,encoder' to skip 'vae'). "
                              "'all' for no filter.")
    args = parser.parse_args()
    if args.allow_variants == "all":
        allow = None
    else:
        allow = {v.strip() for v in args.allow_variants.split(",") if v.strip()}
    loop(Path(args.root), max_runs=args.max_runs,
         poll_seconds=args.poll_seconds, allow_variants=allow)


if __name__ == "__main__":
    main()
