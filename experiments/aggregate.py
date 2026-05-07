"""Collect per-experiment result JSONs into a single CSV/Markdown table.

Run from any pod (or locally after ``rsync``-ing results back)::

    python -m experiments.aggregate --root /workspace/pixel-vs-prior

Writes ``experiments/results.csv`` and ``experiments/results.md``.  The
markdown view groups by data regime so the scaling pattern is easy to
read at a glance.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .config import all_experiments


def collect(root: Path) -> list[dict]:
    results_dir = root / "experiments" / "results"
    rows: list[dict] = []
    for exp in all_experiments():
        path = results_dir / f"{exp.id}.json"
        base = {
            "id":          exp.id,
            "data_size":   exp.data_size,
            "variant":     exp.variant,
            "mode":        exp.mode,
            "seed":        exp.seed,
            "epochs":      exp.epochs,
            "batch_size":  exp.batch_size,
            "embed_dim":   exp.embed_dim,
            "depth":       exp.depth,
        }
        if not path.exists():
            base.update(status="missing", atomic_auc=None, atomic_ap=None,
                         syndrome_auc=None, syndrome_ap=None,
                         trainable_params=None)
        else:
            data = json.loads(path.read_text())
            base.update(
                status="done",
                atomic_auc=   data["test_atomic"]["macro_auc"],
                atomic_ap=    data["test_atomic"]["macro_ap"],
                syndrome_auc= data["test_syndromes"]["macro_auc"],
                syndrome_ap=  data["test_syndromes"]["macro_ap"],
                trainable_params=data.get("trainable_params"),
            )
        rows.append(base)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows: return
    fields = list(rows[0].keys())
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(rows: list[dict], path: Path) -> None:
    """Group by data regime; one table per regime with variant × mode rows."""
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_size[r["data_size"]].append(r)
    lines = [
        "# TUNA-mini scaling experiments — results",
        "",
        "Each cell is the macro mean (AUC / AP) on the held-out test set.",
        "All results trained on the same backbone (depth=12, dim=768, ~87 M params), "
        "same val/test split, single seed unless noted.",
        "",
    ]
    for n in sorted(by_size):
        lines.append(f"## n = {n:,} training samples")
        lines.append("")
        lines.append("| variant | mode | atomic AUC | atomic AP | syndrome AUC | syndrome AP | seed |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in sorted(by_size[n], key=lambda r: (r["variant"], r["mode"], r["seed"])):
            def fmt(x): return "—" if x is None else f"{x:.3f}"
            lines.append(
                f"| `{r['variant']}` | `{r['mode']}` | {fmt(r['atomic_auc'])} "
                f"| {fmt(r['atomic_ap'])} | {fmt(r['syndrome_auc'])} "
                f"| {fmt(r['syndrome_ap'])} | {r['seed']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/pixel-vs-prior")
    args = parser.parse_args()
    root = Path(args.root)
    rows = collect(root)
    out_dir = root / "experiments"
    out_dir.mkdir(exist_ok=True)
    write_csv(rows, out_dir / "results.csv")
    write_markdown(rows, out_dir / "results.md")
    n_done = sum(1 for r in rows if r["status"] == "done")
    print(f"{n_done}/{len(rows)} experiments complete")
    print(f"Wrote {out_dir / 'results.csv'} and {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
