"""Initialize and inspect the experiment queue on a shared filesystem.

The queue is a directory-based state machine on disk:

    <root>/experiments/
        queue/    - pending tasks (one .json per experiment)
        running/  - currently claimed by some pod
        done/     - completed successfully
        failed/   - exited non-zero

State transitions happen via :func:`os.rename`, which is atomic on POSIX
(and on the MooseFS-mounted ``/workspace`` shared between RunPod
instances).  Atomic moves provide concurrency-safe claim semantics so
multiple pods can drain the same queue without coordination.

CLI:
    python -m experiments.manifest init     # create dirs, enqueue missing experiments
    python -m experiments.manifest status   # show all tasks per state
    python -m experiments.manifest reset    # move failed -> queue (re-attempt)
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import all_experiments


STATES = ("queue", "running", "done", "failed")


def state_dirs(root: Path) -> dict[str, Path]:
    return {s: root / "experiments" / s for s in STATES}


def init(root: Path, dry_run: bool = False) -> dict:
    """Create directories and enqueue any experiments not already tracked.

    Idempotent — re-running after editing :mod:`experiments.config` adds
    only the newly-introduced experiments.
    """
    dirs = state_dirs(root)
    if not dry_run:
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for d in dirs.values():
        if d.exists():
            seen.update(p.stem for p in d.glob("*.json"))
    added = []
    for exp in all_experiments():
        if exp.id in seen:
            continue
        added.append(exp.id)
        if dry_run:
            continue
        (dirs["queue"] / f"{exp.id}.json").write_text(
            json.dumps({"id": exp.id, **asdict(exp)}, indent=2) + "\n"
        )
    counts = {s: len(list(dirs[s].glob("*.json"))) for s in STATES if dirs[s].exists()}
    return {"added": added, "counts": counts}


def status(root: Path) -> dict[str, list[str]]:
    return {
        s: sorted(p.stem for p in d.glob("*.json")) if d.exists() else []
        for s, d in state_dirs(root).items()
    }


def reset_failed(root: Path) -> int:
    dirs = state_dirs(root)
    n = 0
    for p in dirs["failed"].glob("*.json"):
        target = dirs["queue"] / p.name
        target.write_bytes(p.read_bytes())
        p.unlink()
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Manage the experiment queue.")
    parser.add_argument("cmd", choices=["init", "status", "reset"])
    parser.add_argument("--root", default="/workspace/pixel-vs-prior",
                        help="repo root on the (shared) filesystem")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)

    if args.cmd == "init":
        r = init(root, dry_run=args.dry_run)
        verb = "Would add" if args.dry_run else "Added"
        print(f"{verb} {len(r['added'])} new experiments to queue.")
        for eid in r["added"][:30]:
            print(f"  + {eid}")
        if len(r["added"]) > 30:
            print(f"  ... and {len(r['added']) - 30} more")
        if r["counts"]:
            print("\nQueue state:")
            for s, n in r["counts"].items():
                print(f"  {s:<8} {n}")

    elif args.cmd == "status":
        s = status(root)
        for state, ids in s.items():
            print(f"\n[{state}]  ({len(ids)})")
            for eid in ids:
                print(f"  {eid}")

    elif args.cmd == "reset":
        n = reset_failed(root)
        print(f"Moved {n} failed -> queue")


if __name__ == "__main__":
    main()
