#!/usr/bin/env bash
# Push the local repo (code only — no datasets, no caches, no results) to a
# pod's /workspace/pixel-vs-prior.  /workspace is shared across pods in the same
# RunPod region, so syncing once from any pod (or laptop) is enough for
# all of them.
#
# Usage:
#     ./scripts/sync_to_pod.sh USER@HOST -p PORT [-i identity]
#
# Example:
#     ./scripts/sync_to_pod.sh root@213.192.2.101 -p 40085 -i ~/.ssh/id_ed25519
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 user@host [ssh-options...]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$1"; shift
SSH_FLAGS=("$@")

rsync -av --delete-excluded \
    --exclude '__pycache__' \
    --exclude '.DS_Store' \
    --exclude '.git' \
    --exclude '.claude' \
    --exclude 'datasets/' \
    --exclude 'output_v3/' \
    --exclude 'experiments/queue/' \
    --exclude 'experiments/running/' \
    --exclude 'experiments/done/' \
    --exclude 'experiments/failed/' \
    --exclude 'experiments/logs/' \
    --exclude 'experiments/results/' \
    --exclude '*.log' \
    --exclude '*.pyc' \
    -e "ssh ${SSH_FLAGS[*]}" \
    "$REPO_ROOT/data" \
    "$REPO_ROOT/models" \
    "$REPO_ROOT/experiments" \
    "$REPO_ROOT/scripts" \
    "$REPO_ROOT/README.md" \
    "$REPO_ROOT/pyproject.toml" \
    "$DEST:/workspace/pixel-vs-prior/"

echo "Synced to $DEST:/workspace/pixel-vs-prior/"
