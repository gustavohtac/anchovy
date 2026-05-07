#!/usr/bin/env bash
# Launch sharded generation of the n=1,000,000 dataset across all pods in
# pods.txt.  Each pod generates a contiguous slice of indices and writes
# its records to a per-shard JSONL.  When all shards finish, run::
#
#     ssh ... "cd /workspace/pixel-vs-prior && python3 -m data.merge_shards \
#         --out /workspace/pixel-vs-prior/datasets/n1000000"
#
# to assemble labels.csv / annotations.json / reports.json / splits.
#
# Run from the laptop:
#     bash scripts/launch_1m_gen.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PODS_FILE="$REPO_ROOT/pods.txt"
TOTAL=1000000
WORKERS_PER_POD=16
OUT="/workspace/pixel-vs-prior/datasets/n${TOTAL}"

# Read non-comment, non-empty lines from pods.txt (bash 3.2-compatible)
PODS=()
while IFS= read -r line; do
    PODS+=("$line")
done < <(grep -v '^\s*#' "$PODS_FILE" | grep -v '^\s*$')
N_PODS=${#PODS[@]}
if [ "$N_PODS" -eq 0 ]; then
    echo "No pods listed in $PODS_FILE" >&2; exit 1
fi

PER_POD=$((TOTAL / N_PODS))
EXTRA=$((TOTAL - PER_POD * N_PODS))   # last pod takes the remainder

echo "Sharding $TOTAL samples across $N_PODS pods ($PER_POD each, +$EXTRA on the last)"
echo "Output: $OUT"
echo ""

# Pre-create the shared output directories on one pod before launching all
FIRST_POD="${PODS[0]}"
ssh $FIRST_POD "mkdir -p $OUT/{images,masks,_shards}"

PIDS=()
for i in "${!PODS[@]}"; do
    POD="${PODS[$i]}"
    START=$((i * PER_POD + 1))
    if [ "$i" -eq $((N_PODS - 1)) ]; then
        N=$((PER_POD + EXTRA))
    else
        N=$PER_POD
    fi
    END=$((START + N - 1))
    LOG="$OUT/_shards/log_${START}_${END}.txt"
    echo "[shard $((i+1))/$N_PODS] $POD  start=$START  n=$N  -> $LOG"
    # Each ssh runs in parallel in the foreground locally; the remote
    # nohup detaches the actual generation process so ssh can return.
    ssh -o ConnectTimeout=15 $POD "cd /workspace/pixel-vs-prior && \
              nohup python3 -m data.generate \
                  --n $N --start-idx $START --total-n $TOTAL \
                  --seed 42 --workers $WORKERS_PER_POD --shard-only \
                  --out $OUT \
                  > $LOG 2>&1 < /dev/null & echo \$! > $OUT/_shards/pid_${START}_${END} \
                  ; disown" &
    PIDS+=("$!")
done
wait "${PIDS[@]}" 2>/dev/null || true

echo ""
echo "All $N_PODS shards launched in background."
echo "Monitor any shard with:"
echo "  ssh $FIRST_POD \"tail -f $OUT/_shards/log_*.txt\""
echo ""
echo "When all shards finish, run merge:"
echo "  ssh $FIRST_POD \"cd /workspace/pixel-vs-prior && python3 -m data.merge_shards --out $OUT\""
