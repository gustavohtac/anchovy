#!/usr/bin/env bash
# Generate synthetic radiograph datasets at one or more sizes into
# /workspace/pixel-vs-prior/datasets/n<N>/.  Each call to data.generate uses a
# fixed seed so the first N samples of the larger pool match the smaller
# pool — making smaller regimes a strict prefix of larger ones for fair
# scaling comparisons.
#
# Usage:
#     bash scripts/prep_data.sh                # default: 1k 10k 100k
#     bash scripts/prep_data.sh 1000000        # also generate the 1M pool (~5h on 32 CPUs)
#     bash scripts/prep_data.sh 1000 10000     # custom set
set -euo pipefail
cd /workspace/pixel-vs-prior

if [ "$#" -ge 1 ]; then
    SIZES=("$@")
else
    SIZES=(1000 10000 100000)
fi

mkdir -p datasets
for N in "${SIZES[@]}"; do
    OUT="datasets/n${N}"
    if [ -e "$OUT/labels.csv" ]; then
        echo "[skip] $OUT already exists ($(wc -l < $OUT/labels.csv) rows)"
        continue
    fi
    echo "[generate] n=${N}  ->  $OUT"
    python3 -m data.generate \
        --n "$N" \
        --workers 32 \
        --out "$OUT" \
        --seed 42 \
        --train-ratio 0.80 --val-ratio 0.10 --split-seed 0
done
echo "datasets ready: ${SIZES[*]}"
