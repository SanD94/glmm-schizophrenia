#!/usr/bin/env bash
set -euo pipefail

LOGDIR="logs"
mkdir -p "$LOGDIR" results plots

echo "══════════════════════════════════════════════════════════════"
echo "  GLMM False Discovery Simulation Pipeline"
echo "══════════════════════════════════════════════════════════════"

# ── Step 1: Data generation (must finish before experiments) ──────────────────
echo ""
echo "▶ [1/3] Generating data..."
Rscript 01_data_generation.R 2>&1 | tee "$LOGDIR/01_data_generation.log"
echo "✓ Data generation complete"

# ── Step 2: Run experiments 02–06 in parallel ─────────────────────────────────
echo ""
echo "▶ [2/3] Launching experiments 02–06 in parallel..."

declare -A PIDS
declare -A NAMES

NAMES[02]="input-size sweep"
NAMES[03]="sample-size sweep"
NAMES[04]="complexity sweep"
NAMES[05]="regularisation sweep"
NAMES[06]="signal sweep"

for script in 02_glmm_experiment.R 03_sample_size.R 04_complexity.R 05_regularisation.R 06_signal.R; do
    num="${script%%_*}"
    echo "  ├─ Starting $num: ${NAMES[$num]}"
    Rscript "$script" > "$LOGDIR/${script%.R}.log" 2>&1 &
    PIDS[$num]=$!
done

echo "  └─ All 5 experiments launched (PIDs: ${PIDS[*]})"
echo ""

# Wait for each and report
FAILED=0
for num in 02 03 04 05 06; do
    if wait "${PIDS[$num]}"; then
        echo "  ✓ $num: ${NAMES[$num]} completed"
    else
        echo "  ✗ $num: ${NAMES[$num]} FAILED (see $LOGDIR/)"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "ERROR: $FAILED experiment(s) failed. Check logs in $LOGDIR/"
    exit 1
fi

echo ""
echo "✓ All experiments complete"

# ── Step 3: Visualisation ─────────────────────────────────────────────────────
echo ""
echo "▶ [3/3] Generating plots..."
Rscript 07_visualisation.R 2>&1 | tee "$LOGDIR/07_visualisation.log"
echo "✓ Plots saved to plots/"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Pipeline complete. Results in results/, plots in plots/"
echo "══════════════════════════════════════════════════════════════"
