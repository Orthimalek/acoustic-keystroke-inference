#!/usr/bin/env bash
# =============================================================================
# run_missing_experiments.sh
# Runs all missing multi-seed experiments for IEEE Access revision.
#
# Prerequisites (run once if needed):
#   pip install timm librosa torch torchvision torchaudio
#
# Usage:
#   chmod +x run_missing_experiments.sh
#   ./run_missing_experiments.sh
#
# Expected runtime on M1 Mac (MPS): ~6–10 hours total
# Results written to: ./results/multiseed/
# =============================================================================

set -e  # exit on any error

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

EPOCHS=100
SEEDS="42,123,456,789,1337"
OUTPUT="./results/multiseed"
MKA_ROOT="${MKA_ROOT:-$HOME/Downloads/Multi-Keyboard Acoustic (MKA) Datasets/MKA datasets}"

echo "============================================================"
echo "  Acoustic Keystroke — Missing Multi-Seed Experiments"
echo "  Seeds : $SEEDS"
echo "  Epochs: $EPOCHS"
echo "  Output: $OUTPUT"
echo "  MKA   : $MKA_ROOT"
echo "============================================================"
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
run() {
    echo ""
    echo ">>> $*"
    python train_multiseed.py "$@" \
        --epochs   $EPOCHS \
        --seeds    "$SEEDS" \
        --output_dir "$OUTPUT"
}

# =============================================================================
# BLOCK 1: CoAtNet-0  — E2 and E1+E2  (E1 already done with 5 seeds)
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BLOCK 1 / 4 : CoAtNet-0  E2 + E1+E2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run --dataset custom --env E2   --model coatnet --max_clips 50
run --dataset both              --model coatnet --max_clips 50

# =============================================================================
# BLOCK 2: MaxViT-Tiny — E2 and E1+E2  (E1 already done with 5 seeds)
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BLOCK 2 / 4 : MaxViT-Tiny  E2 + E1+E2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run --dataset custom --env E2   --model maxvit  --max_clips 50
run --dataset both              --model maxvit  --max_clips 50

# =============================================================================
# BLOCK 3: Swin-Tiny — ALL conditions  (none done yet for Swin)
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BLOCK 3 / 4 : Swin-Tiny  E1, E2, E1+E2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run --dataset custom --env E1   --model swin    --max_clips 50
run --dataset custom --env E2   --model swin    --max_clips 50
run --dataset both              --model swin    --max_clips 50

# =============================================================================
# BLOCK 4: MKA multi-seed — all 3 models  (CoAtNet + MaxViT MKA already done)
# Note: re-running CoAtNet + MaxViT MKA is fast since we just need to confirm;
#       skip them if you already have results/multiseed/mka_coatnet_multiseed.json
#       and results/multiseed/mka_maxvit_multiseed.json
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BLOCK 4 / 4 : MKA multi-seed — Swin-Tiny + subsampled (36 clips)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Swin-Tiny on MKA full (36 samples/class in MKA by design)
run --dataset mka --model swin --root "$MKA_ROOT"

# Swin-Tiny on MKA subsampled to 36 clips (verifies parity with Custom setup)
# max_clips=36 mirrors MKA's 36-sample-per-class size on the Custom data
run --dataset custom --env E1 --model swin --max_clips 36
run --dataset custom --env E2 --model swin --max_clips 36

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results: $OUTPUT/"
echo "  Run:  python print_results.py  to see formatted table"
echo "============================================================"
