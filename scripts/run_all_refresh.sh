#!/usr/bin/env bash
# Regenerate the paper tables, figures and track-level exports.
#
# POSIX equivalent of run_all_refresh.ps1.
#
# Requires the dataset and the trained checkpoints, which are NOT shipped with
# this repository (see README). Set DATA_DIR / CKPT_DIR to point at them.
#
#   DATA_DIR=/path/to/dataset_v3 CKPT_DIR=/path/to/checkpoints/full \
#       bash scripts/run_all_refresh.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${CODE_DIR:-$REPO_ROOT/code}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/dataset_v3}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints/full}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/reruns/a0_proposed_refresh}"
DEVICE="${DEVICE:-cuda}"

PAPER_OUT="$OUT_ROOT/paper_outputs"
TRACK_ROOT="$OUT_ROOT/track_level"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$PAPER_OUT" "$TRACK_ROOT" "$LOG_DIR"

MODELS_ALL="A0_Proposed,A1_NoRoomInput,A2_withPrefLoss,A3_NoPrefInput,E1,E2,E3,E4,E5,E6,AC1,AC2,AC3"
COMPARE_CANDIDATES="A1_NoRoomInput,A2_withPrefLoss,A3_NoPrefInput,E1,E2,E3,E4,E5,E6,AC1,AC2,AC3"

for d in "$DATA_DIR" "$CKPT_DIR"; do
    [ -d "$d" ] || { echo "ERROR: not found: $d" >&2; exit 1; }
done

run_and_log() {
    local name="$1"; shift
    echo ""
    echo "=== $name ==="
    "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

cd "$CODE_DIR"

run_and_log "check_frequency_grid_consistency" \
    python check_frequency_grid_consistency.py --data_dir "$DATA_DIR"

    python experiments_fixed_updated.py --data_dir "$DATA_DIR" \
        --ckpt_dir "$CKPT_DIR" --out_dir "$PAPER_OUT" --device "$DEVICE" --models all

run_and_log "track_test_synth_all" \
    python export_track_level_predictions.py --data_dir "$DATA_DIR" --split test_synth \
        --ckpt_dir "$CKPT_DIR" --models all --candidates none --device "$DEVICE" \
        --out_dir "$TRACK_ROOT/test_synth_all"

run_and_log "track_test_synth_compare" \
    python export_track_level_predictions.py --data_dir "$DATA_DIR" --split test_synth \
        --ckpt_dir "$CKPT_DIR" --models "$MODELS_ALL" --baseline A0_Proposed \
        --candidates "$COMPARE_CANDIDATES" --group-key track_id --device "$DEVICE" \
        --out_dir "$TRACK_ROOT/test_synth_compare"

run_and_log "track_test_real_all" \
    python export_track_level_predictions.py --data_dir "$DATA_DIR" --split test_real \
        --ckpt_dir "$CKPT_DIR" --models all --candidates none --device "$DEVICE" \
        --out_dir "$TRACK_ROOT/test_real_all"

run_and_log "track_test_real_compare" \
    python export_track_level_predictions.py --data_dir "$DATA_DIR" --split test_real \
        --ckpt_dir "$CKPT_DIR" --models "$MODELS_ALL" --baseline A0_Proposed \
        --candidates "$COMPARE_CANDIDATES" --group-key track_id --device "$DEVICE" \
        --out_dir "$TRACK_ROOT/test_real_compare"

run_and_log "track_paired_mode_all" \
    python export_track_level_predictions.py --data_dir "$DATA_DIR" --split paired_mode_test \
        --ckpt_dir "$CKPT_DIR" --models all --candidates none --device "$DEVICE" \
        --out_dir "$TRACK_ROOT/paired_mode_all"

run_and_log "track_paired_mode_compare" \
    python export_track_level_predictions.py --data_dir "$DATA_DIR" --split paired_mode_test \
        --ckpt_dir "$CKPT_DIR" --models "$MODELS_ALL" --baseline A0_Proposed \
        --candidates "$COMPARE_CANDIDATES" --group-key pair_id --device "$DEVICE" \
        --out_dir "$TRACK_ROOT/paired_mode_compare"

echo ""
echo "All refresh jobs completed. Outputs:"
echo "  $OUT_ROOT"
