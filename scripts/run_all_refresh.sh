#!/usr/bin/env bash
# Regenerate every table, figure and track-level export the paper draws on.
#
# POSIX equivalent of run_all_refresh.ps1.
#
# The dataset and the trained checkpoints are NOT shipped with this repository
# (see README); point at them with DATA_DIR / CKPT_DIR / REV_CKPT_DIR /
# EVAL_CKPT_DIR. Everything is written under OUT_ROOT and nothing already in
# the repository is touched unless PUBLISH=1.
#
#   DATA_DIR=/path/to/dataset_v3 CKPT_DIR=/path/to/checkpoints_original \
#   REV_CKPT_DIR=/path/to/checkpoints_revision \
#       bash scripts/run_all_refresh.sh
#
#   PUBLISH=1 bash scripts/run_all_refresh.sh   # also copy into tables/ figures/ results_json/
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${CODE_DIR:-$REPO_ROOT/code}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/dataset_v3}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints/full}"
REV_CKPT_DIR="${REV_CKPT_DIR:-$CODE_DIR/checkpoints}"
EVAL_CKPT_DIR="${EVAL_CKPT_DIR:-$CODE_DIR/ckpt_eval}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/reruns/refresh}"
DEVICE="${DEVICE:-cuda}"
PUBLISH="${PUBLISH:-0}"

RESULTS="$OUT_ROOT/results"
PAPER_OUT="$OUT_ROOT/paper_outputs"
TRACK_ROOT="$OUT_ROOT/track_level"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$RESULTS" "$PAPER_OUT" "$TRACK_ROOT" "$LOG_DIR"

for d in "$DATA_DIR" "$CKPT_DIR" "$REV_CKPT_DIR" "$EVAL_CKPT_DIR"; do
    [ -d "$d" ] || { echo "ERROR: not found: $d" >&2; exit 1; }
done

# Shared by every generator that loads a model.
P=(--data_dir "$DATA_DIR" --ckpt_dir "$CKPT_DIR" --rev_ckpt_dir "$REV_CKPT_DIR"
   --eval_ckpt_dir "$EVAL_CKPT_DIR" --out_dir "$RESULTS")

MODELS_ALL="A0_Proposed,A1_NoRoomInput,A2_withPrefLoss,A3_NoPrefInput,E1,E2,E3,E4,E5,E6,AC1,AC2,AC3"
COMPARE_CANDIDATES="A1_NoRoomInput,A2_withPrefLoss,A3_NoPrefInput,E1,E2,E3,E4,E5,E6,AC1,AC2,AC3"

step() {
    local name="$1"; shift
    echo ""
    echo "=== $name ==="
    "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

cd "$CODE_DIR"

# ── 1. dataset sanity ────────────────────────────────────────────────────────
step frequency_grid python check_frequency_grid_consistency.py --data_dir "$DATA_DIR"

# ── 2. paper tables ──────────────────────────────────────────────────────────
step table2_main        python table2_revision.py    "${P[@]}"
step table5_ood         python table5_ood.py         "${P[@]}"
step table7_perceptual  python table7_perceptual.py  "${P[@]}"
step table4_paired      python table4_paired.py      "${P[@]}"
step paired_stats       python paired_stats.py       "${P[@]}"
step track_stats        python track_stats.py        "${P[@]}"
step ac_biquad_table    python ac_biquad_table.py    "${P[@]}"
# table6_biquad assembles rows produced above; it loads no model and needs no GPU.
step table6_biquad      python table6_biquad.py --results_dir "$RESULTS" --out_dir "$RESULTS"

# ── 3. gain / centre-frequency summaries and saturation ──────────────────────
GF=(--eval_only --data_dir "$DATA_DIR" --save_dir "$REV_CKPT_DIR" --out_dir "$RESULTS")
step gf_A0_synth python run_gain_freq_ablation.py "${GF[@]}" --variant A0 --test_split test_synth --seeds 42 123 7 --configs g6_f16k g12_f16k g12_f20k
step gf_A0_real  python run_gain_freq_ablation.py "${GF[@]}" --variant A0 --test_split test_real  --seeds 42 123 7 --configs g6_f16k g12_f16k
step gf_A2_synth python run_gain_freq_ablation.py "${GF[@]}" --variant A2 --test_split test_synth --seeds 42 123 7 --configs g12_f16k
step gf_A2_real  python run_gain_freq_ablation.py "${GF[@]}" --variant A2 --test_split test_real  --seeds 42 123 7 --configs g12_f16k

# g6_f20k is the one cell of the 2x2 matrix that was never trained (the +/-6
# baseline is only needed at 16 kHz), so the configs are named rather than "all".
PD=(--data_dir "$DATA_DIR" --save_dir "$REV_CKPT_DIR" --out_dir "$RESULTS" --seeds 42 123 7)
step param_dist_synth python param_dist_gain_freq.py "${PD[@]}" --test_split test_synth --configs g6_f16k g12_f16k g12_f20k
step param_dist_real  python param_dist_gain_freq.py "${PD[@]}" --test_split test_real  --configs g6_f16k g12_f16k

# ── 4. track-level exports (three splits: dump, then paired comparison) ──────
for split in test_synth test_real paired_mode_test; do
    case "$split" in
        paired_mode_test) key=pair_id;  tag=paired_mode ;;
        *)                key=track_id; tag="$split" ;;
    esac
    step "track_${tag}_all" python export_track_level_predictions.py --data_dir "$DATA_DIR" --split "$split" --ckpt_dir "$CKPT_DIR" --models all --candidates none --device "$DEVICE" --out_dir "$TRACK_ROOT/${tag}_all"
    step "track_${tag}_compare" python export_track_level_predictions.py --data_dir "$DATA_DIR" --split "$split" --ckpt_dir "$CKPT_DIR" --models "$MODELS_ALL" --baseline A0_Proposed --baseline-seed 42 --candidates "$COMPARE_CANDIDATES" --group-key "$key" --device "$DEVICE" --out_dir "$TRACK_ROOT/${tag}_compare"
done

# ── 5. figures and the consolidated tables ───────────────────────────────────
step consolidate    python consolidate.py --results_dir "$RESULTS" --out_dir "$PAPER_OUT"
step overlay_sample python extract_overlay_sample.py --data_dir "$DATA_DIR" --rev_ckpt_dir "$REV_CKPT_DIR" --out_dir "$RESULTS"
step overlay_figure python make_overlay_figure.py --results_dir "$RESULTS" --out_dir "$PAPER_OUT"

# fig_ac_fitting needs the per-sample .npy dumps written by ac_fitting_A.py /
# ac_fitting_C.py. Those are refit/retrain runs and their dumps are not shipped,
# so this figure is only regenerated when STAT_DIR points at them.
if [ -n "${STAT_DIR:-}" ] && [ -d "$STAT_DIR" ]; then
    step ac_fitting_figure python plot_ac_fitting.py --stat_dir "$STAT_DIR" --out_dir "$PAPER_OUT"
else
    echo ""
    echo "SKIP fig_ac_fitting: set STAT_DIR to the ac_fitting_{A,C}.py .npy dumps to regenerate it"
fi

# ── 6. optional: copy into the locations this repository publishes ───────────
# The generators write under OUT_ROOT; the committed copies live in tables/,
# figures/ and results_json/. This step is what keeps the two in sync, and is
# the manual copy that used to be undocumented.
if [ "$PUBLISH" = "1" ]; then
    echo ""
    echo "=== publish ==="
    mkdir -p "$REPO_ROOT/tables" "$REPO_ROOT/figures" "$REPO_ROOT/results_json"
    cp "$RESULTS"/table*.csv             "$REPO_ROOT/tables/"
    cp "$PAPER_OUT"/tables/T[2-5]_*.csv  "$REPO_ROOT/tables/"
    cp "$PAPER_OUT"/figures/F[1-6]_*.png "$PAPER_OUT"/figures/F[1-6]_*.pdf "$REPO_ROOT/figures/"
    cp "$PAPER_OUT"/figures/fig_*.png "$PAPER_OUT"/figures/fig_*.pdf "$REPO_ROOT/figures/" 2>/dev/null || true
    # param_dist writes its histograms next to its JSON rather than into paper_outputs/
    cp "$RESULTS"/param_dist_*.png "$RESULTS"/param_dist_*.pdf "$REPO_ROOT/figures/"
    cp "$RESULTS"/*.json                 "$REPO_ROOT/results_json/"
    echo "  copied into tables/ figures/ results_json/"
fi

echo ""
echo "All refresh jobs completed. Outputs:"
echo "  $OUT_ROOT"
[ "$PUBLISH" = "1" ] || echo "  (set PUBLISH=1 to copy them into tables/ figures/ results_json/)"
