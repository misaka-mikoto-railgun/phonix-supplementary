# Regenerate every table, figure and track-level export the paper draws on.
#
# Windows equivalent of run_all_refresh.sh.
#
# The dataset and the trained checkpoints are NOT shipped with this repository
# (see README); point at them with -DataDir / -CkptDir / -RevCkptDir /
# -EvalCkptDir. Everything is written under -OutRoot and nothing already in the
# repository is touched unless -Publish is given.
#
#   .\scripts\run_all_refresh.ps1 -DataDir D:\dataset_v3 -CkptDir D:\checkpoints_original `
#       -RevCkptDir D:\checkpoints_revision
#
#   .\scripts\run_all_refresh.ps1 -Publish    # also copy into tables\ figures\ results_json\
#
[CmdletBinding()]
param(
    [string]$DataDir,
    [string]$CkptDir,
    [string]$RevCkptDir,
    [string]$EvalCkptDir,
    [string]$OutRoot,
    [string]$StatDir,
    [string]$Device = "cuda",
    [switch]$Publish
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$codeDir = Join-Path $repoRoot "code"
if (-not $DataDir)     { $DataDir     = Join-Path $repoRoot "data\dataset_v3" }
if (-not $CkptDir)     { $CkptDir     = Join-Path $repoRoot "checkpoints\full" }
if (-not $RevCkptDir)  { $RevCkptDir  = Join-Path $codeDir  "checkpoints" }
if (-not $EvalCkptDir) { $EvalCkptDir = Join-Path $codeDir  "ckpt_eval" }
if (-not $OutRoot)     { $OutRoot     = Join-Path $repoRoot "reruns\refresh" }

$results  = Join-Path $OutRoot "results"
$paperOut = Join-Path $OutRoot "paper_outputs"
$trackRoot = Join-Path $OutRoot "track_level"
$logDir   = Join-Path $OutRoot "logs"
New-Item -ItemType Directory -Force -Path $results, $paperOut, $trackRoot, $logDir | Out-Null

foreach ($d in @($DataDir, $CkptDir, $RevCkptDir, $EvalCkptDir)) {
    if (-not (Test-Path $d)) { throw "not found: $d" }
}

# Shared by every generator that loads a model.
$P = @("--data_dir", $DataDir, "--ckpt_dir", $CkptDir, "--rev_ckpt_dir", $RevCkptDir,
       "--eval_ckpt_dir", $EvalCkptDir, "--out_dir", $results)

$modelsAll = "A0_Proposed,A1_NoRoomInput,A2_withPrefLoss,A3_NoPrefInput,E1,E2,E3,E4,E5,E6,AC1,AC2,AC3"
$compareCandidates = "A1_NoRoomInput,A2_withPrefLoss,A3_NoPrefInput,E1,E2,E3,E4,E5,E6,AC1,AC2,AC3"

function Step {
    param([string]$Name, [string[]]$Args)
    Write-Host "`n=== $Name ==="
    & python @Args 2>&1 | Tee-Object -FilePath (Join-Path $logDir "$Name.log")
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

Push-Location $codeDir
try {
    # ── 1. dataset sanity ────────────────────────────────────────────────────
    Step "frequency_grid" @("check_frequency_grid_consistency.py", "--data_dir", $DataDir)

    # ── 2. paper tables ──────────────────────────────────────────────────────
    Step "table2_main"       (@("table2_revision.py")   + $P)
    Step "table5_ood"        (@("table5_ood.py")        + $P)
    Step "table7_perceptual" (@("table7_perceptual.py") + $P)
    Step "table4_paired"     (@("table4_paired.py")     + $P)
    Step "paired_stats"      (@("paired_stats.py")      + $P)
    Step "track_stats"       (@("track_stats.py")       + $P)
    Step "ac_biquad_table"   (@("ac_biquad_table.py")   + $P)
    # table6_biquad assembles rows produced above; it loads no model and needs no GPU.
    Step "table6_biquad"     @("table6_biquad.py", "--results_dir", $results, "--out_dir", $results)

    # ── 3. gain / centre-frequency summaries and saturation ──────────────────
    $gf = @("--eval_only", "--data_dir", $DataDir, "--save_dir", $RevCkptDir, "--out_dir", $results)
    Step "gf_A0_synth" (@("run_gain_freq_ablation.py") + $gf + @("--variant", "A0",
        "--configs", "g6_f16k", "g12_f16k", "g12_f20k", "--test_split", "test_synth",
        "--seeds", "42", "123", "7"))
    Step "gf_A0_real"  (@("run_gain_freq_ablation.py") + $gf + @("--variant", "A0",
        "--configs", "g6_f16k", "g12_f16k", "--test_split", "test_real",
        "--seeds", "42", "123", "7"))
    Step "gf_A2_synth" (@("run_gain_freq_ablation.py") + $gf + @("--variant", "A2",
        "--configs", "g12_f16k", "--test_split", "test_synth", "--seeds", "42", "123", "7"))
    Step "gf_A2_real"  (@("run_gain_freq_ablation.py") + $gf + @("--variant", "A2",
        "--configs", "g12_f16k", "--test_split", "test_real", "--seeds", "42", "123", "7"))

    $pd = @("--data_dir", $DataDir, "--save_dir", $RevCkptDir, "--out_dir", $results,
            "--seeds", "42", "123", "7")
    # g6_f20k is the one cell of the 2x2 matrix that was never trained (the +/-6
    # baseline is only needed at 16 kHz), so the configs are named rather than "all".
    Step "param_dist_synth" (@("param_dist_gain_freq.py") + $pd + @("--test_split", "test_synth",
        "--configs", "g6_f16k", "g12_f16k", "g12_f20k"))
    Step "param_dist_real"  (@("param_dist_gain_freq.py") + $pd + @("--test_split", "test_real",
        "--configs", "g6_f16k", "g12_f16k"))

    # ── 4. track-level exports (three splits: dump, then paired comparison) ──
    foreach ($split in @("test_synth", "test_real", "paired_mode_test")) {
        if ($split -eq "paired_mode_test") { $key = "pair_id"; $tag = "paired_mode" }
        else                               { $key = "track_id"; $tag = $split }
        Step "track_${tag}_all" @("export_track_level_predictions.py",
            "--data_dir", $DataDir, "--split", $split, "--ckpt_dir", $CkptDir,
            "--models", "all", "--candidates", "none", "--device", $Device,
            "--out_dir", (Join-Path $trackRoot "${tag}_all"))
        Step "track_${tag}_compare" @("export_track_level_predictions.py",
            "--data_dir", $DataDir, "--split", $split, "--ckpt_dir", $CkptDir,
            "--models", $modelsAll, "--baseline", "A0_Proposed", "--baseline-seed", "42",
            "--candidates", $compareCandidates, "--group-key", $key, "--device", $Device,
            "--out_dir", (Join-Path $trackRoot "${tag}_compare"))
    }

    # ── 5. figures and the consolidated tables ───────────────────────────────
    Step "consolidate" @("consolidate.py", "--results_dir", $results, "--out_dir", $paperOut)

    Step "overlay_sample" @("extract_overlay_sample.py", "--data_dir", $DataDir,
        "--rev_ckpt_dir", $RevCkptDir, "--out_dir", $results)
    Step "overlay_figure" @("make_overlay_figure.py", "--results_dir", $results,
        "--out_dir", $paperOut)

    # fig_ac_fitting needs the per-sample .npy dumps written by ac_fitting_A.py /
    # ac_fitting_C.py. Those are refit/retrain runs and their dumps are not shipped,
    # so this figure is only regenerated when -StatDir points at them.
    if ($StatDir -and (Test-Path $StatDir)) {
        Step "ac_fitting_figure" @("plot_ac_fitting.py", "--stat_dir", $StatDir,
            "--out_dir", $paperOut)
    } else {
        Write-Host "`nSKIP fig_ac_fitting: pass -StatDir with the ac_fitting_{A,C}.py .npy dumps to regenerate it"
    }
}
finally {
    Pop-Location
}

# ── 6. optional: copy into the locations this repository publishes ───────────
# The generators write under $OutRoot; the committed copies live in tables\,
# figures\ and results_json\. This step is what keeps the two in sync, and is
# the manual copy that used to be undocumented.
if ($Publish) {
    Write-Host "`n=== publish ==="
    $tabDst = Join-Path $repoRoot "tables"
    $figDst = Join-Path $repoRoot "figures"
    $jsonDst = Join-Path $repoRoot "results_json"
    New-Item -ItemType Directory -Force -Path $tabDst, $figDst, $jsonDst | Out-Null
    Copy-Item (Join-Path $results "table*.csv") $tabDst -Force
    Copy-Item (Join-Path $paperOut "tables\T[2-5]_*.csv") $tabDst -Force
    Copy-Item (Join-Path $paperOut "figures\F[1-6]_*.png") $figDst -Force
    Copy-Item (Join-Path $paperOut "figures\F[1-6]_*.pdf") $figDst -Force
    Copy-Item (Join-Path $paperOut "figuresig_*.png") $figDst -Force -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $paperOut "figuresig_*.pdf") $figDst -Force -ErrorAction SilentlyContinue
    # param_dist writes its histograms next to its JSON rather than into paper_outputs
    Copy-Item (Join-Path $results "param_dist_*.png") $figDst -Force
    Copy-Item (Join-Path $results "param_dist_*.pdf") $figDst -Force
    Copy-Item (Join-Path $results "*.json") $jsonDst -Force
    Write-Host "  copied into tables\ figures\ results_json\"
}

Write-Host "`nAll refresh jobs completed. Outputs:"
Write-Host "  $OutRoot"
if (-not $Publish) { Write-Host "  (pass -Publish to copy them into tables\ figures\ results_json\)" }
