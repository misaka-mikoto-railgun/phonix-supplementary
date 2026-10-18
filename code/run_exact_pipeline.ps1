# run_exact_pipeline.ps1 — exact-biquad revision: remaining training + all tables/figures
#
# Run from repo_clean\code :
#   .\run_exact_pipeline.ps1                      # everything
#   .\run_exact_pipeline.ps1 -SkipTraining        # tables/figures only (checkpoints already there)
#
# Layout assumed (override with parameters):
#   code dir      = this script's directory          (repo_clean\code)
#   dataset       = ..\..\data\dataset_v3            (relative to code\)
#   rev ckpts     = code\checkpoints                 (A0_g12_f16k_s42.pt, A2_..., AC*_Biquad_g12.pt)
#   "original"    = code\checkpoints\full            (A0_Proposed.pt, E3/E4, AC dense ... from train_full)
#   staging       = code\ckpt_eval                   (rebuilt by make_eval_staging.py)
#   outputs       = code\reruns\exact                (results\, paper_outputs\, logs\)
#
# Training steps are skipped automatically when their checkpoint already exists.
# Every step logs to reruns\exact\logs\<step>.log and the script stops on the first failure.

[CmdletBinding()]
param(
    [string]$DataDir,
    [string]$RevCkptDir,
    [string]$CkptDir,
    [string]$EvalCkptDir,
    [string]$OutRoot,
    [switch]$SkipTraining,
    [switch]$Publish
)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"            # cp949 console: keep non-ASCII prints (em dash, ±) from killing a step
$env:PYTHONIOENCODING = "utf-8"

$codeDir  = $PSScriptRoot
$repoRoot = Split-Path -Parent $codeDir
if (-not $DataDir)     { $DataDir     = (Resolve-Path (Join-Path $codeDir "..\..\data\dataset_v3")).Path }
if (-not $RevCkptDir)  { $RevCkptDir  = Join-Path $codeDir "checkpoints" }
if (-not $CkptDir)     { $CkptDir     = Join-Path $codeDir "checkpoints\full" }
if (-not $EvalCkptDir) { $EvalCkptDir = Join-Path $codeDir "ckpt_eval" }
if (-not $OutRoot)     { $OutRoot     = Join-Path $codeDir "reruns\exact" }

$results  = Join-Path $OutRoot "results"
$paperOut = Join-Path $OutRoot "paper_outputs"
$statDir  = Join-Path $paperOut "stats"
$logDir   = Join-Path $OutRoot "logs"
New-Item -ItemType Directory -Force -Path $results, $paperOut, $statDir, $logDir | Out-Null
foreach ($d in @($DataDir, $RevCkptDir, $CkptDir)) { if (-not (Test-Path $d)) { throw "not found: $d" } }

function Step {
    # NB: parameter must not be called $Args — that is a PowerShell automatic variable and comes out empty inside the function
    param([string]$Name, [string[]]$PyArgs)
    if (-not $PyArgs -or $PyArgs.Count -eq 0) { throw "Step ${Name}: empty argument list" }
    $t0 = Get-Date
    Write-Host "`n=== $Name ===  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "    python -u $($PyArgs -join ' ')" -ForegroundColor DarkGray
    $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"   # otherwise the first stderr line aborts and hides the traceback
    & python -u @PyArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $logDir "$Name.log")
    $ErrorActionPreference = $eap
    if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit $LASTEXITCODE) — see $logDir\$Name.log" }
    Write-Host "    done in $([int]((Get-Date) - $t0).TotalMinutes) min" -ForegroundColor DarkGray
}
function AllExist { param([string[]]$Paths) foreach ($p in $Paths) { if (-not (Test-Path $p)) { return $false } }; return $true }

Push-Location $codeDir
try {
    # ── sanity: the exact response must be wired in, otherwise everything below is wasted ──
    $wired = Select-String -Path "model.py","arch_biquad.py","baselines.py" -Pattern "ExactPEQResponse" -Quiet
    if (-not $wired) { throw "ExactPEQResponse not referenced in model.py / arch_biquad.py / baselines.py — swap the surrogate first" }
    python exact_peq_response.py; if ($LASTEXITCODE -ne 0) { throw "exact_peq_response self-test failed" }
    python exact_peq_np.py;       if ($LASTEXITCODE -ne 0) { throw "exact_peq_np self-test failed" }

    if (-not $SkipTraining) {
        # ── 1. A0 / A2 three seeds (±12, fc 16k) + A0 fc 20k + the ±6 control ──────────
        $gf = @("run_gain_freq_ablation.py", "--data_dir", $DataDir, "--save_dir", $RevCkptDir, "--out_dir", $results)

        $need = @(42,123,7) | ForEach-Object { Join-Path $RevCkptDir "A0_g12_f16k_s$_.pt"; Join-Path $RevCkptDir "A0_g12_f20k_s$_.pt" }
        if (AllExist $need) { Write-Host "skip A0 g12 (checkpoints present)" }
        else { Step "train_A0_g12" ($gf + @("--variant","A0","--configs","g12_f16k","g12_f20k","--seeds","42","123","7")) }

        if (Test-Path (Join-Path $RevCkptDir "A0_g6_f16k_s42.pt")) { Write-Host "skip A0 g6 (present)" }
        else { Step "train_A0_g6" ($gf + @("--variant","A0","--configs","g6_f16k","--seeds","42")) }

        $need = @(42,123,7) | ForEach-Object { Join-Path $RevCkptDir "A2_g12_f16k_s$_.pt" }
        if (AllExist $need) { Write-Host "skip A2 g12 (present)" }
        else { Step "train_A2_g12" ($gf + @("--variant","A2","--configs","g12_f16k","--seeds","42","123","7")) }
    }

    # ── 2. manifest + evaluation staging (A0/A2 = seed 7 ±12, rest from checkpoints\full) ──
    Step "make_manifests"    @("make_manifests.py", "--data_dir", $DataDir, "--ckpt_dir", $CkptDir, "--rev_ckpt_dir", $RevCkptDir, "--out_dir", $repoRoot)
    Step "make_eval_staging" @("make_eval_staging.py", "--manifest", (Join-Path $repoRoot "checkpoints_manifest.json"), "--ckpt_dir", $CkptDir, "--rev_ckpt_dir", $RevCkptDir, "--out_dir", $EvalCkptDir)
    # option C checkpoints carry the _g12 tag; experiments_fixed_updated looks for the untagged name
    foreach ($arch in @("AC1_BiLSTM","AC2_GRU","AC3_Conformer")) {
        $src = Join-Path $RevCkptDir "${arch}_Biquad_g12.pt"
        if (-not (Test-Path $src)) { throw "missing option C checkpoint: $src" }
        Copy-Item $src (Join-Path $EvalCkptDir "${arch}_Biquad.pt") -Force
    }

    # ── 3. per-sample stats for every model (feeds ac_fitting_A/C/D and fig_ac_fitting) ──
    Step "experiments_stats" @("experiments_fixed_updated.py", "--data_dir", $DataDir, "--ckpt_dir", $EvalCkptDir, "--out_dir", $paperOut, "--device", "cuda")
    Step "save_a0_lsd"       @("save_a0_lsd.py", "--data_dir", $DataDir, "--rev_ckpt_dir", $RevCkptDir, "--stat_dir", $statDir)

    if (-not $SkipTraining) {
        # ── 4. Option C — biquad-head retraining of the three AC encoders (pref_curve shortcut removed) ──
        $shortcut = Select-String -Path "arch_biquad.py" -Pattern "room_corr \+ pref_curve \+ pref_res" -Quiet
        if ($shortcut) { throw "arch_biquad.py still adds pref_curve to the output — change line ~92 to 'pred = room_corr + pref_res' first" }
        foreach ($arch in @("AC1_BiLSTM","AC2_GRU","AC3_Conformer")) {
            if (Test-Path (Join-Path $RevCkptDir "${arch}_Biquad_g12.pt")) { Write-Host "skip optionC $arch (present)"; continue }
            Step "optionC_$arch" @("ac_fitting_C.py", "--arch", $arch, "--data_dir", $DataDir, "--ckpt_dir", $RevCkptDir,
                                   "--stat_dir", $statDir, "--out_dir", $paperOut, "--gain_max", "12", "--tag", "_g12", "--seed", "42")
        }
    } else {
        foreach ($arch in @("AC1_BiLSTM","AC2_GRU","AC3_Conformer")) {   # eval-only pass to refresh the .npy dumps
            Step "optionC_eval_$arch" @("ac_fitting_C.py", "--arch", $arch, "--data_dir", $DataDir, "--ckpt_dir", $RevCkptDir,
                                        "--stat_dir", $statDir, "--out_dir", $paperOut, "--gain_max", "12", "--tag", "_g12", "--skip_train")
        }
    }

    # ── 5. Option A (per-sample SciPy fit, exact biquad) and Option D (naive 7-pt, exact biquad) ──
    Step "optionA_AC2" @("ac_fitting_A.py", "--stat_dir", $statDir, "--out_dir", $paperOut, "--model", "AC2_GRU", "--gain_max", "12")
    Step "optionD_AC2" @("ac_fitting_D.py", "--stat_dir", $statDir, "--out_dir", $paperOut, "--model", "AC2_GRU")
    # table6_biquad looks for these names under results (else it falls back to the repo's old results_json copies!)
    Copy-Item (Join-Path $paperOut "tables\table_ac_fitting_A.csv")   (Join-Path $results "ac_fitting_A_g12.csv")   -Force
    Copy-Item (Join-Path $paperOut "tables\ac_fitting_D_naive7pt.csv") (Join-Path $results "ac_fitting_D_naive7pt.csv") -Force

    # ── 6. all paper tables, stats, figures (run_all_refresh with this layout) ──
    $refresh = Join-Path $repoRoot "scripts\run_all_refresh.ps1"
    $refreshArgs = @{ DataDir=$DataDir; CkptDir=$CkptDir; RevCkptDir=$RevCkptDir; EvalCkptDir=$EvalCkptDir; OutRoot=$OutRoot; StatDir=$statDir }
    if ($Publish) { $refreshArgs.Publish = $true }
    Write-Host "`n=== run_all_refresh ===" -ForegroundColor Cyan
    & $refresh @refreshArgs
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne $null) { throw "run_all_refresh failed" }

    # ── 7. final check: Gaussian column must equal exact column for every model ──
    Step "exact_check" @("eval_exact_biquad.py", "--data_dir", $DataDir, "--rev_ckpt_dir", $RevCkptDir, "--eval_ckpt_dir", $EvalCkptDir, "--out_dir", (Join-Path $results "exact_check"))
}
finally { Pop-Location }

Write-Host "`nPipeline complete." -ForegroundColor Green
Write-Host "  tables/stats : $results"
Write-Host "  figures      : $paperOut"
Write-Host "  logs         : $logDir"
Write-Host "  NOT run (layout-specific): embedded\onnx\export_onnx.py → quantize_ptq.py → eval_int8.py"