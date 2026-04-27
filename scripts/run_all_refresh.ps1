$ErrorActionPreference = "Stop"

$root = Join-Path (Get-Location) "reruns\a0_proposed_refresh_20260426"
$paperOut = Join-Path $root "paper_outputs"
$trackRoot = Join-Path $root "track_level"
$logDir = Join-Path $root "logs"

New-Item -ItemType Directory -Force -Path $paperOut, $trackRoot, $logDir | Out-Null

$dataDir = ".\data\dataset_v3"
$ckptDir = ".\checkpoints\full"
$modelsAll = "A0_Proposed,A1_NoRoomInput,A2_withPrefLoss,A3_NoPrefInput,E1,E2,E3,E4,E5,E6,AC1,AC2,AC3"
$compareCandidates = "A1_NoRoomInput,A2_withPrefLoss,A3_NoPrefInput,E1,E2,E3,E4,E5,E6,AC1,AC2,AC3"

function Run-And-Log {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    $logPath = Join-Path $logDir "$Name.log"
    Write-Host "`n=== $Name ==="
    & {
        & $Command 2>&1 | Tee-Object -FilePath $logPath
    }
}

Run-And-Log "check_frequency_grid_consistency" {
    python check_frequency_grid_consistency.py --data_dir $dataDir
}

Run-And-Log "fairness_table" {
    python alpha_sweep.py --task fairness --data_dir $dataDir --ckpt_dir $ckptDir --table_out_dir (Join-Path $paperOut "tables")
}

Run-And-Log "paper_experiments" {
    python experiments_fixed_updated.py --data_dir $dataDir --ckpt_dir $ckptDir --out_dir $paperOut --device cuda --models all
}

Run-And-Log "track_test_synth_all" {
    python export_track_level_predictions.py --data_dir $dataDir --split test_synth --ckpt_dir $ckptDir --models all --candidates none --device cuda --out_dir (Join-Path $trackRoot "test_synth_all")
}

Run-And-Log "track_test_synth_compare" {
    python export_track_level_predictions.py --data_dir $dataDir --split test_synth --ckpt_dir $ckptDir --models $modelsAll --baseline A0_Proposed --candidates $compareCandidates --group-key track_id --device cuda --out_dir (Join-Path $trackRoot "test_synth_compare")
}

Run-And-Log "track_test_real_all" {
    python export_track_level_predictions.py --data_dir $dataDir --split test_real --ckpt_dir $ckptDir --models all --candidates none --device cuda --out_dir (Join-Path $trackRoot "test_real_all")
}

Run-And-Log "track_test_real_compare" {
    python export_track_level_predictions.py --data_dir $dataDir --split test_real --ckpt_dir $ckptDir --models $modelsAll --baseline A0_Proposed --candidates $compareCandidates --group-key track_id --device cuda --out_dir (Join-Path $trackRoot "test_real_compare")
}

Run-And-Log "track_paired_mode_all" {
    python export_track_level_predictions.py --data_dir $dataDir --split paired_mode_test --ckpt_dir $ckptDir --models all --candidates none --device cuda --out_dir (Join-Path $trackRoot "paired_mode_all")
}

Run-And-Log "track_paired_mode_compare" {
    python export_track_level_predictions.py --data_dir $dataDir --split paired_mode_test --ckpt_dir $ckptDir --models $modelsAll --baseline A0_Proposed --candidates $compareCandidates --group-key pair_id --device cuda --out_dir (Join-Path $trackRoot "paired_mode_compare")
}

Write-Host "`nAll refresh jobs completed. Outputs:"
Write-Host "  $root"
