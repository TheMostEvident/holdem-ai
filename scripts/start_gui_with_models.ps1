# Launch the local GUI API with explicit or conventionally located trusted model files.
# Run from the project root:
#   .\scripts\start_gui_with_models.ps1
#   .\scripts\start_gui_with_models.ps1 -PpoCheckpoint ".\runs\my_ppo\best.pt" -TournamentCheckpoint ".\runs\my_expand\best.pt"
# Model files stay local, are not committed, and are validated at session creation.
[CmdletBinding()]
param(
    [string]$PpoCheckpoint = "",
    [string]$QBoostCheckpoint = "",
    [string]$TournamentCheckpoint = "",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [ValidateSet("sampled", "greedy")]
    [string]$PolicyMode = "sampled"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Resolve-TrustedCheckpoint {
    param(
        [string]$Provided,
        [string]$EnvironmentName,
        [string[]]$Candidates
    )
    $configured = [Environment]::GetEnvironmentVariable($EnvironmentName, "Process")
    $value = if ($Provided) { $Provided } elseif ($configured) { $configured } else { "" }
    # Legacy launches sometimes pin last.pt/latest.pt in a PowerShell env var.
    # Return to the best.pt from the SAME run; never silently select a different run.
    if ($value -and ((Split-Path -Leaf $value).ToLowerInvariant() -in @("last.pt", "latest.pt"))) {
        $parent = Split-Path -Parent $value
        if (-not $parent) { $parent = "." }
        $best = Join-Path -Path $parent -ChildPath "best.pt"
        if (-not (Test-Path -LiteralPath $best -PathType Leaf)) {
            throw "$EnvironmentName points to last.pt/latest.pt, but its sibling best.pt is missing: $best. Supply an existing -...Checkpoint best.pt path."
        }
        Write-Host "$EnvironmentName: switching last/latest checkpoint to sibling best.pt."
        $value = $best
    }
    if ($value) {
        $file = Get-Item -LiteralPath $value -ErrorAction Stop
        if ($file.PSIsContainer -or $file.Extension -ne ".pt") {
            throw "$EnvironmentName must point to an existing .pt file."
        }
        return $file.FullName
    }
    foreach ($candidate in $Candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return ""
}

$ppo = Resolve-TrustedCheckpoint -Provided $PpoCheckpoint -EnvironmentName "HOLDEM_AI_PPO_CHECKPOINT" -Candidates @(
    ".\runs\phase14_main\seed_52\ppo\best.pt",
    ".\runs\phase14_pilot\seed_41\ppo\best.pt",
    ".\runs\phase17_ppo_5h_seed_01\best.pt"
)
$qboost = Resolve-TrustedCheckpoint -Provided $QBoostCheckpoint -EnvironmentName "HOLDEM_AI_QBOOST_CHECKPOINT" -Candidates @(
    ".\runs\phase14_pilot\seed_41\qboost\best.pt"
)
$tournament = Resolve-TrustedCheckpoint -Provided $TournamentCheckpoint -EnvironmentName "HOLDEM_AI_TOURNAMENT_CHECKPOINT" -Candidates @(
    ".\runs\expand_phase03_pilot\best.pt"
)

$env:HOLDEM_AI_PPO_CHECKPOINT = $ppo
$env:HOLDEM_AI_QBOOST_CHECKPOINT = $qboost
$env:HOLDEM_AI_TOURNAMENT_CHECKPOINT = $tournament
$env:HOLDEM_AI_GUI_DEVICE = $Device
$env:HOLDEM_AI_GUI_POLICY_MODE = $PolicyMode
$env:HOLDEM_AI_TOURNAMENT_DEVICE = $Device
$env:HOLDEM_AI_TOURNAMENT_POLICY_MODE = $PolicyMode

Write-Host "HUNL PPO checkpoint: $(if ($ppo) {  $ppo  } else { 'not found - model can be selected but cannot start' })"
Write-Host "HUNL Q-Boost checkpoint: $(if ($qboost) {  $qboost  } else { 'not found - model can be selected but cannot start' })"
Write-Host "Tournament PPO checkpoint: $(if ($tournament) {  $tournament  } else { 'not found - model can be selected but cannot start' })"
Write-Host "Mode: $PolicyMode / Device: $Device"
Write-Host "Checkpoint existence does not guarantee compatibility. Real weights are checked only when a trained game starts."
Write-Host "To use a different checkpoint, stop the server and relaunch with the appropriate *Checkpoint argument. last.pt/latest.pt inputs are redirected to sibling best.pt."
uv run --extra gui --extra gpu python -m backend.main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
