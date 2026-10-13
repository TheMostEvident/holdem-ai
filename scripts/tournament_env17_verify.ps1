# Tournament ENV Phase 1.7 full local verification, PowerShell 5.1+.
# NO Git commit, push, model training, or journal deletion.
param(
    [switch]$Quick,
    [switch]$WithAgentSmoke
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path (Join-Path $root 'frontend\package.json'))) {
    throw 'Run the script inside the holdem-ai project under scripts/.'
}
function Run-Checked([string]$Title, [scriptblock]$Command) {
    Write-Host "`n=== $Title ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Title FAILED (exit code $LASTEXITCODE)"
    }
}
Push-Location $root
try {
    if ($Quick) {
        Run-Checked 'Tournament integration tests' {
            & uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env17_integration.py
        }
    } else {
        Run-Checked 'Full Python regression' {
            & uv run --extra dev --extra gui --extra gpu python -m pytest -q
        }
    }
    if ($WithAgentSmoke) {
        if (-not $env:HOLDEM_AI_TOURNAMENT_CHECKPOINT) {
            throw 'Set HOLDEM_AI_TOURNAMENT_CHECKPOINT in this same PowerShell session.'
        }
        if (-not (Test-Path -LiteralPath $env:HOLDEM_AI_TOURNAMENT_CHECKPOINT -PathType Leaf)) {
            throw 'Tournament checkpoint path does not exist.'
        }
        Run-Checked 'Frozen multi-seat PPO inference smoke' {
            & uv run --extra dev --extra gui --extra gpu python -m scripts.tournament_env14_smoke
        }
    }
    Push-Location (Join-Path $root 'frontend')
    try {
        Run-Checked 'TypeScript typecheck' { & npm.cmd run typecheck }
        Run-Checked 'Vite production build' { & npm.cmd run build }
        Run-Checked 'Vitest' { & npm.cmd run test }
    } finally {
        Pop-Location
    }
    Run-Checked 'Git privacy audit (read only)' {
        & uv run --extra dev --extra gui python -m scripts.tournament_env17_release_audit
    }
    Write-Host "`nTournament ENV 1.7: LOCAL VERIFICATION PASSED" -ForegroundColor Green
    Write-Host 'Note: stage/commit/push not performed. Run the stage script only when ready.'
} finally {
    Pop-Location
}
