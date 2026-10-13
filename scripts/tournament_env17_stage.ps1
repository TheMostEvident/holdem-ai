# Tournament ENV 1.7: stage ONLY files explicitly listed in the manifest.
# Does not commit, push, or stage data/checkpoints/model artifacts.
# Append-only .gitignore hardening preserves existing custom rules.
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$manifest = Join-Path $PSScriptRoot 'tournament_env17_manifest.txt'
if (-not (Test-Path $manifest)) { throw 'Release manifest missing' }
Push-Location $root
try {
    & git rev-parse --show-toplevel | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Not inside Git repository' }
    if (-not (Test-Path 'frontend\package.json')) { throw 'Not a holdem-ai project root' }
    $ignorePath = Join-Path $root '.gitignore'
    $additions = @(
        '.local/',
        'runs/',
        'checkpoints/',
        '*.pt',
        '*.pth',
        '*.ckpt',
        '*.safetensors',
        '*.sqlite3',
        '*.sqlite3-wal',
        '*.sqlite3-shm',
        '*.jsonl',
        '*.ndjson',
        '.env.*',
        '!.env.example'
    )
    $existing = if (Test-Path -LiteralPath $ignorePath) {
        [System.IO.File]::ReadAllText($ignorePath)
    } else { '' }
    $existingLines = @($existing -split "`r?`n")
    $missing = @($additions | Where-Object { $existingLines -cnotcontains $_ })
    if ($missing.Count -gt 0) {
        Write-Host 'These patterns will be appended to .gitignore:' -ForegroundColor Yellow
        $missing | ForEach-Object { Write-Host "  $_" }
        if (-not $DryRun) {
            $suffix = "`n# Tournament ENV 1.7 - local checkpoints, exports and journals`n" + ($missing -join "`n") + "`n"
            [System.IO.File]::AppendAllText($ignorePath, $suffix, [System.Text.UTF8Encoding]::new($false))
        }
    }
    $paths = Get-Content -LiteralPath $manifest -Encoding UTF8 | Where-Object { $_ -and -not $_.StartsWith('#') }
    $count = 0
    foreach ($relative in $paths) {
        if ($relative -eq '.gitignore' -and $DryRun) { continue }
        $absolute = Join-Path $root ($relative.Replace('/', [System.IO.Path]::DirectorySeparatorChar))
        if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) {
            Write-Warning "Not present; skip: $relative"
            continue
        }
        if ($DryRun) {
            Write-Host "WOULD STAGE $relative"
            continue
        }
        & git add -- $relative
        if ($LASTEXITCODE -ne 0) { throw "git add failed: $relative" }
        $count += 1
    }
    if ($DryRun) {
        Write-Host "`nDry-run only. No Git index or file content changed." -ForegroundColor Green
    } else {
        Write-Host "`nSelected files submitted to Git index: $count" -ForegroundColor Green
        & git diff --cached --check
        if ($LASTEXITCODE -ne 0) { throw 'Staged diff has whitespace problems' }
        & uv run --extra dev --extra gui python -m scripts.tournament_env17_release_audit --strict
        if ($LASTEXITCODE -ne 0) { throw 'Privacy audit blocked the release' }
        & git diff --cached --stat
        Write-Host 'Review with: git diff --cached --name-status' -ForegroundColor Yellow
        Write-Host 'No commit or push was performed.' -ForegroundColor Yellow
    }
} finally {
    Pop-Location
}
