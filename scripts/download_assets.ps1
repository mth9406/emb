param(
    [Parameter(Mandatory = $true)]
    [string]$WorkspaceRoot
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$aria2 = Get-ChildItem -LiteralPath "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
    -Filter aria2c.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $aria2) {
    throw "aria2c is required for parallel archive downloads"
}
$env:ARIA2C = $aria2.FullName

python -u "$repoRoot/scripts/download_glami_dresses.py" `
    --data-root "$WorkspaceRoot/data/raw"
python -u "$repoRoot/scripts/download_checkpoints.py" `
    --ckpt-root "$WorkspaceRoot/ckpts"
