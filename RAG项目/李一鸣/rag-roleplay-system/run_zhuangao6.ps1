$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "D:\develop_tool1\anaconda3\envs\zhuangao6\python.exe"
$packageOverlay = Join-Path $projectRoot "conda_packages"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Conda zhuangao6 Python not found: $python"
}
if (-not (Test-Path -LiteralPath $packageOverlay)) {
    throw "D-drive package overlay not found: $packageOverlay"
}

Set-Location -LiteralPath $projectRoot
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = $packageOverlay
& $python run.py
