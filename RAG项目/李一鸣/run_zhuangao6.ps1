$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
# 固定使用 D 盘的 zhuangao6 环境，避免误用系统 Python 或 C 盘用户包。
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
# 以项目根目录为工作目录启动，确保 .env 和 app 包都能被找到。
& $python run.py
