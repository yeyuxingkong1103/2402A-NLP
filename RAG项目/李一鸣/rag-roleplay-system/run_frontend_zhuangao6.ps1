$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "D:\develop_tool1\anaconda3\envs\zhuangao6\python.exe"
$frontendRoot = Join-Path $projectRoot "frontend"

if (-not (Test-Path -LiteralPath $python)) { throw "Conda zhuangao6 Python not found: $python" }
if (-not (Test-Path -LiteralPath $frontendRoot)) { throw "Frontend directory not found: $frontendRoot" }

Set-Location -LiteralPath $frontendRoot
& $python -s -m http.server 3100 --bind 127.0.0.1
