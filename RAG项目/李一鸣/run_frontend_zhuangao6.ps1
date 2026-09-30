$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
# 前端是静态 HTML，使用同一个环境的 Python 启动轻量 HTTP 服务。
$python = "D:\develop_tool1\anaconda3\envs\zhuangao6\python.exe"
$frontendRoot = Join-Path $projectRoot "frontend"

if (-not (Test-Path -LiteralPath $python)) { throw "Conda zhuangao6 Python not found: $python" }
if (-not (Test-Path -LiteralPath $frontendRoot)) { throw "Frontend directory not found: $frontendRoot" }

Set-Location -LiteralPath $frontendRoot
# 3100 端口用于避开 Milvus Attu 常用的 3000 端口。
& $python -s -m http.server 3100 --bind 127.0.0.1
