<#
.SYNOPSIS
  启动 Role RAG 服务（Windows PowerShell）。

.EXAMPLE
  .\start.ps1                 # 启动服务（默认 127.0.0.1:8020）
  .\start.ps1 -Ingest         # 先重建知识库再启动
  .\start.ps1 -Check          # 只做环境自检
  .\start.ps1 -Port 8030      # 指定端口
#>
param(
    [string]$Python = "D:\an\envs\rags_\python.exe",
    [int]$Port = 8020,
    [switch]$Ingest,
    [switch]$Check,
    [switch]$NoWarmup
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# 控制台按 UTF-8 输出，避免中文乱码
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONPATH = Join-Path $Root "src"
$env:ROLE_RAG_PORT = "$Port"
if ($NoWarmup) { $env:ROLE_RAG_WARMUP = "false" }

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "[FAIL] 找不到 Python：$Python" -ForegroundColor Red
    Write-Host "       请确认 conda 环境 rags_ 存在，或用 -Python 指定解释器。" -ForegroundColor Yellow
    exit 1
}

Write-Host "==> 环境自检" -ForegroundColor Cyan
& $Python (Join-Path $Root "tools\check_env.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] 环境自检未通过，已中止。" -ForegroundColor Red
    exit 1
}

if ($Check) { exit 0 }

if ($Ingest) {
    Write-Host "==> 重建并写入知识库" -ForegroundColor Cyan
    & $Python (Join-Path $Root "tools\ingest_cli.py") --all --recreate
    if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] 入库失败。" -ForegroundColor Red; exit 1 }
}

Write-Host "==> 启动服务：http://127.0.0.1:$Port/  （Ctrl+C 停止）" -ForegroundColor Green
& $Python -m uvicorn role_rag.api.app:app --host 127.0.0.1 --port $Port
