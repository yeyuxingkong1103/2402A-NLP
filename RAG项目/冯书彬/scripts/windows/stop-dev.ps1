param(
    [switch]$KeepDependencies
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$pidDir = Join-Path $repoRoot "runtime/pids"

function Stop-ManagedProcess([string]$Name) {
    $pidFile = Join-Path $pidDir "$Name.pid"
    if (-not (Test-Path $pidFile)) {
        return
    }
    $processId = [int](Get-Content $pidFile -Raw).Trim()
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Write-Host "停止 $Name（PID $processId）..."
        Stop-Process -Id $processId -Force
    }
    Remove-Item $pidFile -Force
}

Stop-ManagedProcess "celery"
Stop-ManagedProcess "api"

if (-not $KeepDependencies) {
    if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
        throw "找不到命令：docker，请先安装 Docker Desktop。"
    }
    Set-Location $repoRoot
    Write-Host "停止 Docker 依赖服务（保留数据卷）..."
    docker compose down
}

Write-Host "停止完成。"
