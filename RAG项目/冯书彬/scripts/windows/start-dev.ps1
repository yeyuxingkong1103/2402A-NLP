param(
    [switch]$SkipDependencies,
    [switch]$SkipMigrations,
    [int]$ApiPort = 8010
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$runtimeDir = Join-Path $repoRoot "runtime"
$logDir = Join-Path $runtimeDir "logs"
$pidDir = Join-Path $runtimeDir "pids"

function Test-CommandExists([string]$CommandName) {
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "找不到命令：$CommandName，请先安装并加入 PATH。"
    }
}

function Wait-TcpPort([string]$HostName, [int]$Port, [int]$TimeoutSeconds = 120) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            $client.Connect($HostName, $Port)
            if ($client.Connected) {
                $client.Close()
                return
            }
        } catch {
            Start-Sleep -Seconds 2
        } finally {
            $client.Dispose()
        }
    } while ((Get-Date) -lt $deadline)
    throw "等待 $HostName`:$Port 超时。"
}

function Stop-ExistingProcess([string]$PidFile) {
    if (-not (Test-Path $PidFile)) {
        return
    }
    $existingPid = [int](Get-Content $PidFile -Raw).Trim()
    $process = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Stop-Process -Id $existingPid -Force
    }
    Remove-Item $PidFile -Force
}

Test-CommandExists "docker"
Test-CommandExists "python"
New-Item -ItemType Directory -Force -Path $logDir, $pidDir | Out-Null
Set-Location $repoRoot

if (-not $SkipDependencies) {
    Write-Host "启动 MySQL、Redis、etcd、MinIO 和 Milvus..."
    docker compose up -d mysql redis etcd minio milvus
    Wait-TcpPort "127.0.0.1" 3306
    Wait-TcpPort "127.0.0.1" 6380
    Wait-TcpPort "127.0.0.1" 19530
}

if (-not $SkipMigrations) {
    Write-Host "执行数据库迁移..."
    python -m alembic upgrade head
}

Stop-ExistingProcess (Join-Path $pidDir "api.pid")
Stop-ExistingProcess (Join-Path $pidDir "celery.pid")

$apiLog = Join-Path $logDir "api.log"
$apiErrorLog = Join-Path $logDir "api-error.log"
$celeryLog = Join-Path $logDir "celery.log"
$celeryErrorLog = Join-Path $logDir "celery-error.log"

Write-Host "启动 API：http://127.0.0.1:$ApiPort"
$apiProcess = Start-Process -FilePath "python" `
    -ArgumentList @("-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "$ApiPort") `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $apiLog `
    -RedirectStandardError $apiErrorLog `
    -PassThru
$apiProcess.Id | Set-Content (Join-Path $pidDir "api.pid")

Write-Host "启动 Celery worker..."
$celeryProcess = Start-Process -FilePath "python" `
    -ArgumentList @("-m", "celery", "-A", "backend.app.workers.celery_app.celery_app", "worker", "--loglevel=info", "--pool=solo") `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $celeryLog `
    -RedirectStandardError $celeryErrorLog `
    -PassThru
$celeryProcess.Id | Set-Content (Join-Path $pidDir "celery.pid")

Write-Host "启动完成。日志目录：$logDir"
