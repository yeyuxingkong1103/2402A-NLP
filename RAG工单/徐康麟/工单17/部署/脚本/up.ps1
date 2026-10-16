# =============================================================================
# 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
# 文件：部署/脚本/up.ps1 —— 启动整套服务（rag-api:8200 / rag-ui:8201 / redis）
#
# 用法（在 工单10/ 目录下）：
#     pwsh -NoProfile -File 部署/脚本/up.ps1                  # 起核心三件套
#     pwsh -NoProfile -File 部署/脚本/up.ps1 -WithMonitoring  # 额外起 prometheus
#     pwsh -NoProfile -File 部署/脚本/up.ps1 -NoBuild         # 不起镜像（已构建过）
#
# 做的事（顺序有讲究）：
#   ① 校验 compose 配置（`docker compose config -q`）—— 配置错早失败，别等拉镜像；
#   ② 起服务（-d 后台）；
#   ③ 等 rag-api 健康（轮询 docker inspect 的 Health.Status）；
#   ④ 打印端口探活结果（验收项②的真实证据）。
#
# 退出码：0 成功 / 1 失败 / 127 无 docker CLI
# =============================================================================
[CmdletBinding()]
param(
    [string]$EnvFile = '部署/配置/.env.example',
    [switch]$WithMonitoring,
    [switch]$NoBuild,
    [int]$WaitSeconds = 300,
    [string]$ApiPort = '8200',
    [string]$UiPort = '8201'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$logDir = Join-Path $repoRoot '部署\日志'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logFile = Join-Path $logDir "启动-$stamp.log"

function Write-Both([string]$Text) {
    Write-Host $Text
    Add-Content -LiteralPath $logFile -Value $Text -Encoding utf8
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Both "[错误] 找不到 docker CLI"
    exit 127
}

$compose = @('compose', '--env-file', $EnvFile, '-f', '部署/docker/docker-compose.yml')
$profile = @()
if ($WithMonitoring) { $profile = @('--profile', 'monitoring') }

Write-Both "===== 启动服务 ====="
Write-Both "时间     ：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Both "环境文件 ：$EnvFile"
Write-Both "监控栈   ：$(if ($WithMonitoring) { '启用 prometheus' } else { '不启用' })"

# ① 配置校验
Write-Both "`n[1/4] 校验 compose 配置（docker compose config -q）"
& docker @compose @profile config -q 2>&1 | Tee-Object -FilePath $logFile -Append
if ($LASTEXITCODE -ne 0) {
    Write-Both "[FAIL] compose 配置不合法，已终止（详见 部署/日志/）"
    exit 1
}
Write-Both "[OK] compose 配置校验通过"

# ② 起服务
Write-Both "`n[2/4] 启动容器"
$upArgs = @('up', '-d')
if (-not $NoBuild) { $upArgs += '--build' }
& docker @compose @profile @upArgs 2>&1 | Tee-Object -FilePath $logFile -Append
if ($LASTEXITCODE -ne 0) {
    Write-Both "[FAIL] 启动失败（常见原因：Docker 管道权限 / 端口占用 / 镜像拉取超时）"
    exit 1
}

# ③ 等健康
Write-Both "`n[3/4] 等待 rag-api 健康（最长 $WaitSeconds 秒）"
$deadline = (Get-Date).AddSeconds($WaitSeconds)
$healthy = $false
while ((Get-Date) -lt $deadline) {
    $status = (& docker inspect --format '{{.State.Health.Status}}' rag-api 2>$null)
    Write-Both "  $(Get-Date -Format 'HH:mm:ss') rag-api health = $status"
    if ($status -eq 'healthy') { $healthy = $true; break }
    Start-Sleep -Seconds 5
}

# ④ 端口探活（真实证据）
Write-Both "`n[4/4] 端口探活"
foreach ($p in @(@{n = 'rag-api'; port = $ApiPort; path = '/health' },
                 @{n = 'rag-ui'; port = $UiPort; path = '/health' })) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$($p.port)$($p.path)" -TimeoutSec 10 -UseBasicParsing
        Write-Both ("  [OK]   {0} http://127.0.0.1:{1}{2} → HTTP {3}（{4} 字节）" -f $p.n, $p.port, $p.path, $r.StatusCode, $r.RawContentLength)
    } catch {
        Write-Both ("  [FAIL] {0} http://127.0.0.1:{1}{2} → {3}" -f $p.n, $p.port, $p.path, $_.Exception.Message)
    }
}

& docker @compose @profile ps 2>&1 | Tee-Object -FilePath $logFile -Append
Write-Both "`n日志：$logFile"
if ($healthy) { Write-Both '[OK] 启动完成，rag-api 已 healthy'; exit 0 }
Write-Both '[WARN] rag-api 未在时限内 healthy，请查看 部署/脚本/logs.ps1 的日志'
exit 1
