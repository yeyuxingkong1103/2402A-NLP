# =============================================================================
# 工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单
# 文件：部署/脚本/起服务.ps1 —— 本机等价部署（Docker 不可用时的替代路径）
#
# 为什么不用 docker-compose？
#   本机 docker CLI 走命名管道 npipe:////./pipe/dockerDesktopLinuxEngine，
#   在 workspace-write 沙箱下被拒（实测 `permission denied ...`，退出码 1），
#   且本会话审批提示被禁用 → 无法提权。证据见 设计/环境事实.md §2.1。
#   ⇒ 按共享约定 §5 工单10 条目：「跑不通就如实记录并给等价的本机容器化验证」。
#
# 等价性（同一批要验的事，在本机同构复现）：
#   ① 同一份源码   研发/service/api.py + 研发/app   ← 与镜像 COPY 的是同一批文件
#   ② 同一端口     8200                              ← 与 compose ports 映射一致
#   ③ 卷语义同构   部署/卷/{logs,sessions,runtime,uploads} = 具名卷；
#                  索引目录**只读引用** 工单7 的真实索引（对应 -v ...:ro）
#   ④ 依赖地址同构 127.0.0.1:11434 ←→ 容器内 host.docker.internal:11434
#                  127.0.0.1:6379  ←→ 容器内 redis:6379（本机等价用 db 15 隔离）
#
# 用法（工作目录 = E:\gao6gongdan\工单17）：
#     pwsh -NoProfile -File 部署/脚本/起服务.ps1                       # 起"优化前"口径（PERF_ENABLED=0）
#     pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 1        # 起"优化后"口径
#     pwsh -NoProfile -File 部署/脚本/起服务.ps1 -NoWait               # 只准备卷，不起服务
#
# 产出：部署/日志/本地运行-API-<时间戳>.log / .err.log
#       部署/日志/本地进程.json（记录 PID，供 本地停止.ps1 使用）
# =============================================================================
[CmdletBinding()]
param(
    [int]$ApiPort = 8200,
    [int]$WaitSeconds = 300,
    [switch]$NoWait,
    [ValidateSet('0', '1')][string]$PerfEnabled = '0',
    [string]$Tag = '',
    [string]$OllamaBaseUrl = 'http://127.0.0.1:11434',
    [string]$IndexDir = 'E:\gao6gongdan\工单7\研发\data\index',
    [string]$RawDir = 'E:\gao6gongdan\工单7\研发\data\raw'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$logDir = Join-Path $repoRoot '部署\日志'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (-not $Tag) { $Tag = if ($PerfEnabled -eq '1') { '优化后' } else { '优化前' } }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'

# —— 「卷」目录（与 compose 的挂载点一一对应）——
$volRoot = Join-Path $repoRoot '部署\卷'
$volLogs = Join-Path $volRoot 'logs'
$volSessions = Join-Path $volRoot 'sessions'
$volRuntime = Join-Path $volRoot 'runtime'
$volUploads = Join-Path $volRoot 'uploads'
foreach ($d in @($volLogs, $volSessions, $volRuntime, $volUploads)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

if (-not (Test-Path $IndexDir)) { Write-Host "[错误] 索引目录不存在：$IndexDir"; exit 127 }

# —— 环境变量（阈值全部配置化，代码里不写死）——
$env:RAG_DATA__INDEX_DIR = $IndexDir
$env:RAG_DATA__RAW_DIR = $RawDir
$env:RAG_DATA__PROCESSED_DIR = (Join-Path $IndexDir '..\processed')
$env:RAG_LOG__DIR = $volLogs
$env:RAG_LLM__OLLAMA_BASE_URL = $OllamaBaseUrl
$env:RAG_LLM__OLLAMA_EMBED_MODEL = 'bge-m3:latest'
$env:RAG_LLM__OLLAMA_GEN_MODEL = 'qwen2.5:3b'
$env:RAG_LOG__LEVEL = 'INFO'
$env:DEPLOY_API_HOST = '127.0.0.1'
$env:DEPLOY_API_PORT = "$ApiPort"
$env:DEPLOY_LOG_DIR = $volLogs
$env:DEPLOY_SESSION_DIR = $volSessions
$env:DEPLOY_RUNTIME_DIR = $volRuntime
$env:DEPLOY_UPLOAD_DIR = $volUploads
$env:DEPLOY_INDEX_DIR = $IndexDir
$env:DEPLOY_LOG_JSON = '1'
$env:DEPLOY_SESSION_BACKEND = 'auto'
$env:DEPLOY_REDIS_ENABLED = '1'
$env:DEPLOY_REDIS_URL = 'redis://127.0.0.1:6379/15'
$env:DEPLOY_BUILD_REF = "local-17-$Tag"
$env:DEPLOY_WARMUP = '1'
$env:DEPLOY_WARMUP_WITH_GENERATE = '0'
# —— 工单17 性能开关（PERF_ENABLED=0 时行为与工单10 基座逐位一致）——
$env:PERF_ENABLED = $PerfEnabled
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $repoRoot '研发'
$env:PYTHONUNBUFFERED = '1'

Write-Host "==================================================================="
Write-Host "工单17 本机等价部署（Docker 不可用时的替代路径）"
Write-Host "工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"
Write-Host "时间：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')   标签：$Tag"
Write-Host "==================================================================="
Write-Host "解释器        ：E:\gao6gongdan\工单1\.venv\Scripts\python.exe"
Write-Host "索引（只读）  ：$IndexDir"
Write-Host "日志卷        ：$volLogs"
Write-Host "会话卷        ：$volSessions"
Write-Host "运行期卷      ：$volRuntime"
Write-Host "上传卷        ：$volUploads"
Write-Host "API 端口      ：$ApiPort"
Write-Host "PERF_ENABLED  ：$PerfEnabled  （0=基座行为 / 1=优化后）"
Write-Host "Ollama        ：$OllamaBaseUrl"

if ($NoWait) { Write-Host "`n[-NoWait] 仅准备卷目录，不启动服务"; exit 0 }

$py = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Host "[错误] 找不到解释器 $py"; exit 127 }

$apiOut = Join-Path $logDir "本地运行-API-$Tag-$stamp.log"
$apiErr = Join-Path $logDir "本地运行-API-$Tag-$stamp.err.log"

# 会话 python 基线（**必须在起服务之前记录**，供停止脚本判断"哪些是本次新增的"）
# ⚠️ 每次都**重写**：会话过程中宿主的 python 辅助进程可能换过 PID，旧基线会变成过期集合，
#    停止脚本就会把当前会话进程误判为"新增"并杀掉（实测踩过，宿主的辅助进程被误杀后又被重启）。
$baselineFile = Join-Path $logDir 'python基线.json'
@{
    recorded_at = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
    note        = '会话基线 python PID（起服务前记录；停止脚本据此做差集，绝不误杀）'
    pids        = @(Get-Process python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $baselineFile -Encoding utf8
Write-Host "`n[0/2] 已记录 python 基线（起服务前）：$baselineFile"

Write-Host "`n[1/2] 启动 rag-api（本机等价）→ http://127.0.0.1:$ApiPort"
$api = Start-Process -FilePath $py -PassThru -WindowStyle Hidden -WorkingDirectory (Join-Path $repoRoot '研发') `
    -RedirectStandardOutput $apiOut -RedirectStandardError $apiErr `
    -ArgumentList @('-m', 'uvicorn', 'service.api:app', '--host', '127.0.0.1', '--port', "$ApiPort", '--no-access-log')
Write-Host "      PID = $($api.Id)  日志：$apiOut"

# ⚠️ 实测（本机）：Start-Process 出来的 PID 是个 1 MB 的**启动器 stub**，
#    真正的 uvicorn 服务进程是它的子进程，PID 只能从 /healthz 的 `pid` 字段拿。
#    停止脚本必须两个都杀，否则会留下 2.4 GB 的孤儿进程（工单9 的教训）。
$realPid = $null
for ($i = 0; $i -lt 20 -and -not $realPid; $i++) {
    try {
        $hz = (Invoke-WebRequest -Uri "http://127.0.0.1:$ApiPort/healthz" -TimeoutSec 3 -UseBasicParsing).Content | ConvertFrom-Json
        $realPid = [int]$hz.pid
    } catch { Start-Sleep -Milliseconds 500 }
}
if ($realPid) { Write-Host "      真实服务进程 PID = $realPid（启动器 PID = $($api.Id)）" }

@{
    stamp        = $stamp
    tag          = $Tag
    perf_enabled = $PerfEnabled
    api_pid      = $api.Id
    api_real_pid = $realPid
    ui_pid       = $null
    api_port     = $ApiPort
    ui_port      = 0
    api_log      = $apiOut
    api_err_log  = $apiErr
    started_at   = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
    index_dir    = $IndexDir
    log_dir      = $volLogs
    session_dir  = $volSessions
    runtime_dir  = $volRuntime
} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $logDir '本地进程.json') -Encoding utf8

Write-Host "`n[2/2] 等待服务就绪（最长 $WaitSeconds 秒；索引加载 + 预热约需 30~90 秒）"
# ⚠️ 实测踩坑：PowerShell 的 ``Invoke-WebRequest -TimeoutSec`` 在本机偶发**不生效**
#    （服务已就绪却一直挂住，把整个启动脚本拖到工具超时）。
#    改用 ``curl.exe --max-time``：超时行为可靠，且不会因为等待而卡住整个脚本。
$deadline = (Get-Date).AddSeconds($WaitSeconds)
$hOk = $false; $rOk = $false
while ((Get-Date) -lt $deadline) {
    if (-not $hOk) {
        $code = (& curl.exe -s -o NUL -w '%{http_code}' --max-time 5 "http://127.0.0.1:$ApiPort/healthz" 2>$null)
        if ("$code" -eq '200') { $hOk = $true; Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] /healthz = 200" }
    }
    if ($hOk -and -not $rOk) {
        $code2 = (& curl.exe -s -o NUL -w '%{http_code}' --max-time 25 "http://127.0.0.1:$ApiPort/readyz" 2>$null)
        if ("$code2" -eq '200') { $rOk = $true; Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] /readyz  = 200（索引就绪）" }
    }
    if ($hOk -and $rOk) { break }
    if ($api.HasExited) { Write-Host "[FAIL] API 进程已退出（退出码 $($api.ExitCode)），看 $apiErr"; exit 1 }
    Start-Sleep -Seconds 3
}

Write-Host ""
if ($hOk -and $rOk) {
    $p = Get-Process -Id $api.Id
    Write-Host ("[OK] rag-api 已就绪 http://127.0.0.1:$ApiPort  PID=$($api.Id)  私有内存={0} MB" -f [math]::Round($p.PrivateMemorySize64 / 1MB))
    Write-Host "[入口] 停止服务：pwsh -NoProfile -File 部署/脚本/本地停止.ps1"
    exit 0
}
Write-Host "[FAIL] rag-api 未就绪（healthz=$hOk readyz=$rOk）；看 $apiErr"
exit 1
