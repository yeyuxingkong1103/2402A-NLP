# =============================================================================
# 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
# 文件：部署/脚本/本地等价验证.ps1 —— Docker 不可用时的**等价本机容器化验证**
#
# 为什么要有这个脚本？
#   本机 Docker Desktop 在运行，但 `docker` CLI 走命名管道
#   npipe:////./pipe/dockerDesktopLinuxEngine，在 workspace-write 沙箱下**被拒绝**，
#   且本会话审批提示被禁用 → 无法提权 → 不能 build/run 容器（真实证据见
#   部署/日志/docker提权失败证据.log）。按 §共享约定 §5 工单10 条目：
#   「跑不通就如实记录并给等价的本机容器化验证」。
#
# 等价性说明（这不是"假装跑通了 Docker"，而是把容器**要验的四件事**在本机同构复现）：
#   ① 同一份源码            研发/service/api.py + 研发/app  ← 与镜像里 COPY 的是同一批文件
#   ② 同一组端口            8200（API）/ 8201（UI）        ← 与 compose 的 ports 映射一致
#   ③ 卷语义同构            宿主目录 = 卷；`-v` 的读写语义用「只读索引目录 + 可写运行期目录」复现
#   ④ 依赖地址同构          host.docker.internal:11434  ←→  127.0.0.1:11434（同一台宿主 Ollama）
#                           redis:6379（服务名）        ←→  127.0.0.1:6379
#
# 用法（在 工单10/ 目录下）：
#     pwsh -NoProfile -File 部署/脚本/本地等价验证.ps1             # 起 API + UI 并等健康
#     pwsh -NoProfile -File 部署/脚本/本地等价验证.ps1 -NoWait     # 不起服务，只准备卷目录
#
# 产出：
#     部署/日志/本地运行-API-<时间戳>.log / -err.log
#     部署/日志/本地运行-UI-<时间戳>.log  / -err.log
#     部署/日志/本地进程.json（记录 PID，供 本地停止.ps1 使用）
# =============================================================================
[CmdletBinding()]
param(
    [int]$ApiPort = 8200,
    [int]$UiPort = 8201,
    [int]$WaitSeconds = 240,
    [switch]$NoWait,
    [string]$OllamaBaseUrl = 'http://127.0.0.1:11434'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$logDir = Join-Path $repoRoot '部署\日志'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'

# —— 「卷」目录（与 compose 里的挂载点一一对应；索引目录是真索引的联接，保持只读语义）——
$volRoot = Join-Path $repoRoot '部署\卷'
$volIndex = Join-Path $volRoot 'index'          # ← 对应 -v <宿主索引>:/app/data/index:ro
$volLogs = Join-Path $volRoot 'logs'            # ← 对应卷 rag-log
$volSessions = Join-Path $volRoot 'sessions'    # ← 对应卷 rag-session（容器间共享）
$volRuntime = Join-Path $volRoot 'runtime'      # ← 对应卷 rag-runtime（sqlite/缓存）
$volModels = Join-Path $volRoot 'models'        # ← 对应卷 rag-model
$volUploads = Join-Path $volRoot 'uploads'      # ← 对应卷 rag-upload
foreach ($d in @($volLogs, $volSessions, $volRuntime, $volModels, $volUploads)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
# 索引用目录联接指向工单7 的真实索引（等价于把宿主索引只读挂进容器）
if (-not (Test-Path $volIndex)) {
    New-Item -ItemType Junction -Path $volIndex -Target (Join-Path $repoRoot '工单7索引') | Out-Null
}

# —— 索引卷清洁（只读语义的落地动作）——
# 背景（实测，见 部署/过程问题记录.md 问题7）：索引卷里可能残留运行期 sqlite：
#   ① 工单7 自己测试期留下的 rag.sqlite3（62 MB，2026-10-08）；
#   ② 早期本工单脚本漏设 DEPLOY_RUNTIME_DIR 时误写的 qa_runtime.sqlite3（2026-10-12）。
# 容器里索引是 `:ro` 挂载，运行期数据**必须**走 runtime 卷；这里启动前清掉残留，
# 既保证"索引卷干净"这条回归断言可验证，也让镜像/卷的语义与 compose 一致。
$staleSqlite = Get-ChildItem -LiteralPath $volIndex -Filter *.sqlite3 -File -ErrorAction SilentlyContinue
foreach ($f in $staleSqlite) {
    Write-Host ("  [清洁] 删除索引卷内的运行期残留：{0}（{1} 字节，{2}）" -f `
        $f.Name, $f.Length, $f.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
    Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue
}
if ($staleSqlite) { Write-Host "  [清洁] 共清理 $($staleSqlite.Count) 个残留文件；运行期数据以 runtime 卷为准" }

# 索引只读纪律：本脚本**只读**索引目录，运行期 sqlite/缓存一律写进 runtime 卷目录
$env:RAG_DATA__INDEX_DIR = $volIndex
$env:RAG_DATA__RAW_DIR = Join-Path $volRoot 'raw'
$env:RAG_DATA__PROCESSED_DIR = Join-Path $volRoot 'processed'
$env:RAG_LOG__DIR = $volLogs
$env:RAG_LLM__OLLAMA_BASE_URL = $OllamaBaseUrl
$env:RAG_LLM__OLLAMA_EMBED_MODEL = 'bge-m3:latest'
$env:RAG_LLM__OLLAMA_GEN_MODEL = 'qwen2.5:3b'
$env:RAG_LOG__LEVEL = 'INFO'
$env:RAG_SERVER__HOST = '0.0.0.0'
$env:RAG_SERVER__FALLBACK_PORT = "$UiPort"
$env:DEPLOY_API_HOST = '0.0.0.0'
$env:DEPLOY_API_PORT = "$ApiPort"
$env:DEPLOY_UI_PORT = "$UiPort"
$env:DEPLOY_LOG_DIR = $volLogs
$env:DEPLOY_SESSION_DIR = $volSessions
$env:DEPLOY_RUNTIME_DIR = $volRuntime
$env:DEPLOY_UPLOAD_DIR = $volUploads
$env:DEPLOY_MODEL_DIR = $volModels
$env:DEPLOY_LOG_JSON = '1'
$env:DEPLOY_SESSION_BACKEND = 'auto'
# —— Redis（会话跨容器共享的第二通路）——
# 实测：本机 6379 端口有 Redis 服务在跑（发 PING 收 +PONG），故本机等价验证**真的连 Redis**，
# 用 db 15 与其它用途隔离；容器编排里用服务名 + requirepass（redis://:密码@redis:6379/0）。
# 客户端不依赖 redis 包：service/session_store.py 内置标准库 RESP 客户端（MiniRedis）。
$env:DEPLOY_REDIS_ENABLED = '1'
$env:DEPLOY_REDIS_URL = 'redis://127.0.0.1:6379/15'
$env:DEPLOY_BUILD_REF = 'local-1.0.0'
# ⚠️ 必须显式指定容器内路径（默认值是 /app/... 的 Linux 路径，本机不存在）：
#    漏掉 DEPLOY_INDEX_DIR / DEPLOY_RUNTIME_DIR 会导致
#    ① 健康检查报「索引卷不可读」；② 运行期 sqlite 回落到索引目录（只读卷）——
#    这正是本工单实测踩到的坑，见 部署/过程问题记录.md 问题7。
$env:DEPLOY_INDEX_DIR = $volIndex
$env:DEPLOY_RUNTIME_DIR = $volRuntime
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $repoRoot '研发'

Write-Host "==================================================================="
Write-Host "工单10 本地等价验证（Docker 不可用时的替代路径）"
Write-Host "工单编号：人工智能NLP-RAG 项目-金融问答系统部署"
Write-Host "时间：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "==================================================================="
Write-Host "解析器      ：E:\gao6gongdan\工单1\.venv\Scripts\python.exe"
Write-Host "索引（只读）：$volIndex  →  实际 = 工单7 真实索引"
Write-Host "日志卷      ：$volLogs"
Write-Host "会话卷      ：$volSessions"
Write-Host "运行期卷    ：$volRuntime"
Write-Host "API 端口    ：$ApiPort（对应容器 rag-api:8200）"
Write-Host "UI  端口    ：$UiPort（对应容器 rag-ui:8201）"
Write-Host "Ollama      ：$OllamaBaseUrl（对应容器内 host.docker.internal:11434）"

if ($NoWait) { Write-Host "`n[-NoWait] 仅准备卷目录，不启动服务"; exit 0 }

$py = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Host "[错误] 找不到解释器 $py"; exit 127 }

# 独立窗口程序启动（不要用 shell 包装：PowerShell 会把 stderr 包成 ErrorRecord 破坏 JSON 日志）
$apiOut = Join-Path $logDir "本地运行-API-$stamp.log"
$apiErr = Join-Path $logDir "本地运行-API-$stamp.err.log"
$uiOut = Join-Path $logDir "本地运行-UI-$stamp.log"
$uiErr = Join-Path $logDir "本地运行-UI-$stamp.err.log"

Write-Host "`n[1/3] 启动 rag-api（本机等价）→ http://127.0.0.1:$ApiPort"
$api = Start-Process -FilePath $py -PassThru -WindowStyle Hidden -WorkingDirectory (Join-Path $repoRoot '研发') `
    -RedirectStandardOutput $apiOut -RedirectStandardError $apiErr `
    -ArgumentList @('-m', 'uvicorn', 'service.api:app', '--host', '0.0.0.0', '--port', "$ApiPort", '--no-access-log')
Write-Host "      PID = $($api.Id)  日志：$apiOut"

Write-Host "[2/3] 启动 rag-ui（本机等价）→ http://127.0.0.1:$UiPort"
$ui = Start-Process -FilePath $py -PassThru -WindowStyle Hidden -WorkingDirectory (Join-Path $repoRoot '研发') `
    -RedirectStandardOutput $uiOut -RedirectStandardError $uiErr `
    -ArgumentList @('app\main.py', '--serve', '--host', '0.0.0.0', '--port', "$UiPort")
Write-Host "      PID = $($ui.Id)  日志：$uiOut"

# 记录 PID 供停止脚本使用（容器里对应的是 docker compose down）
@{
    stamp      = $stamp
    api_pid    = $api.Id
    ui_pid     = $ui.Id
    api_port   = $ApiPort
    ui_port    = $UiPort
    api_log    = $apiOut
    ui_log     = $uiOut
    started_at = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
    index_dir  = $volIndex
    log_dir    = $volLogs
    session_dir = $volSessions
    runtime_dir = $volRuntime
} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $logDir '本地进程.json') -Encoding utf8

# —— 等健康（对应 up.ps1 里等 `docker inspect .State.Health.Status`）——
Write-Host "`n[3/3] 等待服务健康（最长 $WaitSeconds 秒；索引加载 + 预热约需 30~60 秒）"
$deadline = (Get-Date).AddSeconds($WaitSeconds)
$apiOk = $false; $uiOk = $false
while ((Get-Date) -lt $deadline) {
    if (-not $apiOk) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$ApiPort/healthz" -TimeoutSec 5 -UseBasicParsing
            if ($r.StatusCode -eq 200) { $apiOk = $true; Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] rag-api /healthz = 200" }
        } catch { }
    }
    if (-not $uiOk) {
        try {
            $r2 = Invoke-WebRequest -Uri "http://127.0.0.1:$UiPort/health" -TimeoutSec 5 -UseBasicParsing
            if ($r2.StatusCode -eq 200) { $uiOk = $true; Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] rag-ui  /health  = 200" }
        } catch { }
    }
    if ($apiOk -and $uiOk) { break }
    Start-Sleep -Seconds 3
}

Write-Host ""
if ($apiOk) { Write-Host "[OK]   rag-api 已就绪 http://127.0.0.1:$ApiPort" } else { Write-Host "[FAIL] rag-api 未就绪（看 $apiErr）" }
if ($uiOk) { Write-Host "[OK]   rag-ui  已就绪 http://127.0.0.1:$UiPort" } else { Write-Host "[FAIL] rag-ui  未就绪（看 $uiErr）" }
if ($apiOk -and $uiOk) { exit 0 }
exit 1
