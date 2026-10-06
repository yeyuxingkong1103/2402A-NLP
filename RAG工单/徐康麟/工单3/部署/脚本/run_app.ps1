# =============================================================================
# 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件：部署/脚本/run_app.ps1 —— 统一启动入口（Windows 本机 / 算力云有 pwsh 时通用）
#
# 用途：一条命令完成「环境与索引预检 → 端口检查 → 启动服务 → 健康检查 → 打印日志与停止方式」，
#       并把每一步的真实结论写成结构化 JSON Lines（部署/日志/deploy.log），禁止静默失败。
#
# 四种模式（业务逻辑同一套 研发/app/core，详见 设计/系统架构.md）：
#   fallback  纯标准库界面（http.server）—— **本机唯一可跑模式**（本机无 streamlit 且断网）
#   api       FastAPI 服务（uvicorn app.main:app，**必须 --app-dir 研发**）—— 本机实测 /api/health 200
#   streamlit 真 Streamlit 应用 —— **本机不可跑**（缺依赖）；算力云装好 streamlit 后使用
#   check     只做预检，不启动（部署前自检用）
#
# 用法（工作目录 = E:\gao6gongdan\工单3）：
#   pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode check
#   pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode fallback -Port 8600
#   pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode api -Port 8600 -Background
#   pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Stop -Port 8600
#   pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode streamlit -EnvFile 部署/配置/config.example.env
#
# 退出码：0=成功；1=运行期失败；2=预检失败（依赖/索引/语料缺失）；3=端口被占用；
#         4=所选模式在本机不可用（缺依赖，未加 -AllowDegraded）；127=解释器缺失（与 run_py.ps1 一致）。
# =============================================================================
[CmdletBinding()]
param(
    [ValidateSet('fallback', 'api', 'streamlit', 'check')]
    [string] $Mode = 'fallback',
    [int]    $Port = 0,
    [string] $ServerHost = '',
    [string] $EnvFile = '',
    [string] $Python = '',
    [int]    $HealthTimeoutSec = 90,
    [switch] $NoWarmup,
    [switch] $Background,
    [switch] $Stop,
    [switch] $AllowDegraded
)

$ErrorActionPreference = 'Stop'
$WorkOrder = '人工智能NLP-RAG-PDF文档的表格解析及检索优化'

# 仓库根 = 本脚本上两级（部署/脚本 → 工单3）；**先切目录**，保证相对路径（研发/…）稳定
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $RepoRoot
# 统一 UTF-8：中文路径/中文输出在 Windows 控制台不乱码（与 run_py.ps1 同一约定）
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'   # 禁止向只读共享 venv 写 .pyc（红线：不得修改 工单1/2）

# -----------------------------------------------------------------------------
# 0. 结构化日志（自实现，写 JSON Lines；不依赖 loguru —— 本机 loguru 目录存在但 import 必失败）
#    每条含 ts/level/event/func/inputs/outputs/elapsed_ms/error(含堆栈)，禁止只写「开始/结束」。
# -----------------------------------------------------------------------------
$LogDir    = Join-Path $RepoRoot '部署\日志'
$DeployLog = Join-Path $LogDir 'deploy.log'
if (-not (Test-Path -LiteralPath $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

function Write-DeployLog {
    param(
        [Parameter(Mandatory)][string] $Event,
        [string] $Level = 'INFO',
        [string] $Func = '',
        $Inputs = $null,
        $Outputs = $null,
        $ElapsedMs = $null,
        $Error = $null
    )
    $record = [ordered]@{
        ts         = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss.fffzzz')
        level      = $Level
        event      = $Event
        func       = $Func
        work_order = $WorkOrder
        module     = '部署/脚本/run_app.ps1'
        pid        = $PID
        inputs     = $Inputs
        outputs    = $Outputs
        elapsed_ms = $ElapsedMs
        error      = $Error
    }
    Add-Content -LiteralPath $DeployLog -Value ($record | ConvertTo-Json -Compress -Depth 6) -Encoding utf8
    if ($Level -in @('ERROR', 'CRITICAL')) { Write-Host "[$Level] $($record.event) :: $($record.func) :: $($record.error.message)" }
}

# 函数入口/出口包装：成功写 func.exit（含耗时与输出摘要），异常写 func.error（含堆栈）后**原样抛出**。
function Invoke-Step {
    param(
        [Parameter(Mandatory)][string] $Name,
        [hashtable] $Inputs = @{},
        [Parameter(Mandatory)][scriptblock] $Action
    )
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-DeployLog -Event 'func.enter' -Func $Name -Inputs $Inputs
    try {
        $result = & $Action
        $sw.Stop()
        $summary = if ($result -is [hashtable]) { $result } elseif ($null -eq $result) { @{ done = $true } } else { @{ value = "$result" } }
        Write-DeployLog -Event 'func.exit' -Func $Name -Inputs $Inputs -Outputs $summary -ElapsedMs $sw.ElapsedMilliseconds
        return $result
    }
    catch {
        $sw.Stop()
        Write-DeployLog -Event 'func.error' -Level 'ERROR' -Func $Name -Inputs $Inputs -ElapsedMs $sw.ElapsedMilliseconds `
            -Error @{ type = $_.Exception.GetType().Name; message = $_.Exception.Message; stack = $_.ScriptStackTrace }
        throw
    }
}

# -----------------------------------------------------------------------------
# 1. 读取 env 文件（KEY=VALUE，跳过空行与 # 注释；只注入本进程，不落盘；空值 = 不设置）
# -----------------------------------------------------------------------------
function Import-EnvFile {
    param([string] $Path)
    if (-not $Path) { return @{} }
    $resolved = $Path
    if (-not [System.IO.Path]::IsPathRooted($resolved)) { $resolved = Join-Path $RepoRoot $Path }
    if (-not (Test-Path -LiteralPath $resolved)) { throw "env 文件不存在：$resolved" }
    $applied = @{}
    foreach ($raw in [System.IO.File]::ReadAllLines($resolved, [System.Text.Encoding]::UTF8)) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith('#')) { continue }
        $idx = $line.IndexOf('=')
        if ($idx -le 0) { continue }
        $key = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        if ($value -eq '') { continue }
        Set-Item -Path "Env:$key" -Value $value
        $applied[$key] = $value
    }
    return $applied
}

# -----------------------------------------------------------------------------
# 2. 解析 Python 解释器（与 run_py.ps1 同一套规则，禁止各自为政）
# -----------------------------------------------------------------------------
function Resolve-Python {
    param([string] $Override)
    if ($Override) { return @{ path = $Override; source = '参数 -Python' } }
    if ($env:RAG_SCHEDULER_PYTHON) {
        return @{ path = $env:RAG_SCHEDULER_PYTHON.Trim().Trim('"').Trim("'"); source = 'RAG_SCHEDULER_PYTHON' }
    }
    return @{ path = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'; source = '默认（工单1 venv，Python 3.11.15）' }
}

# -----------------------------------------------------------------------------
# 3. 预检：真实 import 依赖 / 索引 / 语料 / LLM 后端（**不用 find_spec 冒充可用**：
#    本机 loguru 目录存在但 import 必抛 ModuleNotFoundError，只能靠真实 import 判定）
# -----------------------------------------------------------------------------
$PreflightTemplate = @'
import json, os, sys, urllib.request, pathlib
sys.path.insert(0, str(pathlib.Path("研发").resolve()))
out = {"python": sys.version.split()[0], "executable": sys.executable, "cwd": os.getcwd()}
missing = []
for mod in ("numpy", "pymupdf", "jieba"):
    try:
        __import__(mod)
    except Exception as exc:  # noqa: BLE001 —— 显式收集，不静默
        missing.append({"module": mod, "error": f"{type(exc).__name__}: {exc}"})
out["missing_core"] = missing
mode_missing = []
_mode_deps = json.loads(os.environ.get("RAG_PREFLIGHT_MODE_DEPS", "[]"))
if isinstance(_mode_deps, str):      # 防御：单个模块名被 JSON 化成字符串时不得按字符遍历
    _mode_deps = [_mode_deps]
for mod in _mode_deps:
    try:
        __import__(mod)
    except Exception as exc:  # noqa: BLE001
        mode_missing.append({"module": mod, "error": f"{type(exc).__name__}: {exc}"})
out["missing_mode_deps"] = mode_missing
index_dir = pathlib.Path("研发/data/index")
manifests = sorted(index_dir.glob("*/index_manifest.json"))
out["index_manifests"] = [str(p) for p in manifests]
out["chunks"] = None
out["vectors"] = None
out["dim"] = None
out["bm25_vocab"] = None
out["embed_model"] = None
out["index_dir_used"] = None
out["index_files"] = []
out["index_created_at"] = None
if manifests:
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    embedding = manifest.get("embedding") or {}
    bm25 = manifest.get("bm25") or {}
    out["chunks"] = manifest.get("total_chunks") or embedding.get("count") or bm25.get("count")
    out["vectors"] = embedding.get("count")
    out["dim"] = embedding.get("dim")
    out["bm25_vocab"] = bm25.get("vocab_size")
    out["embed_model"] = embedding.get("model")
    out["index_dir_used"] = str(manifests[-1].parent)
    out["index_created_at"] = manifest.get("created_at")
    out["index_files"] = [f.get("file_name") for f in (manifest.get("files") or [])]
raw = pathlib.Path("研发/data/raw")
out["raw_pdfs"] = sorted(p.name for p in raw.glob("*.pdf")) if raw.is_dir() else []
base = os.environ.get("RAG_LLM__OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
try:
    with urllib.request.urlopen(base + "/api/tags", timeout=0.5) as resp:
        names = [r.get("name") for r in json.loads(resp.read().decode("utf-8") or "{}").get("models", [])]
    out["ollama"] = {"available": True, "models": names[:8]}
except Exception as exc:  # noqa: BLE001
    out["ollama"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
oa = os.environ.get("RAG_LLM__OPENAI_BASE_URL", "")
if oa:
    try:
        with urllib.request.urlopen(oa.rstrip("/") + "/models", timeout=0.5) as resp:
            out["openai_compat"] = {"available": True, "probe": "HTTP 200"}
    except Exception as exc:  # noqa: BLE001
        out["openai_compat"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
else:
    out["openai_compat"] = {"available": False, "error": "未配置 RAG_LLM__OPENAI_BASE_URL（本机默认不使用该后端）"}
print(json.dumps(out, ensure_ascii=False))
'@

function Invoke-Preflight {
    param([string] $PythonPath, [string] $ModeName)
    $deps = switch ($ModeName) {
        'api'       { @('fastapi', 'uvicorn') }
        'streamlit' { @('streamlit') }
        default     { @() }
    }
    # 手工拼 JSON 数组：`@('streamlit') | ConvertTo-Json` 会退化成字符串 "streamlit"，
#    Python 侧按字符遍历 → 误报 9 个缺失模块（本脚本研发期实测踩过，必须避免）。
    $env:RAG_PREFLIGHT_MODE_DEPS = '[' + ((@($deps) | ForEach-Object { '"' + $_ + '"' }) -join ',') + ']'
    $tmp = Join-Path $env:TEMP ("rag_preflight_{0}.py" -f ([guid]::NewGuid().ToString('N').Substring(0, 8)))
    [System.IO.File]::WriteAllText($tmp, $PreflightTemplate, (New-Object System.Text.UTF8Encoding($false)))
    try {
        $output = & $PythonPath $tmp 2>&1 | Out-String
        $code = $LASTEXITCODE
    }
    finally {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        Remove-Item Env:\RAG_PREFLIGHT_MODE_DEPS -ErrorAction SilentlyContinue
    }
    if ($code -ne 0) { throw "预检脚本执行失败（退出码 $code）：$($output.Trim())" }
    return ($output.Trim() | ConvertFrom-Json)
}

function Assert-Environment {
    param($Pre, [string] $ModeName)
    $problems = @()
    $coreMissing = @($Pre.missing_core)
    if ($coreMissing.Count -gt 0) {
        $problems += "核心依赖缺失：$((($coreMissing | ForEach-Object { "$($_.module)（$($_.error)）" }) -join '；'))"
    }
    $modeMissing = @($Pre.missing_mode_deps)
    if ($modeMissing.Count -gt 0) {
        $detail = (($modeMissing | ForEach-Object { "$($_.module)（$($_.error)）" }) -join '；')
        if ($ModeName -eq 'streamlit' -and $AllowDegraded) {
            # 显式降级（不是静默跳过）：本机缺 streamlit 是**实测事实**，降级到 fallback 并留痕
            Write-DeployLog -Event 'run_app.mode_degraded' -Level 'WARNING' -Func 'Assert-Environment' `
                -Inputs @{ mode = $ModeName; missing = $detail } -Outputs @{ fallback = 'fallback'; reason = $detail }
            Write-Host "⚠️ streamlit 不可用（$detail）→ 按 -AllowDegraded 显式降级为 fallback 模式" -ForegroundColor Yellow
            $script:Mode = 'fallback'
        }
        elseif ($ModeName -eq 'streamlit') {
            $problems += "模式 streamlit 依赖缺失：$detail —— 本机断网无法安装；本机请用 -Mode fallback（纯标准库，无需额外依赖）"
        }
        else {
            $problems += "模式 $ModeName 依赖缺失：$detail"
        }
    }
    if (@($Pre.index_manifests).Count -eq 0) {
        $problems += ("索引不存在（未找到 研发/data/index/*/index_manifest.json）：先执行 " +
                      "「pwsh -NoProfile -File run_py.ps1 研发/scripts/parse_corpus.py」，再执行 " +
                      "「pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py」")
    }
    if (@($Pre.raw_pdfs).Count -eq 0) {
        $problems += "语料为空：研发/data/raw 下没有 *.pdf（系统按目录自动发现，禁止硬编码文件名）"
    }
    return @($problems)
}

# -----------------------------------------------------------------------------
# 4. 端口占用检查（真实监听一次，不依赖 netstat 文本解析）
# -----------------------------------------------------------------------------
function Test-PortFree {
    param([int] $PortNumber)
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $PortNumber)
        $listener.Start()
        return $true
    }
    catch { return $false }
    finally { if ($listener) { $listener.Stop() } }
}

# -----------------------------------------------------------------------------
# 5. 健康检查（api/fallback → /api/health；streamlit → /_stcore/health）
# -----------------------------------------------------------------------------
function Wait-AppHealth {
    param([string] $Url, [int] $TimeoutSec)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $last = ''
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -TimeoutSec 3 -UseBasicParsing
            if ($resp.StatusCode -eq 200) {
                return @{ ok = $true; url = $Url; body = $resp.Content; wait_ms = $sw.ElapsedMilliseconds }
            }
            $last = "HTTP $($resp.StatusCode)"
        }
        catch { $last = $_.Exception.Message }
        Start-Sleep -Milliseconds 700
    }
    return @{ ok = $false; url = $Url; body = $last; wait_ms = $sw.ElapsedMilliseconds }
}

# -----------------------------------------------------------------------------
# 6. 主流程
# -----------------------------------------------------------------------------
try {
    $pyInfo = Resolve-Python -Override $Python
    if (-not (Test-Path -LiteralPath $pyInfo.path)) {
        Write-DeployLog -Event 'func.error' -Level 'ERROR' -Func 'Resolve-Python' `
            -Inputs @{ python = $pyInfo.path; source = $pyInfo.source } `
            -Error @{ type = 'FileNotFoundError'; message = '找不到 Python 解释器'; stack = '' }
        Write-Host "❌ 找不到 Python 解释器：$($pyInfo.path)（来源：$($pyInfo.source)）" -ForegroundColor Red
        Write-Host '   本机唯一可用解释器：E:\gao6gongdan\工单1\.venv\Scripts\python.exe（Python 3.11.15）' -ForegroundColor Yellow
        exit 127
    }

    # ---- 停止模式 ----
    if ($Stop) {
        $targetPort = if ($Port -gt 0) { $Port } else { 8600 }
        $pidFile = Join-Path $LogDir "run_app_$targetPort.pid.json"
        $null = Invoke-Step -Name 'run_app.Stop' -Inputs @{ port = $targetPort; pid_file = $pidFile } -Action {
            if (-not (Test-Path -LiteralPath $pidFile)) { throw "没有找到 PID 文件：$pidFile（该端口未经本脚本后台启动）" }
            $meta = Get-Content -LiteralPath $pidFile -Raw -Encoding utf8 | ConvertFrom-Json
            $proc = Get-Process -Id $meta.pid -ErrorAction SilentlyContinue
            if ($proc) { Stop-Process -Id $meta.pid -Force; Start-Sleep -Milliseconds 500 }
            Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
            @{ stopped_pid = $meta.pid; was_running = [bool]$proc; mode = $meta.mode }
        }
        Write-Host "✅ 已停止（端口 $targetPort）"
        exit 0
    }

    # ---- env 文件注入（失败即 exit 2：配置类错误与预检失败同一退出码语义）----
    if ($EnvFile) {
        try {
            $null = Invoke-Step -Name 'run_app.ImportEnvFile' -Inputs @{ env_file = $EnvFile } -Action {
                $applied = Import-EnvFile -Path $EnvFile
                @{ applied_keys = @($applied.Keys).Count; keys = (@($applied.Keys) -join ',') }
            }
        }
        catch {
            Write-Host "❌ env 文件处理失败：$($_.Exception.Message)" -ForegroundColor Red
            Write-Host "   示例文件：部署/配置/config.example.env（KEY=VALUE，空值表示不覆盖）" -ForegroundColor Yellow
            exit 2
        }
    }
    if ($ServerHost) { $env:RAG_SERVER__HOST = $ServerHost }
    if ($Port -gt 0)  { $env:RAG_SERVER__PORT = "$Port" }

    # ---- 预检 ----
    $modeName = $Mode
    $pre = Invoke-Step -Name 'run_app.Preflight' -Inputs @{ mode = $modeName; python = $pyInfo.path; index_dir = '研发/data/index' } -Action {
        Invoke-Preflight -PythonPath $pyInfo.path -ModeName $modeName
    }
    $problems = Assert-Environment -Pre $pre -ModeName $modeName
    $modeName = $script:Mode
    if (@($problems).Count -gt 0) {
        Write-DeployLog -Event 'run_app.preflight_failed' -Level 'ERROR' -Func 'Assert-Environment' `
            -Inputs @{ mode = $modeName } -Outputs @{ problems = @($problems) }
        Write-Host '❌ 预检未通过：' -ForegroundColor Red
        @($problems) | ForEach-Object { Write-Host "   - $_" -ForegroundColor Red }
        if ($modeName -eq 'streamlit') { exit 4 }
        exit 2
    }
    $ollamaText = if ($pre.ollama.available) { '可用' } else { "不可用（$($pre.ollama.error)）→ 生成按设计降级 extractive" }
    $openaiText = if ($pre.openai_compat.available) { '可用' } else { '未使用/不可用' }
    Write-Host ("✅ 预检通过：Python {0}；索引块 {1}（向量 {2} 条 / 维度 {3} / BM25 词表 {4}，嵌入模型 {5}）；语料 {6} 份 PDF {7}；Ollama {8}；OpenAI 兼容 {9}" -f `
        $pre.python, $pre.chunks, $pre.vectors, $pre.dim, $pre.bm25_vocab, $pre.embed_model,
        @($pre.raw_pdfs).Count, (@($pre.raw_pdfs) -join '/'), $ollamaText, $openaiText)

    if ($modeName -eq 'check') {
        Write-Host "ℹ️ -Mode check：仅预检，不启动服务（索引目录：$($pre.index_dir_used)）"
        exit 0
    }

    $portToUse = if ($Port -gt 0) { $Port } elseif ($modeName -eq 'streamlit') { 8501 } else { 8600 }
    $hostToUse = if ($ServerHost) { $ServerHost } elseif ($env:RAG_SERVER__HOST) { $env:RAG_SERVER__HOST } else { '127.0.0.1' }
    $healthHost = if ($hostToUse -in @('0.0.0.0', '::')) { '127.0.0.1' } else { $hostToUse }
    $healthPath = if ($modeName -eq 'streamlit') { '/_stcore/health' } else { '/api/health' }
    $healthUrl = "http://{0}:{1}{2}" -f $healthHost, $portToUse, $healthPath

    if (-not (Test-PortFree -PortNumber $portToUse)) {
        Write-DeployLog -Event 'run_app.port_busy' -Level 'ERROR' -Func 'Test-PortFree' -Inputs @{ port = $portToUse } `
            -Error @{ type = 'PortInUse'; message = "端口 $portToUse 已被占用"; stack = '' }
        Write-Host "❌ 端口 $portToUse 已被占用。换端口（-Port 8601）或先停止：" -ForegroundColor Red
        Write-Host "   pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Stop -Port $portToUse" -ForegroundColor Yellow
        exit 3
    }

    $entry = Join-Path $RepoRoot 'run_py.ps1'
    if ($modeName -eq 'fallback') {
        $appArgs = @('研发/app/ui/serve_fallback.py', '--host', $hostToUse, '--port', "$portToUse")
        if ($NoWarmup) { $appArgs += '--no-warmup' }
    }
    elseif ($modeName -eq 'api') {
        $appArgs = @('-m', 'uvicorn', 'app.main:app', '--app-dir', '研发', '--host', $hostToUse, '--port', "$portToUse")
    }
    else {
        $appArgs = @('-m', 'streamlit', 'run', '研发/app/ui/streamlit_app.py',
                     '--server.address', $hostToUse, '--server.port', "$portToUse", '--server.headless', 'true')
    }
    Write-Host ("🚀 启动模式 {0}：{1} {2}" -f $modeName, $entry, ($appArgs -join ' '))
    Write-Host ("   访问地址：http://{0}:{1}/　健康检查：{2}" -f $hostToUse, $portToUse, $healthUrl)

    # ---- 前台模式：直接执行并把退出码原样透传 ----
    if (-not $Background) {
        Write-DeployLog -Event 'run_app.start_foreground' -Inputs @{ mode = $modeName; port = $portToUse; args = ($appArgs -join ' ') }
        & pwsh -NoProfile -File $entry @appArgs
        $code = $LASTEXITCODE
        $level = if ($code -eq 0) { 'INFO' } else { 'ERROR' }
        Write-DeployLog -Event 'run_app.exit_foreground' -Level $level -Inputs @{ mode = $modeName; port = $portToUse } -Outputs @{ exit_code = $code }
        exit $code
    }

    # ---- 后台模式：独立进程 + PID 文件 + 健康检查（失败则打印输出日志尾部并非 0 退出）----
    $outLog  = Join-Path $LogDir "run_app_$portToUse.out.log"
    $errLog  = Join-Path $LogDir "run_app_$portToUse.err.log"
    $pidFile = Join-Path $LogDir "run_app_$portToUse.pid.json"
    # 输出编码（可移植性，2026-10-04）：Start-Process 的 -RedirectStandardOutput 是 **OS 级句柄重定向**，
    # 子进程按控制台 ANSI 代码页（本机 CP936）写文件 → 中文可读但 emoji 会退化成 '?'，且消费方必须按 GBK 读。
    # 因此这里让子进程**自己先把 [Console]::OutputEncoding 设为 UTF-8 无 BOM**，再执行启动命令：
    # 重定向文件变成确定的 UTF-8（`pwsh ... *> file` 本来就是 UTF-8，两种重定向从此一致）。
    $quotedArgs = (@($appArgs) | ForEach-Object { "'" + $_ + "'" }) -join ' '
    $childCommand = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(`$false); " +
                    "`$env:PYTHONIOENCODING='utf-8'; `$env:PYTHONUTF8='1'; `$env:PYTHONUNBUFFERED='1'; " +
                    "& '$entry' $quotedArgs"
    $procArgs = @('-NoProfile', '-Command', $childCommand)
    $proc = Start-Process -FilePath 'pwsh' -ArgumentList $procArgs -WorkingDirectory $RepoRoot -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    $meta = [ordered]@{
        pid = $proc.Id; mode = $modeName; port = $portToUse; host = $hostToUse
        started_at = (Get-Date).ToString('o'); python = $pyInfo.path; args = $appArgs
        run_id = $env:RAG_RUN_ID; out_log = $outLog; err_log = $errLog
    }
    [System.IO.File]::WriteAllText($pidFile, ($meta | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding($false)))
    Write-DeployLog -Event 'run_app.start_background' -Inputs @{ mode = $modeName; port = $portToUse; pid = $proc.Id } -Outputs $meta

    Write-Host ("⏳ 等待健康检查（最多 {0} s）…" -f $HealthTimeoutSec)
    $health = Invoke-Step -Name 'run_app.WaitAppHealth' -Inputs @{ url = $healthUrl; timeout_s = $HealthTimeoutSec } -Action {
        Wait-AppHealth -Url $healthUrl -TimeoutSec $HealthTimeoutSec
    }
    if (-not $health.ok) {
        Write-DeployLog -Event 'run_app.health_failed' -Level 'ERROR' -Func 'Wait-AppHealth' -Inputs @{ url = $healthUrl } `
            -Outputs @{ wait_ms = $health.wait_ms; last_error = "$($health.body)" } `
            -Error @{ type = 'HealthTimeout'; message = "健康检查未通过：$($health.body)"; stack = '' }
        Write-Host "❌ 健康检查未通过（等待 $($health.wait_ms) ms）：$($health.body)" -ForegroundColor Red
        Write-Host "   进程输出尾部（$outLog）："
        if (Test-Path -LiteralPath $outLog) { Get-Content -LiteralPath $outLog -Tail 25 -Encoding utf8 }
        Write-Host "   错误输出尾部（$errLog）："
        if (Test-Path -LiteralPath $errLog) { Get-Content -LiteralPath $errLog -Tail 25 -Encoding utf8 }
        Write-Host "   清理：pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Stop -Port $portToUse" -ForegroundColor Yellow
        exit 1
    }

    Write-Host ("✅ 健康检查通过（{0} ms，PID {1}）" -f $health.wait_ms, $proc.Id) -ForegroundColor Green
    $bodyText = "$($health.body)"
    try {
        $h = $bodyText | ConvertFrom-Json
        if ($h.retriever) {
            Write-Host ("   检索块 {0}（维度 {1}，BM25 词表 {2}）；生成后端 {3} / {4}" -f `
                $h.retriever.chunks, $h.retriever.dim, $h.retriever.bm25_vocab, $h.llm.backend, $h.llm.model)
        }
        else {
            Write-Host ("   健康响应：" + ($bodyText -replace '\s+', ' '))
        }
    }
    catch {
        Write-DeployLog -Event 'run_app.health_parse_warn' -Level 'WARNING' -Func 'run_app' -Outputs @{ raw = $bodyText.Substring(0, [Math]::Min(200, $bodyText.Length)) }
    }

    Write-Host ''
    Write-Host '--- 运维信息 ---'
    Write-Host ("   停止：pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Stop -Port {0}" -f $portToUse)
    Write-Host ("   PID 文件：{0}" -f $pidFile)
    Write-Host ("   进程输出：{0}" -f $outLog)
    Write-Host ("   应用日志：{0}" -f (Join-Path $LogDir 'app.log'))
    Write-Host ("   错误日志：{0}" -f (Join-Path $LogDir 'error.log'))
    Write-Host ("   链路日志：{0}" -f (Join-Path $LogDir 'rag_trace.jsonl'))
    Write-Host ("   部署日志：{0}" -f $DeployLog)
    Write-Host "   日志自检：pwsh -NoProfile -File run_py.ps1 部署/脚本/verify_logs.py"
    # 后台模式的出口留痕（与前台 run_app.exit_foreground 对称）：证明脚本自身已跑完全部步骤
    Write-DeployLog -Event 'run_app.exit_background' -Func 'run_app' `
        -Inputs @{ mode = $modeName; port = $portToUse } `
        -Outputs @{ pid = $proc.Id; health_ms = $health.wait_ms; exit_code = 0 } -ElapsedMs 0
    exit 0
}
catch {
    Write-DeployLog -Event 'run_app.fatal' -Level 'CRITICAL' -Func 'run_app' `
        -Error @{ type = $_.Exception.GetType().Name; message = $_.Exception.Message; stack = $_.ScriptStackTrace }
    Write-Host "❌ 启动流程异常：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace
    exit 1
}
