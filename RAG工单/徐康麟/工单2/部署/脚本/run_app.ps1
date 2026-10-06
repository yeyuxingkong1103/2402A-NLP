<#
================================================================================
工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：部署 / 本机 Windows 启动脚本（Ollama + 标准库备用界面）
================================================================================
用途
    在**本机 Windows**上一键起服务或提问：解析 Python 解释器 → 前置检查
    （解释器 / 索引 / Ollama / 端口）→ 按解析后的配置调用 研发/app 的入口。

    本机可用的 LLM/嵌入服务是 127.0.0.1:11434 的 Ollama（qwen2.5:3b + bge-m3:latest）；
    本机**没有** vLLM/SGLang（Linux+GPU 才用，见同目录 run_vllm.sh / run_sglang.sh）。
    streamlit 在本机**不可用且无法安装**（断网），故默认走 app/ui/serve_fallback.py
    （纯标准库 http.server）；只有显式 -UseStreamlit 才尝试 Streamlit。

参数
    -Action         serve(默认) | ask | health | stats | conversations
    -Question       -Action ask 时的提问（中文或英文）
    -Port           -Action serve 的监听端口（默认 8600；0=随机空闲端口）
    -BindHost       监听地址（默认 127.0.0.1）
    -LlmBackend     auto | ollama(默认，本机) | openai(云端 vLLM/SGLang) | extractive
    -Model          生成模型名（空=用 config 默认 qwen2.5:3b）
    -EmbedModel     嵌入模型名（空=用 config 默认 bge-m3:latest）
    -OllamaBaseUrl  Ollama 地址（空=用 config 默认 http://127.0.0.1:11434）
    -Python         解释器路径（空→$env:RAG_SCHEDULER_PYTHON→工单1 .venv）
    -UseStreamlit   改用 streamlit run（本机预期失败：依赖不可用，会给出明确提示）
    -SkipPreflight  跳过前置检查（排障用）
    -DryRun         只打印将要执行的命令与环境变量，不真正执行
    -SmokeTimeoutSec 前置检查里 HTTP 探测的超时（默认 3 秒）

用法示例（工作目录任意，路径按脚本自身定位）
    pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action health
    pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action ask -Question "注册资本是多少？"
    pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action serve -Port 8600
    pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Action serve -LlmBackend extractive   # 无 LLM 兜底演示

退出码
    0 成功 | 2 环境/解释器问题 | 3 缺少入口文件 | 4 索引未就绪 | 5 依赖服务不可达
    6 端口被占用 | 其他 = 被调用的 python 进程退出码
================================================================================
#>
[CmdletBinding()]
param(
    [ValidateSet('serve', 'ask', 'health', 'stats', 'conversations')]
    [string]$Action = 'serve',

    [string]$Question = '',

    [ValidateRange(0, 65535)]
    [int]$Port = 8600,

    [string]$BindHost = '127.0.0.1',

    [ValidateSet('auto', 'ollama', 'openai', 'extractive')]
    [string]$LlmBackend = 'ollama',

    [string]$Model = '',
    [string]$EmbedModel = '',
    [string]$OllamaBaseUrl = '',
    [string]$Python = '',

    [switch]$UseStreamlit,
    [switch]$SkipPreflight,
    [switch]$DryRun,

    [int]$SmokeTimeoutSec = 3
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$OutputEncoding = [System.Text.Encoding]::UTF8

# ------------------------------------------------------------------ 基础定位
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$WorkspaceRoot = Split-Path $RepoRoot -Parent
$MainPy = Join-Path $RepoRoot '研发\app\main.py'
$ServeFallbackPy = Join-Path $RepoRoot '研发\app\ui\serve_fallback.py'
$StreamlitPy = Join-Path $RepoRoot '研发\app\ui\streamlit_app.py'
$IndexDir = Join-Path $RepoRoot '研发\data\index\bge-m3-1024'
$LogDir = Join-Path $RepoRoot '部署\日志'

function Write-Section([string]$Title) {
    Write-Host ''
    Write-Host "==== $Title ====" -ForegroundColor Cyan
}

function Write-Item([string]$Name, [string]$Value) {
    Write-Host ("  {0,-22} {1}" -f $Name, $Value)
}

function Fail([string]$Message, [int]$Code) {
    Write-Host "[失败] $Message" -ForegroundColor Red
    Write-Host "       退出码 $Code（0 成功 / 2 解释器 / 3 缺文件 / 4 索引 / 5 依赖服务 / 6 端口）"
    exit $Code
}

# ------------------------------------------------------------------ 解释器解析
function Resolve-PythonPath {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-Path -LiteralPath $Explicit)) {
            Fail "指定的解释器不存在：$Explicit" 2
        }
        return (Resolve-Path -LiteralPath $Explicit).Path
    }
    if ($env:RAG_SCHEDULER_PYTHON) {
        if (-not (Test-Path -LiteralPath $env:RAG_SCHEDULER_PYTHON)) {
            Fail "环境变量 RAG_SCHEDULER_PYTHON 指向的解释器不存在：$($env:RAG_SCHEDULER_PYTHON)" 2
        }
        return (Resolve-Path -LiteralPath $env:RAG_SCHEDULER_PYTHON).Path
    }
    $venv = Join-Path $WorkspaceRoot '工单1\.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venv) {
        return (Resolve-Path -LiteralPath $venv).Path
    }
    $fallback = Get-Command python -ErrorAction SilentlyContinue
    if ($fallback) { return $fallback.Source }
    Fail "未找到 Python 解释器。请用 -Python 指定，或设置 RAG_SCHEDULER_PYTHON（本机默认：$venv）" 2
}

Write-Section '工单2 部署 · 本机启动器（Ollama + 备用界面）'
Write-Item '仓库根目录' $RepoRoot
Write-Item 'Action' $Action
$PythonPath = Resolve-PythonPath -Explicit $Python
Write-Item 'Python 解释器' $PythonPath

if (-not (Test-Path -LiteralPath $MainPy)) { Fail "缺少入口文件：$MainPy" 3 }

# ------------------------------------------------------------------ 前置检查
$preflight = [ordered]@{}

if (-not $SkipPreflight) {
    Write-Section '前置检查'

    # 1) 索引就绪
    $vec = Join-Path $IndexDir 'vectors.npy'
    $meta = Join-Path $IndexDir 'vectors_meta.jsonl'
    $indexOk = (Test-Path -LiteralPath $vec) -and (Test-Path -LiteralPath $meta)
    $preflight['索引目录'] = $IndexDir
    $preflight['索引就绪'] = $indexOk
    Write-Item '索引目录' $IndexDir
    if ($indexOk) {
        $sizeMb = [math]::Round((Get-Item -LiteralPath $vec).Length / 1MB, 1)
        Write-Host "  '-> vectors.npy 存在（$sizeMb MB）" -ForegroundColor Green
    }
    else {
        Write-Host "  '-> 缺少 vectors.npy / vectors_meta.jsonl" -ForegroundColor Red
    }

    # 2) Ollama 可达（仅在需要 LLM/嵌入走 ollama 时检查）
    $needOllama = ($LlmBackend -in @('ollama', 'auto')) -or ($LlmBackend -ne 'extractive')
    $ollamaUrl = if ($OllamaBaseUrl) { $OllamaBaseUrl } else { 'http://127.0.0.1:11434' }
    if ($needOllama) {
        $ollamaOk = $false
        try {
            $resp = Invoke-WebRequest -Uri "$ollamaUrl/api/tags" -TimeoutSec $SmokeTimeoutSec -UseBasicParsing
            $ollamaOk = ($resp.StatusCode -eq 200)
            if ($ollamaOk) {
                $tags = ($resp.Content | ConvertFrom-Json).models | ForEach-Object { $_.name }
                Write-Host "  '-> Ollama 可达：$ollamaUrl（模型：$($tags -join ', ')）" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "  '-> Ollama 不可达：$ollamaUrl（$($_.Exception.Message)）" -ForegroundColor Yellow
        }
        $preflight['Ollama 可达'] = $ollamaOk
        if (-not $ollamaOk) {
            Write-Host '     提示：Ollama 未启动时可用 -LlmBackend extractive 走抽取式兜底路径。' -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "  '-> 使用 extractive 路径，跳过 Ollama 检查" -ForegroundColor DarkGray
    }

    # 3) serve：端口占用与入口文件
    if ($Action -eq 'serve') {
        $entry = if ($UseStreamlit) { $StreamlitPy } else { $ServeFallbackPy }
        if (-not (Test-Path -LiteralPath $entry)) { Fail "缺少界面入口文件：$entry" 3 }
        Write-Item '界面入口' $entry

        if ($UseStreamlit) {
            $streamlitOk = $false
            try {
                & $PythonPath -c "import streamlit, sys; sys.exit(0)" 2>$null
                $streamlitOk = ($LASTEXITCODE -eq 0)
            }
            catch { $streamlitOk = $false }
            if (-not $streamlitOk) {
                Fail "本机未安装 streamlit（环境事实 §2.2：依赖不可用且断网，无法安装）。请去掉 -UseStreamlit 走 serve_fallback.py；云端安装 streamlit 后可直接使用。" 2
            }
        }

        if ($Port -ne 0) {
            $portFree = $true
            try {
                $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse($BindHost), $Port)
                $listener.Start()
                $listener.Stop()
            }
            catch { $portFree = $false }
            if (-not $portFree) { Fail "端口 $Port 已被占用（$BindHost）。换 -Port 或先释放端口。" 6 }
            Write-Host "  '-> 端口 $Port 可用" -ForegroundColor Green
        }
    }

    if (-not $indexOk) {
        Fail "索引未就绪。请先构建：pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py --rebuild" 4
    }
}

# ------------------------------------------------------------------ 组装命令
Write-Section '解析后的配置（写入子进程环境变量）'
$childEnv = [ordered]@{}
$childEnv['RAG_LLM__BACKEND'] = $LlmBackend
if ($Model) { $childEnv['RAG_LLM__MODEL'] = $Model }
if ($OllamaBaseUrl) {
    $childEnv['RAG_LLM__OLLAMA_BASE_URL'] = $OllamaBaseUrl
    $childEnv['RAG_EMBEDDING__OLLAMA_BASE_URL'] = $OllamaBaseUrl
}
if ($EmbedModel) { $childEnv['RAG_EMBEDDING__OLLAMA_MODEL'] = $EmbedModel }
foreach ($kv in $childEnv.GetEnumerator()) {
    Write-Item $kv.Key $kv.Value
    Set-Item -Path "Env:$($kv.Key)" -Value $kv.Value
}
if ($childEnv.Count -eq 0) { Write-Host '  （无覆盖，全部使用 config.py 默认值）' }

$argList = @()
switch ($Action) {
    'serve' {
        if ($UseStreamlit) {
            $argList = @('-m', 'streamlit', 'run', $StreamlitPy, '--server.address', $BindHost, '--server.port', "$Port")
        }
        else {
            $argList = @($ServeFallbackPy, '--host', $BindHost, '--port', "$Port")
        }
    }
    'ask' {
        if (-not $Question) { Fail '-Action ask 需要 -Question "你的问题"' 2 }
        $argList = @($MainPy, 'ask', $Question)
    }
    'health' { $argList = @($MainPy, 'health') }
    'stats' { $argList = @($MainPy, 'stats') }
    'conversations' { $argList = @($MainPy, 'conversations') }
}

Write-Section '将要执行的命令'
$pretty = ($argList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
Write-Host "  `"$PythonPath`" $pretty"

if ($DryRun) {
    Write-Host ''
    Write-Host '[DryRun] 仅打印，不执行。' -ForegroundColor Yellow
    exit 0
}

# ------------------------------------------------------------------ 执行
Write-Section '开始执行（日志：部署\日志\{app.log,error.log,rag_trace.jsonl}）'
$procArgs = @{ FilePath = $PythonPath; ArgumentList = $argList; WorkingDirectory = $RepoRoot; NoNewWindow = $true }
if ($Action -ne 'serve') {
    & $PythonPath @argList
    $code = $LASTEXITCODE
}
else {
    Write-Host "界面地址：http://${BindHost}:$Port   （Ctrl+C 停止）" -ForegroundColor Green
    & $PythonPath @argList
    $code = $LASTEXITCODE
}

Write-Host ''
Write-Host "执行结束，退出码 $code" -ForegroundColor ($(if ($code -eq 0) { 'Green' } else { 'Yellow' }))
Write-Host "日志目录：$LogDir"
exit $code
