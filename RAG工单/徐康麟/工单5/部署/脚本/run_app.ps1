# =============================================================================
# 工单：人工智能NLP-RAG-图像内容解析及检索优化
# 文件：部署/脚本/run_app.ps1 —— 一键启动 / 自检入口（Windows 主用）
#
# 三种模式（都走仓库统一入口 run_py.ps1，保证解释器与 UTF-8 环境一致）：
#   -Mode api  （默认）FastAPI 服务   ：部署/脚本/serve_api.py    默认端口 8600
#                       端点：GET /health、POST /ask、POST /ask/stream、
#                             POST /retrieve、GET /questions、POST /eval、GET /docs
#   -Mode web          纯标准库中文 Web：研发/app/main.py --serve  默认端口 8123
#   -Mode cli          命令行单次问答  ：研发/app/main.py --ask "<问题>"
#
# 用法：
#   pwsh -NoProfile -File 部署/脚本/run_app.ps1                        # 前台起 FastAPI（8600）
#   pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode web              # 前台起中文 Web（8123）
#   pwsh -NoProfile -File 部署/脚本/run_app.ps1 -Mode cli -Question "武汉兴图新科电子股份有限公司注册资本是多少？"
#   pwsh -NoProfile -File 部署/脚本/run_app.ps1 -HealthCheckOnly       # 【验收自检】起服务→查 /health→停服务→退出码
#   pwsh -NoProfile -File 部署/脚本/run_app.ps1 -HealthCheckOnly -SmokeAll   # 自检 + 全部端点冒烟
#
# 退出码：0 全部检查通过 / 1 存在失败检查 / 3 索引缺失（E_INDEX_MISSING） / 127 解释器缺失
#
# 说明（实测结论，勿踩）：
#   · 本机 docker CLI 走命名管道 npipe:////./pipe/dockerDesktopLinuxEngine，沙箱默认拒绝；
#     本脚本不依赖 Docker（Docker 见 部署/docker/，属可选交付）。
#   · 本机禁止裸 pip install（落点是只读的 工单1 venv）：需要新依赖用 --target 研发/第三方库。
#   · -HealthCheckOnly 会写机器可读留痕 部署/日志/自检结果.json（供交付物核验引用）。
# =============================================================================
param(
    [ValidateSet('api', 'web', 'cli')][string]$Mode = 'api',
    [string]$BindHost = '127.0.0.1',
    [int]$Port = 0,
    [string]$Question = '',
    [switch]$HealthCheckOnly,
    [switch]$SmokeAll,
    [int]$TimeoutSec = 180,
    [switch]$NoStop,
    [switch]$NoWarmup
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)      # 部署/脚本 → 部署 → 工单4
Set-Location -LiteralPath $root

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

if ($Port -le 0) { $Port = if ($Mode -eq 'web') { 8123 } else { 8600 } }
$baseUrl = "http://{0}:{1}" -f $BindHost, $Port
$healthUrl = "$baseUrl/health"
$logDir = Join-Path $root '部署/日志'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$evidencePath = Join-Path $logDir '自检结果.json'

$checks = New-Object System.Collections.ArrayList
$startedByUs = $false
$serverJob = $null
$failed = 0

function Add-Check([string]$name, [bool]$ok, [string]$detail) {
    [void]$script:checks.Add([ordered]@{ name = $name; ok = [bool]$ok; detail = $detail })
    if (-not $ok) { $script:failed++ }
    $mark = if ($ok) { 'PASS' } else { 'FAIL' }
    Write-Host ("  [{0}] {1} —— {2}" -f $mark, $name, $detail)
}

# 返回 @{ status = <int>; text = <string>; json = <object>|$null; error = <string>|$null }；
# HTTP 错误码不抛异常（用 -SkipHttpErrorCheck），便于把 503/400 也如实记为检查结果。
function Invoke-Api([string]$method, [string]$url, $body, [int]$timeoutSec = 180) {
    $params = @{ Uri = $url; Method = $method; TimeoutSec = $timeoutSec
                 SkipHttpErrorCheck = $true; UseBasicParsing = $true }
    if ($null -ne $body) {
        $json = if ($body -is [string]) { $body } else { $body | ConvertTo-Json -Depth 6 -Compress }
        $params['Body'] = [Text.Encoding]::UTF8.GetBytes($json)
        $params['ContentType'] = 'application/json; charset=utf-8'
    }
    try {
        $resp = Invoke-WebRequest @params
        $text = [string]$resp.Content
        $obj = $null
        try { $obj = $text | ConvertFrom-Json } catch { $obj = $null }
        return @{ status = [int]$resp.StatusCode; text = $text; json = $obj; error = $null }
    } catch {
        return @{ status = 0; text = ''; json = $null; error = $_.Exception.Message }
    }
}

function Test-Ready {
    $r = Invoke-Api 'GET' $healthUrl $null 5
    return ($r.status -eq 200 -or $r.status -eq 503)      # 503 也算「已就绪但依赖不全」，由后续断言判定
}

function Start-App {
    $argList = if ($Mode -eq 'web') {
        @('研发/app/main.py', '--serve', '--host', $BindHost, '--port', "$Port")
    } else {
        @('部署/脚本/serve_api.py', '--host', $BindHost, '--port', "$Port")
    }
    if ($NoWarmup) { $argList += '--no-warmup' }
    Write-Host ("[启动] pwsh -NoProfile -File run_py.ps1 {0}" -f ($argList -join ' '))
    $script:serverJob = Start-Job -ScriptBlock {
        param($r, $a)
        Set-Location -LiteralPath $r
        $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUTF8 = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'
        & pwsh -NoProfile -File run_py.ps1 @a
    } -ArgumentList $root, $argList
}

function Stop-App {
    if ($script:serverJob) {
        Stop-Job -Job $script:serverJob -ErrorAction SilentlyContinue
        Remove-Job -Job $script:serverJob -Force -ErrorAction SilentlyContinue
        $script:serverJob = $null
    }
    $pids = (netstat -ano | Select-String (":{0}\s.*LISTENING" -f $Port) |
             ForEach-Object { ($_.Line -split '\s+')[-1] } | Sort-Object -Unique)
    foreach ($p in $pids) { if ($p -match '^\d+$') { Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue } }
}

# ---------------------------------------------------------------- CLI 模式（前台，立即返回）
if ($Mode -eq 'cli') {
    if (-not $Question) {
        $Question = '武汉兴图新科电子股份有限公司注册资本是多少？'
        Write-Host "[提示] 未传 -Question，使用默认问题：$Question"
    }
    Write-Host ("[CLI] pwsh -NoProfile -File run_py.ps1 研发/app/main.py --ask `"{0}`"" -f $Question)
    & pwsh -NoProfile -File run_py.ps1 研发/app/main.py --ask $Question
    $code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    exit $code
}

# ---------------------------------------------------------------- 服务模式
Write-Host ""
Write-Host ("=" * 78)
Write-Host ("工单4 部署自检：模式={0} 地址={1} 预热={2}" -f $Mode, $baseUrl, (-not $NoWarmup))
Write-Host ("=" * 78)

$already = Test-Ready
if ($already) {
    Write-Host "[复用] $healthUrl 已有服务在响应（本次不新起、结束时也不停它）"
} else {
    Start-App
    $script:startedByUs = $true
    $ok = $false
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 1500
        if (Test-Ready) { $ok = $true; break }
    }
    if (-not $ok) {
        Write-Host ("[失败] 服务 {0} 秒内未就绪——如实标记失败（不伪造）" -f $TimeoutSec)
        if ($serverJob) {
            Write-Host "[诊断] 本次启动的子进程输出（末 20 行）："
            Receive-Job -Job $serverJob -ErrorAction SilentlyContinue | Select-Object -Last 20 | ForEach-Object { Write-Host "    $_" }
        }
        Add-Check '服务就绪' $false ("{0}s 内 {1} 无响应" -f $TimeoutSec, $healthUrl)
        if ($startedByUs -and -not $NoStop) { Stop-App }
        exit 1
    }
    Write-Host "[就绪] 服务已响应 $healthUrl"
}

# 1) /health：HTTP 200 且 status=ok
$health = Invoke-Api 'GET' $healthUrl $null 60
$healthStatus = if ($health.json) { [string]$health.json.status } else { '(无 JSON)' }
Add-Check '/health HTTP 200' ($health.status -eq 200) ("HTTP {0}" -f $health.status)
Add-Check '/health status=ok' ($healthStatus -eq 'ok') ("status={0}" -f $healthStatus)

$chunks = $null; $imageChunks = $null; $warmed = $null
if ($health.json -and $health.json.checks -and $health.json.checks.index) {
    $chunks = $health.json.checks.index.chunks
    $imageChunks = $health.json.checks.index.image_chunks
    Add-Check '索引已加载' ($chunks -gt 0) ("chunks={0} image_chunks={1} dim={2}" -f `
        $chunks, $imageChunks, $health.json.checks.index.dim)
}
if ($health.json) { $warmed = $health.json.warmed }

# 2) 可选：全部端点冒烟（-SmokeAll）
if ($SmokeAll) {
    $t0 = Get-Date
    $ask = Invoke-Api 'POST' "$baseUrl/ask" @{ question = '武汉兴图新科电子股份有限公司注册资本是多少？'
                                              lang = 'zh'; use_cache = $true } 180
    $askMs = [int]((Get-Date) - $t0).TotalMilliseconds
    $answer = if ($ask.json) { [string]$ask.json.answer } else { '' }
    Add-Check 'POST /ask' ($ask.status -eq 200 -and $answer.Contains('5,520')) `
        ("HTTP {0}，{1} ms，答案含 5,520 = {2}" -f $ask.status, $askMs, $answer.Contains('5,520'))

    $ret = Invoke-Api 'POST' "$baseUrl/retrieve" @{ question = '变更募投项目的程序' ; top_k = 5 } 180
    $hitCount = if ($ret.json) { $ret.json.hit_count } else { $null }
    Add-Check 'POST /retrieve' ($ret.status -eq 200 -and $hitCount -gt 0) `
        ("HTTP {0}，hit_count={1}" -f $ret.status, $hitCount)

    $stream = Invoke-Api 'POST' "$baseUrl/ask/stream" @{ question = '公司来自军用领域的收入占比分别是多少？'
                                                         lang = 'zh'; use_cache = $true } 180
    if ($Mode -eq 'api') {
        # /ask/stream 是 FastAPI 服务独有端点；纯标准库 Web（研发/app/ui/serve_fallback.py）
        # 没有这个路由 → 只在 api 模式断言，避免把「模式差异」误记成失败。
        Add-Check 'POST /ask/stream' ($stream.status -eq 200 -and $stream.text.Contains('data:')) `
            ("HTTP {0}，SSE 事件 {1} 个" -f $stream.status, (($stream.text -split "`n" | Where-Object { $_ -like 'data:*' }).Count))
    } else {
        Write-Host ("  [SKIP] POST /ask/stream —— web 模式无该端点（HTTP {0}；只有 FastAPI 服务提供）" -f $stream.status)
    }

    $qn = Invoke-Api 'GET' "$baseUrl/questions" $null 60
    Add-Check 'GET /questions' ($qn.status -eq 200) ("HTTP {0}" -f $qn.status)

    $page = Invoke-Api 'GET' "$baseUrl/" $null 60
    Add-Check 'GET /' ($page.status -eq 200 -and $page.text.Contains('招股说明书')) ("HTTP {0}，{1} 字节" -f $page.status, $page.text.Length)

    if ($Mode -eq 'api') {
        $info = Invoke-Api 'GET' "$baseUrl/eval" $null 60
        Add-Check 'GET /eval（口径说明）' ($info.status -eq 200 -and $null -ne $info.json.golden) `
            ("HTTP {0}，golden sha256={1}" -f $info.status, ([string]$info.json.golden.canonical_sha256).Substring(0, 16))
        $ev = Invoke-Api 'POST' "$baseUrl/eval" @{ limit = 3; top_k = 5; use_cache = $true; include_details = $false } 600
        $acc = if ($ev.json) { $ev.json.summary.accuracy } else { $null }
        $evMs = if ($ev.json) { $ev.json.summary.elapsed_ms } else { $null }
        Add-Check 'POST /eval（前 3 题）' ($ev.status -eq 200 -and $null -ne $acc) `
            ("HTTP {0}，accuracy={1}（{2} ms）" -f $ev.status, $acc, $evMs)
    }
}

# ---------------------------------------------------------------- 留痕 + 退出
$summary = [ordered]@{
    generated_by  = '部署/脚本/run_app.ps1'
    finished_at   = (Get-Date).ToString('s')
    mode          = $Mode
    base_url      = $baseUrl
    health_url    = $healthUrl
    health_http   = $health.status
    health_status = $healthStatus
    index_chunks  = $chunks
    image_chunks  = $imageChunks
    warmed        = $warmed
    reused_service = [bool]$already
    started_by_script = [bool]$startedByUs
    smoke_all     = [bool]$SmokeAll
    exit_code     = $script:failed
    checks        = $script:checks
    note          = '本文件由部署自检实跑写出；数字来自本次运行，未预填。'
}
try {
    ($summary | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $evidencePath -Encoding UTF8
    Write-Host ("[OK] 已写自检留痕 {0}" -f $evidencePath)
} catch {
    Write-Host ("[警告] 自检留痕写出失败：{0}" -f $_.Exception.Message)
}

if (-not $HealthCheckOnly) {
    Write-Host ""
    Write-Host ("服务仍在运行（未加 -HealthCheckOnly）：{0}  —— 用 Ctrl+C 停止" -f $baseUrl)
    if ($startedByUs -and $serverJob) { Wait-Job -Job $serverJob | Out-Null }
    exit $script:failed
}

if ($startedByUs -and -not $NoStop) {
    Write-Host "[清理] 关闭本次自检拉起的服务进程…"
    Stop-App
} elseif ($startedByUs) {
    Write-Host "[保留] 按 -NoStop 保留本次拉起的服务（自检结束后仍在运行）"
}

Write-Host ""
if ($script:failed -gt 0) {
    Write-Host ("结论：{0} 项检查失败，退出码 1" -f $script:failed) -ForegroundColor Red
    if ($health.json -and $health.json.error -and $health.json.error.code -eq 'E_INDEX_MISSING') {
        Write-Host "原因：索引缺失（E_INDEX_MISSING）——先跑 研发/scripts/build_index.py --incremental --include-images" -ForegroundColor Yellow
        exit 3
    }
    exit 1
}
Write-Host ("结论：自检全部通过（模式={0}，{1}）" -f $Mode, $baseUrl) -ForegroundColor Green
exit 0
