# =============================================================================
# 工单：人工智能NLP-RAG-Query 理解优化
# 文件：部署/脚本/run_web.ps1 —— 多轮对话界面 / 服务的一键启动器（Windows）
#
# 三种模式（都走仓库统一入口 run_py.ps1，保证解释器 / UTF-8 / PYTHONPATH 一致）：
#   -Mode gradio （首选界面）部署/界面/gradio_app.py   默认端口 7860  需要 gradio
#   -Mode web    （兜底界面）部署/界面/web_app.py      默认端口 8123  零第三方依赖，必可用
#   -Mode api    （HTTP 服务）部署/脚本/serve_api.py   默认端口 8600  FastAPI（/api/chat 多轮）
#
# 用法：
#   pwsh -NoProfile -File 部署/脚本/run_web.ps1                       # 默认 -Mode web（8123）
#   pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode gradio -Port 7860
#   pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode api -Port 8600
#   pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode gradio -Force   # 端口被占则先清掉占用者
#   pwsh -NoProfile -File 部署/脚本/run_web.ps1 -NoWarmup             # 跳过预热（仅调试）
#
# 退出码：0 正常退出 / 1 端口被占用（未加 -Force） / 2 参数错误 /
#         3 gradio 缺失（gradio 模式；此时请用 -Mode web 兜底） / 127 解释器缺失
#
# 说明（实测结论，勿踩）：
#   · 本机禁止裸 pip install（落点是只读的 工单1 venv）。需要 gradio 时：
#         pwsh -NoProfile -File run_py.ps1 -m pip install --target 研发/第三方库 gradio
#     本脚本若发现 研发/第三方库 存在，会自动把它加进 PYTHONPATH（无需改 run_py.ps1）。
#   · 前台运行（不加 Start-Process）：日志直接打在控制台，Ctrl+C 即停，最简单可靠。
#   · 端口占用只提示、不擅自杀进程；确认要抢占时显式加 -Force。
# =============================================================================
param(
    [ValidateSet('gradio', 'web', 'api')][string]$Mode = 'web',
    [string]$BindHost = '127.0.0.1',
    [int]$Port = 0,
    [switch]$Share,
    [switch]$NoWarmup,
    [switch]$Force
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)      # 部署/脚本 → 部署 → 工单5
Set-Location -LiteralPath $root

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

# 目标脚本与默认端口
switch ($Mode) {
    'gradio' { $targetScript = '部署/界面/gradio_app.py';  $defaultPort = 7860; $label = 'Gradio 首选界面' }
    'web'    { $targetScript = '部署/界面/web_app.py';     $defaultPort = 8123; $label = '标准库兜底界面' }
    'api'    { $targetScript = '部署/脚本/serve_api.py';   $defaultPort = 8600; $label = 'FastAPI 服务（多轮）' }
}
if ($Port -le 0) { $Port = $defaultPort }
if ($Port -lt 1 -or $Port -gt 65535) {
    Write-Host ("[参数错误] -Port {0} 不在 1~65535 范围内" -f $Port) -ForegroundColor Red
    exit 2
}

$entry = Join-Path $root $targetScript
if (-not (Test-Path -LiteralPath $entry)) {
    Write-Host ("[参数错误] 找不到目标脚本：{0}" -f $entry) -ForegroundColor Red
    Write-Host "           该文件可能尚未交付（研发/部署并行开发中）。" -ForegroundColor Yellow
    exit 2
}

# 私有依赖目录（--target 安装落点）：存在则加进 PYTHONPATH，供 gradio 等使用
$vendorDir = Join-Path $root '研发/第三方库'
$pyPath = Join-Path $root '研发'
if (Test-Path -LiteralPath $vendorDir) {
    $env:PYTHONPATH = "$pyPath;$vendorDir"
    Write-Host ("[依赖] 已把私有依赖目录加进 PYTHONPATH：{0}" -f $vendorDir)
} else {
    $env:PYTHONPATH = $pyPath
}

$baseUrl = "http://{0}:{1}" -f $BindHost, $Port
Write-Host ""
Write-Host ("=" * 78)
Write-Host ("工单5 界面启动：模式={0}（{1}）  地址={2}" -f $Mode, $label, $baseUrl)
Write-Host ("脚本：{0}" -f $targetScript)
Write-Host ("=" * 78)

# ---------------------------------------------------------------- 端口占用
$owners = @(netstat -ano | Select-String (":{0}\s.*LISTENING" -f $Port) |
            ForEach-Object { ($_.Line -split '\s+')[-1] } | Sort-Object -Unique | Where-Object { $_ -match '^\d+$' })
if ($owners.Count -gt 0) {
    $names = @()
    foreach ($p in $owners) {
        $proc = Get-Process -Id ([int]$p) -ErrorAction SilentlyContinue
        $names += ("{0}({1})" -f $(if ($proc) { $proc.ProcessName } else { '未知' }), $p)
    }
    Write-Host ("[端口占用] {0} 已被监听：{1}" -f $Port, ($names -join ', ')) -ForegroundColor Yellow
    if ($Force) {
        foreach ($p in $owners) {
            Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue
            Write-Host ("[端口占用] 已按 -Force 结束进程 {0}" -f $p)
        }
        Start-Sleep -Milliseconds 800
    } else {
        Write-Host "  处理方式（三选一）：" -ForegroundColor Yellow
        Write-Host ("    1) 换端口：pwsh -NoProfile -File 部署/脚本/run_web.ps1 -Mode {0} -Port {1}" -f $Mode, ($Port + 1))
        Write-Host ("    2) 查看占用者：netstat -ano | findstr :{0}" -f $Port)
        Write-Host  "    3) 确认可抢占：加 -Force（会结束上述进程，慎用）"
        exit 1
    }
}

# ---------------------------------------------------------------- 前台启动
$argList = @($targetScript, '--host', $BindHost, '--port', "$Port")
if ($Mode -eq 'gradio' -and $Share) { $argList += '--share' }
if ($NoWarmup) { $argList += '--no-warmup' }

Write-Host ("[启动] pwsh -NoProfile -File run_py.ps1 {0}" -f ($argList -join ' '))
Write-Host ("[提示] 界面就绪后浏览器打开 {0}/ ；健康检查 {0}/health ；Ctrl+C 停止" -f $baseUrl)
if ($Mode -eq 'gradio') {
    Write-Host "[提示] gradio 缺失时本命令退出码为 3；此时请改用：-Mode web（零依赖兜底）" -ForegroundColor Yellow
}
Write-Host ""

& pwsh -NoProfile -File (Join-Path $root 'run_py.ps1') @argList
$code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
Write-Host ("[退出] {0}（退出码 {1}）" -f $targetScript, $code)
exit $code
