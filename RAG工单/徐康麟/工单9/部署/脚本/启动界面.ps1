# 工单9 启动 Web 界面（端口 8125）

# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
#
# 页面展示四要素：问题 / 检索到的上下文（含图谱三元组）/ 答案 / 引用。
# 端口 8125 由工单指定（工单8 = 8124，工单7 = 8123，互不冲突）。
#
# 用法：
#   pwsh -NoProfile -File 部署/脚本/启动界面.ps1                 # 前台启动（Ctrl+C 停止）
#   pwsh -NoProfile -File 部署/脚本/启动界面.ps1 -Port 8125 -Background
#   pwsh -NoProfile -File 部署/脚本/启动界面.ps1 -OpenBrowser

[CmdletBinding()]
param(
    [int]$Port = 8125,
    [string]$Host_ = '127.0.0.1',
    [switch]$Background,
    [switch]$OpenBrowser
)

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$env:PYTHONDONTWRITEBYTECODE = '1'
$LogDir = Join-Path $Root '部署\日志'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir "serve_$Port.log"
$url = "http://${Host_}:$Port/"

Write-Host "工单9 Graph RAG 优化版界面" -ForegroundColor Cyan
Write-Host "  地址 : $url"
Write-Host "  日志 : 部署/日志/serve_$Port.log"
Write-Host "  提示 : 首次启动会预热引擎（加载 8847 块索引 + 6717 节点图谱，约 3~5 s）"

if ($Background) {
    $args = @('-NoProfile', '-File', 'run_py.ps1', '研发/app/ui/serve_fallback.py',
              '--port', "$Port", '--host', $Host_)
    Start-Process -FilePath 'pwsh' -ArgumentList $args -WorkingDirectory $Root `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden
    Write-Host '  已在后台启动。' -ForegroundColor Green
    Start-Sleep -Seconds 8
} else {
    if ($OpenBrowser) { Start-Process $url }
    & pwsh -NoProfile -File run_py.ps1 研发/app/ui/serve_fallback.py --port $Port --host $Host_
    exit $LASTEXITCODE
}

# 健康检查
try {
    $h = Invoke-RestMethod "$url`healthz" -TimeoutSec 30
    Write-Host "  健康检查: status=$($h.status)  索引=$($h.checks.index.chunks) 块  " -NoNewline
    Write-Host "图谱=$($h.checks.graph.nodes) 节点" -ForegroundColor Green
} catch {
    Write-Host "  [警告] 健康检查失败：$_" -ForegroundColor Yellow
}
if ($OpenBrowser) { Start-Process $url }
Write-Host "  打开 $url 即可问答（支持中文与英文）" -ForegroundColor Green
