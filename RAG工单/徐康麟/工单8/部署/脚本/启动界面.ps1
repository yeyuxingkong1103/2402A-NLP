# =============================================================================
# 工单：人工智能NLP-RAG-基于Graph RAG 实现金融问答（工单8）
# 脚本：启动 Graph RAG 问答界面（纯标准库 Web，端口 8124）
#
# 用法：
#   pwsh -NoProfile -File 部署\脚本\启动界面.ps1
#   pwsh -NoProfile -File 部署\脚本\启动界面.ps1 -Port 8125 -NoWarmup
#
# 资源纪律（_共享约定.md §5.5）：启动前先探内存，可用 < 4000 MB 直接拒绝启动。
# =============================================================================
param(
    [int]$Port = 8124,
    [string]$HostName = '127.0.0.1',
    [switch]$NoWarmup,
    [switch]$SkipMemoryCheck
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repo

Write-Host "[工单8] 仓库：$repo" -ForegroundColor Cyan

# ---- 1) 内存探测（防 OOM，事故后强制）-------------------------------------
if (-not $SkipMemoryCheck) {
    $free = (Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue
    Write-Host ("[工单8] 可用内存：{0:N0} MB" -f $free)
    if ($free -lt 4000) {
        Write-Warning "[工单8] 可用内存 $([int]$free) MB < 4000 MB，按资源纪律拒绝启动。"
        Write-Warning "        请先释放内存（关闭其它重活/等待其它任务结束）后重试；"
        Write-Warning "        确需强制启动请加 -SkipMemoryCheck。"
        exit 3
    }
}

# ---- 2) 环境变量（与 run_py.ps1 同源：索引指向工单7，端口 8124）-----------
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$baseRepo = 'E:\gao6gongdan\工单7'
if (-not $env:RAG_DATA__INDEX_DIR)     { $env:RAG_DATA__INDEX_DIR     = Join-Path $baseRepo '研发\data\index' }
if (-not $env:RAG_DATA__PROCESSED_DIR) { $env:RAG_DATA__PROCESSED_DIR = Join-Path $baseRepo '研发\data\processed' }
if (-not $env:RAG_LOG__DIR)            { $env:RAG_LOG__DIR            = Join-Path $repo '部署\日志' }
if (-not $env:GRAPHRAG_GRAPH_DIR)      { $env:GRAPHRAG_GRAPH_DIR      = Join-Path $repo '研发\data\graph' }

# ---- 3) 前置产物检查 ------------------------------------------------------
$required = @(
    (Join-Path $repo '研发\data\graph\graph.json'),
    (Join-Path $repo '研发\data\graph\entities.jsonl'),
    (Join-Path $env:RAG_DATA__INDEX_DIR 'bge-m3_1024\chunks_indexed.jsonl')
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Error "[工单8] 缺少前置产物：$path"
        Write-Host  "        请先执行：pwsh -NoProfile -File 部署\脚本\一键跑通.ps1" -ForegroundColor Yellow
        exit 2
    }
}

# ---- 4) 启动 --------------------------------------------------------------
$args = @('研发/app/ui/serve_graph_web.py', '--host', $HostName, '--port', "$Port")
if ($NoWarmup) { $args += '--no-warmup' }
Write-Host "[工单8] 启动界面：http://${HostName}:${Port}/" -ForegroundColor Green
Write-Host "[工单8] 日志：部署\日志\graph_web.log（按 Ctrl+C 停止）"
& pwsh -NoProfile -File (Join-Path $repo 'run_py.ps1') @args *> (Join-Path $repo '部署\日志\graph_web.log')
exit $LASTEXITCODE
