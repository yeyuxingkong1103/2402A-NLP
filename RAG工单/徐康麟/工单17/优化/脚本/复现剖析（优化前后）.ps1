# =============================================================================
# 工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单
# 文件：优化/脚本/复现剖析（优化前后）.ps1
#
# 用途：串行跑完「优化前 / 优化后」的进程内剖析（cProfile + urlopen 计数 + tracemalloc + gc）。
#
# ⚠️ 前置条件：**服务必须已停止** —— 剖析脚本会在自己进程里构造一个 QAEngine
#    （实测私有内存 ~2.4 GB），与服务叠加会超过本机可用内存（共享约定 §5.5）。
#
# 用法（工作目录 = E:\gao6gongdan\工单17）：
#     pwsh -NoProfile -File 部署/脚本/本地停止.ps1
#     pwsh -NoProfile -File 优化/脚本/复现剖析（优化前后）.ps1
# =============================================================================
[CmdletBinding()]
param(
    [int]$N = 8,
    [int]$Rounds = 6,
    [int]$PerRound = 8
)

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

Write-Host "=================================================================="
Write-Host "剖析复现（**服务必须已停止**）"
Write-Host "=================================================================="

# ⚠️ 实测踩坑：``run_py.ps1`` 会把 ``RAG_DATA__INDEX_DIR`` 默认设成
#    ``<工单根>\研发\data\index``（本工单**不复制索引**，该目录不存在），
#    于是剖析脚本会以 ``IndexMissingError`` 直接退出。
#    必须在调用 run_py.ps1 **之前**把索引指到工单7 的真实索引（run_py 只在为空时才设默认值）。
$env:RAG_DATA__INDEX_DIR = 'E:\gao6gongdan\工单7\研发\data\index'
$env:RAG_DATA__RAW_DIR = 'E:\gao6gongdan\工单7\研发\data\raw'
$env:RAG_LOG__DIR = (Join-Path $repoRoot '部署\日志')
$env:RAG_LLM__OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
$env:DEPLOY_RUNTIME_DIR = (Join-Path $repoRoot '部署\卷\runtime')
$env:DEPLOY_LOG_DIR = (Join-Path $repoRoot '部署\日志')
New-Item -ItemType Directory -Force -Path $env:RAG_LOG__DIR, $env:DEPLOY_RUNTIME_DIR | Out-Null
Write-Host "索引（只读）：$env:RAG_DATA__INDEX_DIR"
Write-Host "运行期目录  ：$env:DEPLOY_RUNTIME_DIR"
$alive = $false
try {
    $code = & curl.exe -s -o NUL -w '%{http_code}' --max-time 4 'http://127.0.0.1:8200/healthz' 2>$null
    if ("$code" -eq '200') { $alive = $true }
} catch { }
if ($alive) {
    Write-Host "[中止] 8200 端口仍有服务在响应 —— 先跑 部署/脚本/本地停止.ps1（避免 2×2.4 GB 叠加）" -ForegroundColor Red
    exit 1
}

foreach ($tag in '优化前', '优化后') {
    $flag = if ($tag -eq '优化前') { '0' } else { '1' }
    Write-Host "`n######### $tag（PERF_ENABLED=$flag）#########" -ForegroundColor Cyan
    $env:PERF_ENABLED = $flag
    $env:PERF_LOG_MAX_MB = '64'
    $env:PERF_LOG_BACKUP_COUNT = '5'

    Write-Host "-- 问答链路剖析（cProfile + urlopen 计数）--"
    pwsh -NoProfile -File run_py.ps1 研发/剖析/剖析问答链路.py --tag $tag --n $N --profile

    Write-Host "-- 内存泄漏剖析（tracemalloc + gc）--"
    pwsh -NoProfile -File run_py.ps1 研发/剖析/内存剖析.py --tag $tag --rounds $Rounds --per-round $PerRound
}

Write-Host "`n完成。产物：测试/测试数据/{剖析问答链路_*,tracemalloc_*,cProfile_*}" -ForegroundColor Green
