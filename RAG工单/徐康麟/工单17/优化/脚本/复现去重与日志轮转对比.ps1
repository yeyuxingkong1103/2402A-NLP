# =============================================================================
# 工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单
# 文件：优化/脚本/复现去重与日志轮转对比.ps1
#
# 用途：一次跑完 F11（日志轮转）/ F12（上传去重）的**前后对比**，并顺带跑 soak。
#       严格串行：一次只有一个服务实例 + 一个压测器。
#
# 用法（工作目录 = E:\gao6gongdan\工单17）：
#     pwsh -NoProfile -File 优化/脚本/复现去重与日志轮转对比.ps1 -N 60 -SoakSeconds 600
# =============================================================================
[CmdletBinding()]
param(
    [int]$N = 60,
    [int]$SoakSeconds = 600
)

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

Write-Host "=================================================================="
Write-Host "[1/4] 起服务：PERF_ENABLED=0（不去重、不轮转 = 基座行为）"
Write-Host "=================================================================="
pwsh -NoProfile -File 部署/脚本/本地停止.ps1 | Out-Null
$env:PERF_ENABLED = '0'
pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 0 -Tag 去重前 -WaitSeconds 150

Write-Host "`n[2/4] 优化前口径：同一份 PDF 连续上传 $N 次"
pwsh -NoProfile -File run_py.ps1 研发/scripts/验证去重与日志轮转.py --label 优化前 --n $N

Write-Host "`n[3/4] 起服务：PERF_ENABLED=1（去重开、日志 1 MB 轮转）"
pwsh -NoProfile -File 部署/脚本/本地停止.ps1 | Out-Null
$env:PERF_ENABLED = '1'
$env:PERF_UPLOAD_DEDUP = '1'
$env:PERF_LOG_MAX_MB = '1'
$env:PERF_LOG_BACKUP_COUNT = '5'
$env:PERF_MAX_INFLIGHT = '20'
$env:PERF_QUEUE_MAX = '64'
pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 1 -Tag 去重后 -WaitSeconds 150

pwsh -NoProfile -File run_py.ps1 研发/scripts/验证去重与日志轮转.py --label 优化后 --n $N

Write-Host "`n[4/4] 资源泄漏 soak（$SoakSeconds s，并发 8）"
pwsh -NoProfile -File run_py.ps1 研发/压测/load_runner.py --scenario A --concurrency 8 `
    --duration $SoakSeconds --think-ms 500 --tag soak_优化后 --outdir 测试/测试数据

Write-Host "`n=== 恢复日志轮转默认值（64 MB）并记录轮转状态 ==="
pwsh -NoProfile -File run_py.ps1 -c "import json,urllib.request;d=json.load(urllib.request.urlopen('http://127.0.0.1:8200/api/v1/perf',timeout=10));print(json.dumps({'limiter':d['limiter'],'task_queue':d['task_queue'],'http_pool':d['http_pool']},ensure_ascii=False)[:900])"

Write-Host "`n完成。记得停服务：pwsh -NoProfile -File 部署/脚本/本地停止.ps1" -ForegroundColor Yellow
