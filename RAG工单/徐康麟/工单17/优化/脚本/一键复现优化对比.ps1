# =============================================================================
# 工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单
# 文件：优化/脚本/一键复现优化对比.ps1
#
# 用途：从零复现「优化前 → 优化后」的完整对比数据。
#       严格串行（一次只有一个服务实例 + 一个压测器），每步都探内存。
#
# 资源纪律（共享约定 §5.5 + 工单17 附加）：
#   * 并发硬上限 20；压测器内建熔断（提交内存 >80% 或错误率 >50% 立即停）
#   * 所有重活经 _共享工具\安全跑.ps1 包裹（-PrivateLimitMB 3400，API 自身占 2.4 GB）
#   * 启动前探可用内存，< 1500 MB 直接不跑
#
# 用法（工作目录 = E:\gao6gongdan\工单17）：
#     pwsh -NoProfile -File 优化/脚本/一键复现优化对比.ps1
#     pwsh -NoProfile -File 优化/脚本/一键复现优化对比.ps1 -Duration 180 -SkipSoak
# =============================================================================
[CmdletBinding()]
param(
    [int]$Duration = 180,
    [int]$SoakDuration = 900,
    [switch]$SkipSoak,
    [switch]$SkipBaseline
)

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$safeRun = 'E:\gao6gongdan\_共享工具\安全跑.ps1'
$runner = '研发/压测/load_runner.py'
$results = '优化/评估结果'

function Get-AvailMB {
    try { return [math]::Round((Get-Counter '\Memory\Available MBytes' -ErrorAction Stop).CounterSamples.CookedValue) }
    catch { return -1 }
}

function Invoke-Background([string]$name, [string]$command) {
    Write-Host "`n---------- $name ----------" -ForegroundColor Cyan
    $avail = Get-AvailMB
    Write-Host "  启动前可用内存：$avail MB"
    if ($avail -ge 0 -and $avail -lt 1500) {
        Write-Host "  [跳过] 可用内存不足（< 1500 MB），先清理再跑" -ForegroundColor Yellow
        return $false
    }
    & pwsh -NoProfile -File $safeRun -WorkDir $repoRoot -Command $command -LogName $name -PrivateLimitMB 3400
    return ($LASTEXITCODE -eq 0)
}

Write-Host "=================================================================="
Write-Host "工单17 优化前后一键复现"
Write-Host "参数：场景时长 $Duration s；soak $SoakDuration s；SkipBaseline=$SkipBaseline；SkipSoak=$SkipSoak"
Write-Host "=================================================================="

# ---------------------------------------------------------------- 优化前 --
if (-not $SkipBaseline) {
    Write-Host "`n[1/4] 起服务：PERF_ENABLED=0（基座口径）" -ForegroundColor Green
    pwsh -NoProfile -File 部署/脚本/本地停止.ps1 | Out-Null
    pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 0 -Tag 基准_优化前 -WaitSeconds 300
    if ($LASTEXITCODE -ne 0) { Write-Host "[中止] 服务未就绪" -ForegroundColor Red; exit 1 }

    Invoke-Background '复现_基准_场景A' "pwsh -NoProfile -File run_py.ps1 $runner --scenario A --concurrency 20 --duration $Duration --think-ms 0 --tag 基准_优化前 --outdir 测试/测试数据" | Out-Null
    Invoke-Background '复现_基准_场景A多用户' "pwsh -NoProfile -File run_py.ps1 $runner --scenario A --concurrency 20 --duration $Duration --think-ms 0 --unique-sessions --tag 基准_优化前_多用户 --outdir 测试/测试数据" | Out-Null
    Invoke-Background '复现_基准_场景B' "pwsh -NoProfile -File run_py.ps1 $runner --scenario B --concurrency 10 --duration $Duration --think-ms 300 --tag 基准_优化前 --outdir 测试/测试数据" | Out-Null

    Write-Host "`n[2/4] 停服务并做「优化前」进程内剖析（脚本自建引擎约 2.4 GB，必须停服务）"
    pwsh -NoProfile -File 部署/脚本/本地停止.ps1 | Out-Null
    $env:PERF_ENABLED = '0'
    Invoke-Background '复现_剖析_优化前' "pwsh -NoProfile -File run_py.ps1 研发/剖析/剖析问答链路.py --tag 优化前 --n 8 --profile; pwsh -NoProfile -File run_py.ps1 研发/剖析/内存剖析.py --tag 优化前 --rounds 6 --per-round 12" | Out-Null
}

# ---------------------------------------------------------------- 优化后 --
Write-Host "`n[3/4] 起服务：PERF_ENABLED=1（优化后口径，限流 20/64 以对齐 20 并发验收场景）" -ForegroundColor Green
pwsh -NoProfile -File 部署/脚本/本地停止.ps1 | Out-Null
$env:PERF_MAX_INFLIGHT = '20'; $env:PERF_QUEUE_MAX = '64'
pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 1 -Tag 优化后 -WaitSeconds 300
if ($LASTEXITCODE -ne 0) { Write-Host "[中止] 服务未就绪" -ForegroundColor Red; exit 1 }

Invoke-Background '复现_优化后_场景A' "pwsh -NoProfile -File run_py.ps1 $runner --scenario A --concurrency 20 --duration $Duration --think-ms 0 --tag 优化后 --outdir 测试/测试数据" | Out-Null
Invoke-Background '复现_优化后_场景A多用户' "pwsh -NoProfile -File run_py.ps1 $runner --scenario A --concurrency 20 --duration $Duration --think-ms 0 --unique-sessions --tag 优化后_多用户 --outdir 测试/测试数据" | Out-Null
Invoke-Background '复现_优化后_场景B' "pwsh -NoProfile -File run_py.ps1 $runner --scenario B --concurrency 10 --duration $Duration --think-ms 300 --tag 优化后 --outdir 测试/测试数据" | Out-Null

if (-not $SkipSoak) {
    Write-Host "`n[4/4] 资源泄漏 soak（$SoakDuration s，并发 8）" -ForegroundColor Green
    Invoke-Background '复现_soak_优化后' "pwsh -NoProfile -File run_py.ps1 $runner --scenario A --concurrency 8 --duration $SoakDuration --think-ms 500 --tag soak_优化后 --outdir 测试/测试数据" | Out-Null
}

Write-Host "`n---------- 汇总对比 ----------" -ForegroundColor Cyan
pwsh -NoProfile -File run_py.ps1 研发/scripts/汇总对比.py
Write-Host "`n---------- 生成截图 ----------" -ForegroundColor Cyan
pwsh -NoProfile -File run_py.ps1 测试/测试脚本/生成截图.py

Write-Host "`n全部完成。产物：$results\{基准_优化前.json, 优化后.json, 对比.csv, 对比.md}" -ForegroundColor Green
Write-Host "记得停服务：pwsh -NoProfile -File 部署/脚本/本地停止.ps1" -ForegroundColor Yellow
