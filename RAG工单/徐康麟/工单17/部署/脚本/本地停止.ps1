# =============================================================================
# 工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单
# 文件：部署/脚本/本地停止.ps1 —— 停止本机等价部署起的进程（对应 compose down）
#
# 纪律（工单17 收尾铁律）：
#   * 只杀 部署/日志/本地进程.json 里记录的 PID（启动器 + 真实服务进程）
#     以及「不在 python基线.json 里」的新增 python —— **不做**全端口扫描杀进程；
#   * 结束前必须确认：Get-Process python 只剩会话基线（约 4 个、均 < 200 MB）、
#     API 端口已释放、__pycache__ / .pytest_cache 清零。
#
# ⚠️ 本机实测：`Get-CimInstance Win32_Process` 在沙箱下报「无法从客户端中访问 CIM 资源」，
#    因此**不能**按命令行特征找进程 → 一律按 PID + 基线差集处理。
#
# 用法：
#     pwsh -NoProfile -File 部署/脚本/本地停止.ps1
#     pwsh -NoProfile -File 部署/脚本/本地停止.ps1 -KeepVolumes:$false   # 连"卷"目录一起清
#     pwsh -NoProfile -File 部署/脚本/本地停止.ps1 -KillNewPython        # 额外清掉所有非基线 python
# =============================================================================
[CmdletBinding()]
param(
    [switch]$KeepVolumes = $true,
    [switch]$KillNewPython,
    [int]$BaselinePythonMaxMB = 200
)

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$logDir = Join-Path $repoRoot '部署\日志'
$pidFile = Join-Path $logDir '本地进程.json'
$baselineFile = Join-Path $logDir 'python基线.json'
$script:failed = $false

Write-Host "===== 工单17 停止本机等价部署进程 ====="

# ---------- 0) 读基线 ----------
$baseline = @()
if (Test-Path $baselineFile) {
    try { $baseline = @((Get-Content -LiteralPath $baselineFile -Raw -Encoding utf8 | ConvertFrom-Json).pids) } catch { }
}
Write-Host "python 基线 PID：$(if ($baseline.Count) { ($baseline | Sort-Object) -join ', ' } else { '（未记录）' })"

# ---------- 1) 停 API（启动器 PID + 真实服务 PID，两个都杀） ----------
$apiPort = 8200
if (Test-Path $pidFile) {
    $info = Get-Content -LiteralPath $pidFile -Raw -Encoding utf8 | ConvertFrom-Json
    Write-Host "启动时间：$($info.started_at)   标签：$($info.tag)   PERF_ENABLED=$($info.perf_enabled)"
    $apiPort = [int]$info.api_port
    $targets = @()
    if ($info.api_real_pid) { $targets += @{ n = 'rag-api(真实)'; pid = [int]$info.api_real_pid } }
    if ($info.api_pid)      { $targets += @{ n = 'rag-api(启动器)'; pid = [int]$info.api_pid } }
    foreach ($item in $targets) {
        $p = Get-Process -Id $item.pid -ErrorAction SilentlyContinue
        if ($null -eq $p) { Write-Host "  [$($item.n)] PID $($item.pid) 已不存在"; continue }
        $mb = [math]::Round($p.PrivateMemorySize64 / 1MB)
        Stop-Process -Id $item.pid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 700
        if (Get-Process -Id $item.pid -ErrorAction SilentlyContinue) {
            & taskkill /PID $item.pid /T /F 2>&1 | ForEach-Object { Write-Host "  [$($item.n)] $_" }
        }
        if (Get-Process -Id $item.pid -ErrorAction SilentlyContinue) {
            Write-Host "  [失败] $($item.n) PID $($item.pid)（$mb MB）**仍在运行**" -ForegroundColor Red
            $script:failed = $true
        } else {
            Write-Host "  [OK] $($item.n) PID $($item.pid)（$mb MB）已停止"
        }
    }
} else {
    Write-Host "[跳过] 没有 $pidFile，说明本工单未启动过服务"
}

# ---------- 2) 清掉"本次新增"的 python（压测器 / 剖析脚本 / 漏网的孤儿） ----------
# ⚠️ 安全前提：差集清理**必须显式加 -KillNewPython**。
#    实测教训：基线文件一旦过期（宿主辅助进程换过 PID），差集就等于"全部 python"，
#    会把宿主的辅助进程一起杀掉（杀完宿主还会重启它们，但这是不该发生的行为）。
#    默认行为：只按 PID 停服务 + 清掉**私有内存 > 200 MB** 的残留大进程。
Start-Sleep -Seconds 1
if ($baseline.Count -eq 0 -or -not $KillNewPython) {
    Write-Host "  [差集清理] $(if ($baseline.Count -eq 0) { '基线为空 → 跳过' } else { '未指定 -KillNewPython → 跳过（只清大内存残留）' })" -ForegroundColor Yellow
    $heavy = @(Get-Process python -ErrorAction SilentlyContinue |
               Where-Object { ($_.PrivateMemorySize64 / 1MB) -gt $BaselinePythonMaxMB })
    foreach ($p in $heavy) {
        Write-Host ("      大内存 python PID {0} 私有 {1} MB → 结束（超出 {2} MB 基线）" -f `
            $p.Id, [math]::Round($p.PrivateMemorySize64 / 1MB), $BaselinePythonMaxMB)
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
    if ($heavy.Count -eq 0) { Write-Host "      无残留大内存 python" }
} else {
    $newOnes = @(Get-Process python -ErrorAction SilentlyContinue | Where-Object { $baseline -notcontains $_.Id })
    if ($newOnes.Count -gt 0) {
        Write-Host "  发现 $($newOnes.Count) 个非基线 python 进程："
        foreach ($p in $newOnes) {
            $mb = [math]::Round($p.PrivateMemorySize64 / 1MB)
            Write-Host ("      PID {0,-6} 私有 {1,6} MB → 结束" -f $p.Id, $mb)
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Seconds 1
        $left = @(Get-Process python -ErrorAction SilentlyContinue | Where-Object { $baseline -notcontains $_.Id })
        if ($left.Count -gt 0) { Write-Host "  [警告] 仍有 $($left.Count) 个未结束" -ForegroundColor Yellow; $script:failed = $true }
    } else {
        Write-Host "  无非基线 python 进程（干净）"
    }
}

# ---------- 3) 端口释放确认（以 HTTP 探活为准） ----------
Start-Sleep -Seconds 2
foreach ($port in @($apiPort, 8201)) {
    $alive = $false
    try { if ((Invoke-WebRequest "http://127.0.0.1:$port/healthz" -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200) { $alive = $true } }
    catch { $mq = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue; $alive = [bool]$mq }
    if ($alive) { Write-Host "  [警告] 端口 $port 仍有服务在响应 —— 未停干净" -ForegroundColor Yellow; $script:failed = $true }
    else { Write-Host "  [OK] 端口 $port 已释放" }
}

# ---------- 4) python 进程回归基线 ----------
$all = @(Get-Process python -ErrorAction SilentlyContinue)
Write-Host "  [python 进程] 当前 $($all.Count) 个（会话基线 $($baseline.Count) 个）："
foreach ($p in $all) {
    $mb = [math]::Round($p.PrivateMemorySize64 / 1MB)
    $flag = if ($mb -gt $BaselinePythonMaxMB) { ' ← 超基线(>200MB)' } else { '' }
    Write-Host ("      PID {0,-6} 私有 {1,6} MB{2}" -f $p.Id, $mb, $flag)
    if ($mb -gt $BaselinePythonMaxMB) { $script:failed = $true }
}

# ---------- 5) 缓存目录清零 ----------
$caches = @(Get-ChildItem -LiteralPath $repoRoot -Recurse -Directory -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in @('__pycache__', '.pytest_cache') })
foreach ($c in $caches) { Remove-Item -LiteralPath $c.FullName -Recurse -Force -ErrorAction SilentlyContinue }
Write-Host "  [缓存] 清理 __pycache__ / .pytest_cache：$($caches.Count) 个"

# ---------- 6) 可选：清卷 ----------
if (-not $KeepVolumes) {
    $volRoot = Join-Path $repoRoot '部署\卷'
    if (Test-Path $volRoot) { Write-Host "  [清理] 删除卷目录 $volRoot"; Remove-Item -LiteralPath $volRoot -Recurse -Force -ErrorAction SilentlyContinue }
}

if ($script:failed) { Write-Host "[未完成] 有进程/端口/缓存未达标 —— 请检查上面标红项" -ForegroundColor Yellow; exit 1 }
Write-Host "[OK] 已停止（进程已退出、端口已释放、python 回归基线）"
exit 0
