# =============================================================================
# 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
# 文件：部署/脚本/本地停止.ps1 —— 停止本机等价验证起的进程（对应 compose down）
#
# 用法：
#     pwsh -NoProfile -File 部署/脚本/本地停止.ps1
#     pwsh -NoProfile -File 部署/脚本/本地停止.ps1 -KeepVolumes:$false   # 连"卷"目录一起清
#
# 只杀本地进程.json 里记录的 PID（以及它们的子进程），**不做** 全端口扫描杀进程 ——
# 避免误伤其它工单/用户的进程（共享约定 §4 纪律）。
# =============================================================================
[CmdletBinding()]
param(
    [switch]$KeepVolumes = $true
)

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$logDir = Join-Path $repoRoot '部署\日志'
$pidFile = Join-Path $logDir '本地进程.json'
if (-not (Test-Path $pidFile)) {
    Write-Host "[跳过] 没有 $pidFile，说明本地等价验证进程未启动过"
    exit 0
}

$info = Get-Content -LiteralPath $pidFile -Raw -Encoding utf8 | ConvertFrom-Json
Write-Host "===== 停止本地等价验证进程 ====="
Write-Host "启动时间：$($info.started_at)"
$script:stopFailed = $false

foreach ($item in @(@{ n = 'rag-api'; pid = $info.api_pid }, @{ n = 'rag-ui'; pid = $info.ui_pid })) {
    $p = Get-Process -Id $item.pid -ErrorAction SilentlyContinue
    if ($null -eq $p) {
        Write-Host "  [$($item.n)] PID $($item.pid) 已不存在"
        continue
    }
    # 先用 Stop-Process（本会话内启动的进程可直接停），失败再退回 taskkill。
    # ⚠️ 实测：taskkill 在沙箱下可能返回 "ERROR: Access denied"（对**会话开始前就存在**的
    #    进程），此时**必须如实报失败**，不能因为"命令跑完了"就打印 [OK] 已停止。
    Stop-Process -Id $item.pid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
    if (Get-Process -Id $item.pid -ErrorAction SilentlyContinue) {
        & taskkill /PID $item.pid /T /F 2>&1 | ForEach-Object { Write-Host "  [$($item.n)] $_" }
    }
    if (Get-Process -Id $item.pid -ErrorAction SilentlyContinue) {
        Write-Host "  [失败] $($item.n) PID $($item.pid) **仍在运行**（沙箱权限不足，需在宿主侧手动结束）" -ForegroundColor Red
        $script:stopFailed = $true
    } else {
        Write-Host "  [OK] $($item.n) PID $($item.pid) 已停止"
    }
}

# 确认端口已释放（以 HTTP 探活为准：沙箱下 Get-NetTCPConnection 会漏报监听端口）
Start-Sleep -Seconds 2
foreach ($port in @($info.api_port, $info.ui_port)) {
    $alive = $false
    try {
        $r = Invoke-WebRequest "http://127.0.0.1:$port/healthz" -TimeoutSec 3 -UseBasicParsing
        $alive = ($r.StatusCode -eq 200)
    } catch {
        $mq = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        $alive = [bool]$mq
    }
    if ($alive) {
        Write-Host "  [警告] 端口 $port 仍有服务在响应 —— 未停干净" -ForegroundColor Yellow
        $script:stopFailed = $true
    } else {
        Write-Host "  [OK] 端口 $port 已释放"
    }
}

if (-not $KeepVolumes) {
    $volRoot = Join-Path $repoRoot '部署\卷'
    if (Test-Path $volRoot) {
        Write-Host "  [清理] 删除卷目录 $volRoot（索引联接一并删，不影响工单7 真实索引）"
        # 只删联接本身，绝不动目标（Remove-Item 对 Junction 默认不递归目标）
        cmd /c rmdir "$volRoot\index" 2>&1 | Out-Null
        Remove-Item -LiteralPath $volRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
if ($script:stopFailed) {
    Write-Host "[未完成] 有进程/端口未能停止 —— 请在宿主侧手动结束（沙箱权限不足）"
    exit 1
}
Write-Host "[OK] 已停止（全部进程已退出、端口已释放）"
exit 0
