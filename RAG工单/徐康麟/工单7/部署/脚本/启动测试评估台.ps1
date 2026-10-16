# =============================================================================
# 工单编号：人工智能NLP-RAG-功能测试及评估
# 文件：部署/脚本/启动测试评估台.ps1
#
# 用途：设好本工单的全部环境变量后，启动「测试评估台」Web 界面（纯标准库实现），
#       并把 stdout/stderr 重定向到 部署/日志/界面.log。
#
# 用法（在任意目录）：
#     pwsh -NoProfile -File 部署/脚本/启动测试评估台.ps1
#     pwsh -NoProfile -File 部署/脚本/启动测试评估台.ps1 -Port 7862
#     pwsh -NoProfile -File 部署/脚本/启动测试评估台.ps1 -NoWarmup      # 不预热，首问较慢
#     pwsh -NoProfile -File 部署/脚本/启动测试评估台.ps1 -Background    # 后台起，立即返回
#
# 参数：
#     -Port <int>    监听端口（默认 7861）
#     -Host <str>    监听地址（默认 127.0.0.1，仅本机）
#     -NoWarmup      跳过索引/模型预热
#     -Background    后台启动（返回 job/进程号，日志仍写 部署/日志/界面.log）
#
# 说明：
#   * 本机无 gradio / streamlit / pandas，且工单纪律禁止 pip install，
#     故界面用标准库 http.server + 内嵌 HTML/JS 实现（部署/界面/测试评估台.py）。
#   * Ollama 是 CPU 单实例：界面只在人工点击时单题调用，且内部串行化。
# =============================================================================
param(
    [int]$Port = 7861,
    [string]$HostAddr = '127.0.0.1',
    [switch]$NoWarmup,
    [switch]$Background
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '环境.ps1')

$app = Join-Path $env:GD7_ROOT '部署\界面\测试评估台.py'
if (-not (Test-Path -LiteralPath $app)) {
    throw "找不到界面脚本：$app"
}

$logDir = Join-Path $env:GD7_ROOT '部署\日志'
if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logFile = Join-Path $logDir '界面.log'

# 只清理自己端口上的残留监听（不碰别人的进程）
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    $pids = $existing | Select-Object -ExpandProperty OwningProcess -Unique
    Write-Host "[启动] 端口 $Port 已被 PID $($pids -join ',') 占用，先停止它…" -ForegroundColor Yellow
    foreach ($procId in $pids) {
        try { Stop-Process -Id $procId -Force -ErrorAction Stop; Write-Host "  已停止 PID $procId" }
        catch { Write-Host "  停止 PID $procId 失败：$_" -ForegroundColor Yellow }
    }
    Start-Sleep -Milliseconds 800
}

$pyArgs = @($app, '--host', $HostAddr, '--port', "$Port")
if ($NoWarmup) { $pyArgs += '--no-warmup' }

$url = "http://${HostAddr}:$Port/"
Write-Host ""
Write-Host ("=" * 78)
Write-Host "[工单7] 测试评估台 —— 人工智能NLP-RAG-功能测试及评估"
Write-Host ("=" * 78)
Write-Host "  界面脚本 ：$app"
Write-Host "  解释器   ：$($env:GD7_PY)"
Write-Host "  监听     ：$url"
Write-Host "  日志     ：$logFile"
Write-Host ("=" * 78)

$header = @"
# ---------------------------------------------------------------------------
# 工单7 测试评估台 启动记录  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
# 端口 $Port ｜ 预热 $(if ($NoWarmup) { '关闭' } else { '开启' })
# ---------------------------------------------------------------------------
"@
Add-Content -LiteralPath $logFile -Value $header -Encoding UTF8

if ($Background) {
    $proc = Start-Process -FilePath $env:GD7_PY -ArgumentList $pyArgs `
        -WorkingDirectory $env:GD7_ROOT -WindowStyle Hidden `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err" -PassThru
    Write-Host "[OK] 后台启动，PID = $($proc.Id)" -ForegroundColor Green
    Write-Host "     等待就绪后验证：Invoke-WebRequest $url -UseBasicParsing"
    Write-Host "     停止：Stop-Process -Id $($proc.Id) -Force"
    exit 0
}

Write-Host "[启动] 前台运行（Ctrl+C 停止）。日志同时写入 部署\日志\界面.log …" -ForegroundColor Green
& $env:GD7_PY @pyArgs 2>&1 | Tee-Object -FilePath $logFile -Append
exit $LASTEXITCODE
