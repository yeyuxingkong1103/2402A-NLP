# =============================================================================
# 工单：人工智能NLP-RAG-图像内容解析及检索优化
# 文件：测试/run_tests.ps1 —— 测试三级一键串跑（离线 → 在线 → 用户）
#
# 用途：验收人一条命令跑完全部测试层级，给出汇总、退出码，并写机器可读结果
#       （测试/测试数据/三级串跑结果.json，供 测试/测试报告.md §11.1 引用）。
#   · 离线层：不依赖网络/LLM/服务（golden 契约 + 离线验收）
#   · 在线层：依赖 Ollama 与 Web 服务（服务未启动时本脚本**自行拉起**，跑完自动关闭）
#   · 用户层：模拟验收人照 README 走一遍（CLI/健康检查/交付物/验收判据）
#
# 用法：pwsh -NoProfile -File 测试/run_tests.ps1
#       pwsh -NoProfile -File 测试/run_tests.ps1 -SkipOnline
# =============================================================================
param([switch]$SkipOnline)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

# 防止在仓库根留下 .pytest_cache 游离目录（交付物只落在五个中文文件夹内）
$env:PYTEST_ADDOPTS = '-p no:cacheprovider'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$results = [ordered]@{}
$details = New-Object System.Collections.ArrayList
$serverJob = $null
$startedByUs = $false

function Invoke-Level([string]$name, [string[]]$targets) {
    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host "[$name] pytest $($targets -join ' ')"
    Write-Host ("=" * 78)
    $out = & pwsh -NoProfile -File run_py.ps1 -m pytest @targets -q 2>&1 | Out-String
    $code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    Write-Host $out
    $summary = ($out -split "`r?`n" | Where-Object { $_ -match '(passed|failed|error|no tests ran)' } |
                Select-Object -Last 1)
    if (-not $summary) { $summary = '（未捕获到 pytest 汇总行）' }
    $script:results[$name] = $code
    [void]$script:details.Add([ordered]@{ name = $name; targets = ($targets -join ' ');
                                          exit_code = $code; summary = $summary.Trim() })
}

function Test-Health {
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8123/health' -UseBasicParsing -TimeoutSec 3
        return $r.StatusCode -eq 200
    } catch { return $false }
}

# ---------- 1) 离线层（不依赖服务）----------
Invoke-Level '离线层' @('研发/tests', '测试/离线')

# ---------- 2) 在线层（依赖 Ollama + Web 服务）----------
if (-not $SkipOnline) {
    if (-not (Test-Health)) {
        Write-Host "[在线层] 服务未启动，正在拉起 研发/app/main.py --serve（8123）…"
        $serverJob = Start-Job -ScriptBlock {
            param($r)
            Set-Location -LiteralPath $r
            $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUTF8 = '1'
            & pwsh -NoProfile -File run_py.ps1 研发/app/main.py --serve --port 8123 2>&1
        } -ArgumentList $root
        $startedByUs = $true
        $ok = $false
        $deadline = (Get-Date).AddSeconds(120)      # 硬截止：最多等 120 秒
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 2
            if (Test-Health) { $ok = $true; break }
            if (netstat -ano | Select-String ':8123.*LISTENING') { Start-Sleep -Seconds 1 }
        }
        if (-not $ok) {
            Write-Host "[在线层] 服务 120s 内未就绪——如实标记失败（不跳过、不伪造）"
            $script:results['在线层'] = 1
            [void]$script:details.Add([ordered]@{ name = '在线层'; targets = '测试/在线';
                                                  exit_code = 1; summary = '服务 120s 内未就绪（未执行用例）' })
        }
    }
    if (-not $script:results.Contains('在线层')) {
        Invoke-Level '在线层' @('测试/在线')
    }
} else {
    Write-Host "[在线层] 已按 -SkipOnline 跳过（报告须标注该口径）"
}

# ---------- 3) 用户层 ----------
Invoke-Level '用户层' @('测试/用户')

# ---------- 汇总 + 机器可读结果 ----------
Write-Host ""
Write-Host ("=" * 78)
Write-Host "测试三级串跑汇总"
Write-Host ("=" * 78)
$failed = 0
foreach ($k in $results.Keys) {
    $code = $results[$k]
    $mark = if ($code -eq 0) { '通过' } else { "失败(exit=$code)" }
    if ($code -ne 0) { $failed++ }
    Write-Host ("  {0,-8} {1}" -f $k, $mark)
}

$dataDir = Join-Path $root '测试/测试数据'
if (Test-Path $dataDir) {
    $payload = [ordered]@{
        generated_by    = '测试/run_tests.ps1'
        finished_at     = (Get-Date).ToString('s')
        exit_code       = $failed
        skip_online     = [bool]$SkipOnline
        service_started_by_script = $startedByUs
        levels          = $details
        note            = '三级串跑（离线 → 在线 → 用户）；exit_code 0 = 全部通过。数字来自本次实跑，未预填。'
    }
    ($payload | ConvertTo-Json -Depth 5) | Set-Content -LiteralPath (Join-Path $dataDir '三级串跑结果.json') -Encoding UTF8
    Write-Host "[OK] 已写 测试/测试数据/三级串跑结果.json"
}

if ($startedByUs -and $serverJob) {
    Write-Host "[清理] 关闭本脚本拉起的服务进程…"
    Stop-Job -Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job -Job $serverJob -Force -ErrorAction SilentlyContinue
    $pids = (netstat -ano | Select-String ':8123.*LISTENING' |
             ForEach-Object { ($_.Line -split '\s+')[-1] } | Sort-Object -Unique)
    foreach ($p in $pids) { if ($p -match '^\d+$') { Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue } }
}

if ($failed -gt 0) {
    Write-Host "结论：存在失败层级，退出码 1" -ForegroundColor Red
    exit 1
}
Write-Host "结论：三级全部通过" -ForegroundColor Green
exit 0
