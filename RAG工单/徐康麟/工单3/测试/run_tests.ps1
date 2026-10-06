# =============================================================================
# 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 用途：测试三级套件执行入口（离线 / 在线 / 用户），并把原始 stdout 留痕到 测试/留痕/。
#
# 设计依据：设计/验收标准.md 验收 9「离线 / 在线 / 用户三级测试全部通过并留痕」，
#           复现命令 = pytest 离线 + pytest 在线 + simulate_user。
#
# 用法（工作目录 = E:\gao6gongdan\工单3）：
#   pwsh -NoProfile -File 测试/run_tests.ps1                 # 三级全跑
#   pwsh -NoProfile -File 测试/run_tests.ps1 -Tier offline   # 只跑离线级
#   pwsh -NoProfile -File 测试/run_tests.ps1 -Tier user      # 用户级（pytest + simulate_user）
#   pwsh -NoProfile -File 测试/run_tests.ps1 -Extra '-k','t7'  # 追加 pytest 参数
#
# 纪律：
#   * 本脚本只写 测试/留痕/（tester 独占目录），**不碰** 优化/评估结果/ 与 工单1/2；
#   * 任何一段失败即整体退出码非 0（禁止「部分通过也算过」）；
#   * 输出首行固定「RAGAS 未运行（依赖不可用，本机断网）」，供验收 8/9 复核。
# =============================================================================
param(
    [ValidateSet('offline', 'online', 'user', 'all')]
    [string] $Tier = 'all',
    [string[]] $Extra = @()
)

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent $PSScriptRoot          # 工单3
$entry = Join-Path $repoRoot 'run_py.ps1'
$traceDir = Join-Path $PSScriptRoot '留痕'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
if (-not (Test-Path -LiteralPath $traceDir)) { New-Item -ItemType Directory -Path $traceDir | Out-Null }

$banner = 'RAGAS 未运行（依赖不可用，本机断网）'
Write-Host "[测试入口] 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化"
Write-Host "[测试入口] $banner"
Write-Host "[测试入口] 级别 = $Tier；留痕目录 = 测试/留痕；时间戳 = $stamp"

$results = @()

function Invoke-PytestTier {
    param([string] $Name, [string] $Target)
    $log = Join-Path $traceDir "$Name`_$stamp.txt"
    $banner | Out-File -FilePath $log -Encoding utf8
    "级别：$Name；目标：$Target；开始：$(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
    Write-Host "`n===== [$Name] pytest $Target ====="
    $pyArgs = @('-m', 'pytest', $Target, '-v') + $Extra
    # 先把子进程输出**收进变量**再 tee：否则输出会混进函数返回值，
    # 让 `(Invoke-PytestTier ...) -ne 0` 拿数组比整数 → 永远判失败（本脚本曾有此 bug）。
    $captured = & pwsh -NoProfile -File $entry @pyArgs 2>&1
    $code = $LASTEXITCODE
    $captured | Tee-Object -FilePath $log -Append
    "级别：$Name；退出码：$code" | Out-File -FilePath $log -Append -Encoding utf8
    $script:results += [pscustomobject]@{ tier = $Name; target = $Target; exit_code = $code; log = "测试/留痕/$Name`_$stamp.txt" }
    return [int]$code
}

function Invoke-UserScript {
    $log = Join-Path $traceDir "user_simulate_$stamp.txt"
    $banner | Out-File -FilePath $log -Encoding utf8
    Write-Host "`n===== [user] simulate_user.py ====="
    $pyArgs = @('测试/用户/simulate_user.py') + $Extra
    $captured = & pwsh -NoProfile -File $entry @pyArgs 2>&1
    $code = $LASTEXITCODE
    $captured | Tee-Object -FilePath $log -Append
    "级别：user-script；退出码：$code" | Out-File -FilePath $log -Append -Encoding utf8
    $script:results += [pscustomobject]@{ tier = 'user-script'; target = '测试/用户/simulate_user.py'; exit_code = $code; log = "测试/留痕/user_simulate_$stamp.txt" }
    return [int]$code
}

$failed = $false
switch ($Tier) {
    'offline' { if ((Invoke-PytestTier -Name 'offline' -Target '测试/离线') -ne 0) { $failed = $true } }
    'online'  { if ((Invoke-PytestTier -Name 'online'  -Target '测试/在线') -ne 0) { $failed = $true } }
    'user'    {
        if ((Invoke-PytestTier -Name 'user' -Target '测试/用户') -ne 0) { $failed = $true }
        if ((Invoke-UserScript) -ne 0) { $failed = $true }
    }
    'all' {
        if ((Invoke-PytestTier -Name 'offline' -Target '测试/离线') -ne 0) { $failed = $true }
        if ((Invoke-PytestTier -Name 'online'  -Target '测试/在线') -ne 0) { $failed = $true }
        if ((Invoke-PytestTier -Name 'user'    -Target '测试/用户') -ne 0) { $failed = $true }
        if ((Invoke-UserScript) -ne 0) { $failed = $true }
    }
}

$summary = [pscustomobject]@{
    work_order = '人工智能NLP-RAG-PDF文档的表格解析及检索优化'
    tier       = $Tier
    stamp      = $stamp
    ragas      = $banner
    failed     = $failed
    results    = $results
}
$summaryPath = Join-Path $traceDir "run_tests_$stamp.json"
$summary | ConvertTo-Json -Depth 6 | Out-File -FilePath $summaryPath -Encoding utf8

Write-Host "`n[测试入口] 汇总：$summaryPath"
Write-Host "[测试入口] $banner"
foreach ($row in $results) {
    $mark = if ($row.exit_code -eq 0) { '✅' } else { '❌' }
    Write-Host "  $mark $($row.tier)（$($row.target)）退出码 $($row.exit_code)"
}
if ($failed) { exit 1 } else { exit 0 }
