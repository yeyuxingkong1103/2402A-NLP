# =============================================================================
# 工单：人工智能NLP-RAG 项目-Embedding 模型微调任务（工单11）
# 文件：部署/脚本/一键流水线.ps1
# 用途：一键执行「数据集生成 → 微调训练 → 微调前后评估 → 生成截图 → 单元测试」。
#       全程 CPU，不触碰 GPU；每步失败即停并保留日志。
#
# 用法：
#   pwsh -NoProfile -File 部署/脚本/一键流水线.ps1
#   pwsh -NoProfile -File 部署/脚本/一键流水线.ps1 -跳过训练      # 复用已有模型
#   pwsh -NoProfile -File 部署/脚本/一键流水线.ps1 -仅数据集
# =============================================================================
param(
    [switch]$跳过数据集,
    [switch]$跳过训练,
    [switch]$跳过评估,
    [switch]$跳过测试,
    [switch]$仅数据集
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # 部署/脚本 -> 工单11
Set-Location -LiteralPath $repo

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:WO11_ROOT = $repo

$logDir = Join-Path $repo '部署\日志'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $logDir "一键流水线_$stamp.log"

function 记录([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

function 跑([string]$标题, [string[]]$参数) {
    记录 "===== $标题 ====="
    $t0 = Get-Date
    & pwsh -NoProfile -File (Join-Path $repo 'run_py.ps1') @参数 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir ("{0}_{1}.log" -f $标题, $stamp)) |
        ForEach-Object { $_ }
    if ($LASTEXITCODE -ne 0) {
        记录 "!! $标题 失败，退出码 $LASTEXITCODE，流水线中止"
        exit $LASTEXITCODE
    }
    记录 ("{0} 完成，用时 {1:N1} 秒" -f $标题, ((Get-Date) - $t0).TotalSeconds)
}

记录 "工单11 一键流水线启动（工作目录 $repo）"
$mem = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
记录 ("启动时可用内存 {0:N0} MB" -f $mem)

if (-not $仅数据集) {
    if (-not $跳过数据集) { 跑 '①数据集生成' @('研发/src/generate_dataset.py') }
    if (-not $跳过训练)   { 跑 '②微调训练'   @('研发/src/train.py') }
    if (-not $跳过评估)   { 跑 '③微调前后评估' @('研发/src/evaluate.py') }
    记录 '===== ④生成测试截图 ====='
    & pwsh -NoProfile -File (Join-Path $repo 'run_py.ps1') '测试/生成截图.py' 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir "④测试截图_$stamp.log")
    if (-not $跳过测试) {
        记录 '===== ⑤单元测试 ====='
        & pwsh -NoProfile -File (Join-Path $repo 'run_py.ps1') '-m' 'pytest' '测试' '-q' 2>&1 |
            Tee-Object -FilePath (Join-Path $logDir "⑤单元测试_$stamp.log")
        if ($LASTEXITCODE -ne 0) { 记录 "!! 单元测试未全部通过（退出码 $LASTEXITCODE）" }
    }
} else {
    跑 '①数据集生成' @('研发/src/generate_dataset.py')
}

# 收尾：清理缓存目录
foreach ($d in @('__pycache__', '.pytest_cache')) {
    Get-ChildItem -LiteralPath $repo -Recurse -Force -Directory -Filter $d -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}
记录 "缓存目录已清理"
记录 "流水线全部完成；日志：$log"
