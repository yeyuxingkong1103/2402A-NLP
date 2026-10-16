# 工单9 一键跑通（完整流水线）

# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
#
# 顺序（与 设计/系统架构.md §4 一致）：
#   0) 探内存（资源纪律：可用 <2000 MB 或提交比 >80% 就别启动）
#   1) 构建参考答案集（PDF → jsonl + 与 17 题集逐条核对）
#   2) 抽取层优化（工单8 图谱 → 优化后图谱）
#   3) 单元测试（pytest，纯函数，秒级）
#   4) 全量对比评估（优化前 vs 优化后，含 LLM；-SkipEval 可跳过）
#   5) 稳定性 + 容错（连续 50 次查询）
#   6) 多语言实测
#   7) 截图
#
# 用法：
#   pwsh -NoProfile -File 部署/脚本/一键跑通.ps1
#   pwsh -NoProfile -File 部署/脚本/一键跑通.ps1 -SkipEval -Repeat 10

[CmdletBinding()]
param(
    [switch]$SkipEval,
    [int]$Repeat = 50,
    [switch]$SkipStability
)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$env:PYTHONDONTWRITEBYTECODE = '1'
$LogDir = Join-Path $Root '部署\日志'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Step([string]$Text) {
    Write-Host ''
    Write-Host ('=' * 74) -ForegroundColor DarkCyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ('=' * 74) -ForegroundColor DarkCyan
}

function Invoke-Py([string]$Script, [string]$LogName, [string[]]$Extra = @()) {
    $log = Join-Path $LogDir $LogName
    Write-Host "  > run_py.ps1 $Script $($Extra -join ' ')  =>  $LogName" -ForegroundColor DarkGray
    $args = @('-NoProfile', '-File', 'run_py.ps1', $Script) + $Extra
    & pwsh @args *> $log
    $code = $LASTEXITCODE
    if ($code -ne 0) { Write-Host "  [警告] $Script 退出码 $code（详见 部署/日志/$LogName）" -ForegroundColor Yellow }
    return $code
}

# ---- 0) 探内存 -------------------------------------------------------------
Write-Step '0) 资源探针（资源纪律 §5.5）'
$avail = (Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue
$commit = (Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit').CounterSamples.CookedValue
$ratio = [math]::Round($commit[0] / $commit[1] * 100, 1)
Write-Host "  可用内存: $([math]::Round($avail,0)) MB    提交比: $ratio %"
if ($avail -lt 2000 -or $ratio -gt 80) {
    Write-Host '  [中止] 可用内存 < 2000 MB 或提交比 > 80% —— 请先释放内存再跑。' -ForegroundColor Red
    exit 2
}

# ---- 1) 参考答案集 ---------------------------------------------------------
Write-Step '1/7 构建参考答案集（sample_questions.pdf → jsonl）'
Invoke-Py '研发/scripts/构建参考答案集.py' '一键_1_参考答案集.log' | Out-Null

# ---- 2) 抽取层优化 ---------------------------------------------------------
Write-Step '2/7 抽取层优化（别名归并 / 关系规范化 / 噪声剪枝）'
Invoke-Py '研发/scripts/构建优化图谱.py' '一键_2_构建优化图谱.log' | Out-Null

# ---- 3) 单元测试 -----------------------------------------------------------
Write-Step '3/7 单元测试（pytest）'
$pytestLog = Join-Path $LogDir '一键_3_pytest.log'
& pwsh -NoProfile -File run_py.ps1 -m pytest 测试 -q *> $pytestLog
Get-Content $pytestLog -Tail 4 | ForEach-Object { Write-Host "  $_" }

# ---- 4) 全量对比评估 -------------------------------------------------------
if ($SkipEval) {
    Write-Step '4/7 全量对比评估（-SkipEval，已跳过）'
} else {
    Write-Step '4/7 全量对比评估（优化前 vs 优化后，含 LLM，约 2~4 分钟）'
    Invoke-Py '研发/评估/run_eval_opt.py' 'run_eval_opt.log' @('--profiles', 'before,after') | Out-Null
    $cmp = Join-Path $Root '优化\评估结果\优化前后对比.md'
    if (Test-Path $cmp) {
        Get-Content $cmp | Select-String -Pattern '上下文精度|上下文召回|SLA' | Select-Object -First 6 |
            ForEach-Object { Write-Host "  $_" }
    }
}

# ---- 5) 稳定性 + 容错 ------------------------------------------------------
if ($SkipStability) {
    Write-Step '5/7 稳定性与容错（-SkipStability，已跳过）'
} else {
    Write-Step "5/7 稳定性与容错（连续 $Repeat 次查询 + 8 类异常场景）"
    Invoke-Py '研发/评估/稳定性压测.py' '稳定性压测.log' @('--scenario', 'all', '--repeat', "$Repeat") | Out-Null
}

# ---- 6) 多语言 -------------------------------------------------------------
Write-Step '6/7 多语言实测（中文 + 英文）'
Invoke-Py '研发/评估/多语言实测.py' '多语言实测.log' | Out-Null

# ---- 7) 截图 ---------------------------------------------------------------
Write-Step '7/7 生成测试截图'
Invoke-Py '测试/生成截图.py' '一键_7_截图.log' | Out-Null

# ---- 收尾 ------------------------------------------------------------------
Write-Step '收尾：清理缓存 + 卸载 Ollama 模型'
Get-ChildItem -Path $Root -Recurse -Directory -Include '__pycache__', '.pytest_cache' -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue }
foreach ($m in 'qwen2.5:3b', 'bge-m3:latest') {
    try {
        Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post `
            -Body (@{ model = $m; keep_alive = 0 } | ConvertTo-Json) -ContentType 'application/json' `
            -TimeoutSec 5 | Out-Null
    } catch { }
}
Write-Host '  已清理 __pycache__ / .pytest_cache 并请求卸载 Ollama 模型' -ForegroundColor Green
Write-Host ''
Write-Host '一键跑通完成。产物：设计/ 研发/ 测试/ 优化/ 部署/' -ForegroundColor Green
Write-Host '核验：pwsh -NoProfile -File 部署/脚本/核验交付物.py' -ForegroundColor Green
