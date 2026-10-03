<#
================================================================================
工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：部署 / 评估入口（本机 Windows）
================================================================================
用途
    一键跑**优化前后对比评估**：默认调用 `优化/脚本/compare_optimization.py`（T7 产出，
    负责 基线 vs 优化后 的对比与 CSV/Markdown 报告）；该脚本尚未产出时可显式加
    `-AllowFallback` 退回 `研发/scripts/evaluate.py`（只跑优化后系统的 10 题评估）。

    CLI 契约（compare_optimization.py 必须接受；T7 按此实现）
        --mode {rag,extractive}   评估模式（默认 rag=最终交付路径）
        --limit N                 仅评估前 N 题（0=全部）
        --out DIR                 输出目录（默认 优化/评估结果）
        --golden PATH             判分基准 golden_qa.jsonl（默认 测试/测试数据/golden_qa.jsonl）

参数
    -Mode           rag(默认) | extractive
    -Limit          仅评估前 N 题（0=全部）
    -Out            输出目录（空=各脚本自身默认）
    -Golden         判分基准路径（空=默认）
    -Filename       输出文件名前缀（仅回落 evaluate.py 时使用）
    -CompareScript  对比脚本路径（空=优化/脚本/compare_optimization.py）
    -AllowFallback  对比脚本缺失时回落到 研发/scripts/evaluate.py
    -Python         解释器路径（空→$env:RAG_SCHEDULER_PYTHON→工单1 .venv）
    -DryRun         只打印将执行的命令，不执行

用法示例
    pwsh -NoProfile -File 部署/脚本/evaluate.ps1 -DryRun
    pwsh -NoProfile -File 部署/脚本/evaluate.ps1                       # rag 模式全量
    pwsh -NoProfile -File 部署/脚本/evaluate.ps1 -Mode extractive -Limit 1 -AllowFallback

退出码
    0 成功 | 2 环境/解释器问题 | 3 缺少脚本 | 其他 = 被调用脚本的退出码
================================================================================
#>
[CmdletBinding()]
param(
    [ValidateSet('rag', 'extractive')]
    [string]$Mode = 'rag',

    [ValidateRange(0, 1000)]
    [int]$Limit = 0,

    [string]$Out = '',
    [string]$Golden = '',
    [string]$Filename = '',
    [string]$CompareScript = '',
    [string]$Python = '',
    [switch]$AllowFallback,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$WorkspaceRoot = Split-Path $RepoRoot -Parent

if (-not $CompareScript) { $CompareScript = Join-Path $RepoRoot '优化\脚本\compare_optimization.py' }
$FallbackScript = Join-Path $RepoRoot '研发\scripts\evaluate.py'

function Write-Section([string]$Title) {
    Write-Host ''
    Write-Host "==== $Title ====" -ForegroundColor Cyan
}
function Fail([string]$Message, [int]$Code) {
    Write-Host "[失败] $Message" -ForegroundColor Red
    exit $Code
}

Write-Section '工单2 部署 · 优化前后对比评估'
Write-Host "  仓库根目录          $RepoRoot"
Write-Host "  模式 / 题数         $Mode / $(if ($Limit -eq 0) { '全部' } else { "前 $Limit 题" })"

# ---- 解释器 ----
if ($Python) {
    if (-not (Test-Path -LiteralPath $Python)) { Fail "指定的解释器不存在：$Python" 2 }
    $PythonPath = (Resolve-Path -LiteralPath $Python).Path
}
elseif ($env:RAG_SCHEDULER_PYTHON) {
    if (-not (Test-Path -LiteralPath $env:RAG_SCHEDULER_PYTHON)) {
        Fail "RAG_SCHEDULER_PYTHON 指向的解释器不存在：$($env:RAG_SCHEDULER_PYTHON)" 2
    }
    $PythonPath = (Resolve-Path -LiteralPath $env:RAG_SCHEDULER_PYTHON).Path
}
else {
    $venv = Join-Path $WorkspaceRoot '工单1\.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venv) { $PythonPath = (Resolve-Path -LiteralPath $venv).Path }
    else {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $PythonPath = $cmd.Source } else { Fail '未找到 Python 解释器（用 -Python 指定）' 2 }
    }
}
Write-Host "  Python 解释器       $PythonPath"

# ---- 选择脚本：对比脚本优先，缺失时按需回落 ----
$selected = ''
$kind = ''
if (Test-Path -LiteralPath $CompareScript) {
    $selected = $CompareScript
    $kind = 'compare'
    Write-Host "  对比脚本            $CompareScript" -ForegroundColor Green
}
elseif ($AllowFallback) {
    if (-not (Test-Path -LiteralPath $FallbackScript)) { Fail "回落脚本也不存在：$FallbackScript" 3 }
    $selected = $FallbackScript
    $kind = 'fallback'
    Write-Host "  对比脚本缺失：$CompareScript" -ForegroundColor Yellow
    Write-Host "  '-> 已按 -AllowFallback 回落到 $FallbackScript（仅评估优化后系统，不做基线对比）" -ForegroundColor Yellow
}
else {
    Write-Host "  对比脚本缺失：$CompareScript" -ForegroundColor Red
    Write-Host '  T7 尚未产出该脚本。可选：' -ForegroundColor Yellow
    Write-Host '    ① 等 T7 产出后重跑本脚本；'
    Write-Host '    ② 加 -AllowFallback 退回 研发/scripts/evaluate.py 先跑优化后系统评估。'
    Fail '缺少对比脚本（见上方说明）' 3
}

# ---- 组装参数 ----
$argList = @($selected)
if ($kind -eq 'fallback') {
    $argList += @('--mode', $Mode)
    if ($Limit -gt 0) { $argList += @('--limit', "$Limit") }
    if ($Out) { $argList += @('--out', $Out) }
    if ($Golden) { $argList += @('--golden', $Golden) }
    if ($Filename) { $argList += @('--filename', $Filename) }
}
else {
    $argList += @('--mode', $Mode)
    if ($Limit -gt 0) { $argList += @('--limit', "$Limit") }
    if ($Out) { $argList += @('--out', $Out) }
    if ($Golden) { $argList += @('--golden', $Golden) }
}

Write-Section '将要执行的命令'
$pretty = ($argList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
Write-Host "  `"$PythonPath`" $pretty"
if (-not $Out) { Write-Host '  输出目录：脚本默认（通常为 优化/评估结果）' }

if ($DryRun) {
    Write-Host ''
    Write-Host '[DryRun] 仅打印，不执行。' -ForegroundColor Yellow
    exit 0
}

Write-Section '开始评估（日志：部署\日志\{app.log,error.log,rag_trace.jsonl}）'
& $PythonPath @argList
$code = $LASTEXITCODE
Write-Host ''
Write-Host "评估结束，退出码 $code" -ForegroundColor ($(if ($code -eq 0) { 'Green' } else { 'Yellow' }))
exit $code
