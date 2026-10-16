# =============================================================================
# 工单：人工智能NLP-RAG-基于Graph RAG 实现金融问答（工单8）
# 脚本：一键跑通（构建图谱 → 可视化 → 评估 → 对比 → 截图 → 测试 → 核验）
#
# 用法：
#   pwsh -NoProfile -File 部署\脚本\一键跑通.ps1                     # 全流程
#   pwsh -NoProfile -File 部署\脚本\一键跑通.ps1 -SkipEval           # 跳过 17 题评估（省约 25 分钟）
#   pwsh -NoProfile -File 部署\脚本\一键跑通.ps1 -SkipBuild          # 复用已有图谱
#   pwsh -NoProfile -File 部署\脚本\一键跑通.ps1 -LlmSample 400      # 额外做 Ollama 关系抽样补抽
#
# 资源纪律（_共享约定.md §5.5）：
#   * 启动前探内存（可用 < 4000 MB 拒绝启动，除非 -SkipMemoryCheck）；
#   * 全程**单进程**、无进程池；Ollama 每次请求 keep_alive="0s"（用完即卸载）；
#   * 长日志一律重定向到 部署\日志\。
# =============================================================================
param(
    [switch]$SkipMemoryCheck,
    [switch]$SkipBuild,
    [switch]$SkipViz,
    [switch]$SkipEval,
    [switch]$SkipTests,
    [int]$LlmSample = 0
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repo
$logDir = Join-Path $repo '部署\日志'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Step([string]$title) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    Write-Host "[工单8] $title" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
}

function Run([string]$desc, [string[]]$pyArgs, [string]$logName) {
    Write-Host ">>> $desc" -ForegroundColor Green
    $log = Join-Path $logDir $logName
    & pwsh -NoProfile -File (Join-Path $repo 'run_py.ps1') @pyArgs *> $log
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Warning "$desc 失败（退出码 $code），日志：$log"
    } else {
        Write-Host "    完成（日志：$log）" -ForegroundColor DarkGray
    }
    return $code
}

# ---- 0) 内存探测 ----------------------------------------------------------
if (-not $SkipMemoryCheck) {
    $free = (Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue
    Write-Host ("[工单8] 可用内存：{0:N0} MB" -f $free)
    if ($free -lt 4000) {
        Write-Warning "[工单8] 可用内存不足 4000 MB，按资源纪律拒绝启动（可用 -SkipMemoryCheck 强制）。"
        exit 3
    }
}
$env:PYTHONDONTWRITEBYTECODE = '1'
$started = Get-Date

# ---- 1) 构建知识图谱 ------------------------------------------------------
if (-not $SkipBuild) {
    Step "① 构建知识图谱（规则/词典/正则抽取，全量 7430 块，约 5~10 s）"
    $null = Run "build_graph" @('研发/scripts/build_graph.py', '--llm-sample', "$LlmSample") 'build_graph_stdout.log'
}

# ---- 2) 图谱可视化 --------------------------------------------------------
if (-not $SkipViz) {
    Step "② 生成图谱可视化（力导向 PNG + 自包含 SVG/HTML）"
    $null = Run "render_graph_viz" @('研发/scripts/render_graph_viz.py') 'render_viz.log'
}

# ---- 3) 17 题评估（含 Ollama 生成，约 25 分钟）----------------------------
if (-not $SkipEval) {
    Step "③ 17 题 Graph RAG 评估（含 qwen2.5:3b 生成，约 25 分钟）"
    Write-Host "    提示：期间请勿并行启动其它重活（资源纪律）。" -ForegroundColor Yellow
    $null = Run "run_eval" @('-u', '研发/评估/run_eval.py') 'run_eval_stdout.log'
}

# ---- 4) 与工单7 对比 ------------------------------------------------------
Step "④ 生成与工单7 的并排对比表"
$null = Run "build_comparison" @('研发/scripts/build_comparison.py') 'build_comparison.log'

# ---- 5) 优化前后对比 ------------------------------------------------------
Step "⑤ 生成优化前后对比数据"
$null = Run "优化对比" @('优化/脚本/生成优化对比.py') 'opt_compare.log'

# ---- 6) 测试截图 ----------------------------------------------------------
Step "⑥ 生成测试截图（中文按测试内容命名，真实数字）"
$null = Run "截图" @('测试/生成截图.py') 'shots.log'

# ---- 7) pytest ------------------------------------------------------------
if (-not $SkipTests) {
    Step "⑦ 运行 pytest 用例"
    $null = Run "pytest" @('-m', 'pytest', '测试', '-q') 'pytest.log'
}

# ---- 8) 交付物核验 --------------------------------------------------------
Step "⑧ 交付物核验"
$null = Run "核验" @('部署/脚本/核验交付物.py') 'verify.log'

# ---- 9) 清理缓存 ----------------------------------------------------------
Step "⑨ 清理缓存目录（__pycache__ / .pytest_cache）"
Get-ChildItem -Path $repo -Recurse -Directory -Include '__pycache__', '.pytest_cache' -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
Write-Host "    清理完成" -ForegroundColor DarkGray

$elapsed = (Get-Date) - $started
Step ("全流程完成，总耗时 {0:N1} 分钟" -f $elapsed.TotalMinutes)
Write-Host "产物位置：" -ForegroundColor Cyan
Write-Host "  · 图谱：研发\data\graph\（entities/relations/graph.json/graphml/svg/html）"
Write-Host "  · 评估：测试\测试数据\（GraphRAG评估*.json(l) / 与工单7对比.*）"
Write-Host "  · 截图：测试\测试截图\（5 张中文命名 PNG）"
Write-Host "  · 日志：部署\日志\"
Write-Host "  · 核验：部署\交付物核验.md"
Write-Host ""
Write-Host "启动问答界面：pwsh -NoProfile -File 部署\脚本\启动界面.ps1" -ForegroundColor Yellow
exit 0
