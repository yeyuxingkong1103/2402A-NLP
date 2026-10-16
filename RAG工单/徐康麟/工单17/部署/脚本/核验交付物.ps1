# =============================================================================
# 工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单
# 文件：部署/脚本/核验交付物.ps1
#
# 用途：**从真实文件系统**生成 部署/交付物核验.md ——
#       逐条核对五个中文一级目录、必备文件、测试截图命名规范、以及各类产物数量。
#       不手写清单，避免「文档里写了、磁盘上没有」。
#
# 用法（工作目录 = E:\gao6gongdan\工单17）：
#     pwsh -NoProfile -File 部署/脚本/核验交付物.ps1
# =============================================================================
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot
$outFile = Join-Path $repoRoot '部署\交付物核验.md'

$lines = New-Object System.Collections.Generic.List[string]
$issues = New-Object System.Collections.Generic.List[string]
$now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

function Add-Line([string]$s) { $script:lines.Add($s) }
function Check([string]$label, [bool]$ok, [string]$detail) {
    $mark = if ($ok) { '✅' } else { '❌' }
    Add-Line("| $label | $mark | $detail |")
    if (-not $ok) { $script:issues.Add("$label —— $detail") }
}

Add-Line('# 交付物核验（工单17）')
Add-Line('')
Add-Line('**工单编号**：`人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单`')
Add-Line("**核验时间**：$now")
Add-Line('**核验方式**：由 `部署/脚本/核验交付物.ps1` 扫描真实文件系统自动生成（非手写清单）')
Add-Line('')

# ---------------------------------------------------------------- 一级目录 --
Add-Line('## 1. 五个中文一级目录（共享约定 §1 硬性）')
Add-Line('')
Add-Line('| 目录 | 状态 | 文件数 / 说明 |')
Add-Line('| --- | --- | --- |')
foreach ($d in '设计', '研发', '测试', '优化', '部署') {
    $path = Join-Path $repoRoot $d
    $files = @(Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue)
    $sizeMB = [math]::Round((($files | Measure-Object -Property Length -Sum).Sum) / 1MB, 2)
    Check $d (Test-Path $path) "$($files.Count) 个文件 / $sizeMB MB"
}
Add-Line('')

# ------------------------------------------------------------------ 设计 --
Add-Line('## 2. 设计文档（11 份）')
Add-Line('')
Add-Line('| 文件 | 状态 | 大小 |')
Add-Line('| --- | --- | --- |')
$designFiles = @('工单需求原文.md', '需求分析.md', '压测方案设计.md', '瓶颈排查方法论.md',
                 '优化方案设计.md', '系统架构.md', '接口设计.md', '技术选型.md',
                 '环境事实.md', '验收标准.md', 'README.md')
foreach ($f in $designFiles) {
    $p = Join-Path $repoRoot "设计\$f"
    $ok = Test-Path $p
    $size = if ($ok) { "$([math]::Round((Get-Item $p).Length / 1KB, 1)) KB" } else { '缺失' }
    Check "设计/$f" $ok $size
}
Add-Line('')

# ------------------------------------------------------------------ 研发 --
Add-Line('## 3. 研发源码')
Add-Line('')
Add-Line('| 项 | 状态 | 说明 |')
Add-Line('| --- | --- | --- |')
foreach ($f in '研发\service\api.py', '研发\service\perf_config.py', '研发\service\http_pool.py',
               '研发\service\singletons.py', '研发\service\limiter.py', '研发\service\task_queue.py',
               '研发\service\session_store.py', '研发\service\config.py',
               '研发\app\core\perf_hook.py', '研发\app\core\qa_engine.py', '研发\app\core\llm_client.py',
               '研发\app\core\embedder.py', '研发\app\storage\sqlite_manager.py',
               '研发\压测\load_runner.py', '研发\压测\场景配置.json', '研发\压测\问题集.jsonl',
               '研发\剖析\剖析问答链路.py', '研发\剖析\内存剖析.py', '研发\剖析\redis统计.py',
               '研发\剖析\探GPU.ps1', '研发\scripts\汇总对比.py', '研发\scripts\验证限流与容错.py',
               '研发\requirements.txt', '研发\README.md') {
    $ok = Test-Path (Join-Path $repoRoot $f)
    Check $f $ok $(if ($ok) { "$([math]::Round((Get-Item (Join-Path $repoRoot $f)).Length / 1KB, 1)) KB" } else { '缺失' })
}
# 工单编号注释
$pyFiles = @(Get-ChildItem (Join-Path $repoRoot '研发') -Recurse -Include *.py -File |
             Where-Object { $_.FullName -notlike '*基座快照*' })
$noHeader = @($pyFiles | Where-Object {
    -not (Select-String -Path $_.FullName -Pattern '工单编号' -Quiet -ErrorAction SilentlyContinue) })
Check '源码头部含工单编号注释' ($noHeader.Count -eq 0) `
    "$($pyFiles.Count) 个 .py，缺注释 $($noHeader.Count) 个$(if ($noHeader.Count) { '：' + (($noHeader | ForEach-Object Name) -join ', ') })"
# 禁用模式（大小写敏感，避免误报 threadpool( ）
$hits = @($pyFiles | Select-String -CaseSensitive -Pattern 'multiprocessing|ProcessPoolExecutor|Pool\(|joblib|n_jobs')
Check '禁用并发模式自检（0 命中）' ($hits.Count -eq 0) "$($hits.Count) 处命中"
Add-Line('')

# ------------------------------------------------------------------ 测试 --
Add-Line('## 4. 测试')
Add-Line('')
$shots = @(Get-ChildItem (Join-Path $repoRoot '测试\测试截图') -Filter *.png -File -ErrorAction SilentlyContinue)
Add-Line('### 4.1 测试截图（**≥6 张 + 中文按测试内容命名**）')
Add-Line('')
Add-Line('| 截图 | 大小 | 命名合规 |')
Add-Line('| --- | --- | --- |')
$badName = @()
foreach ($s in $shots | Sort-Object Name) {
    $compliant = ($s.BaseName -match '[\u4e00-\u9fff]') -and ($s.BaseName -notmatch '^[\d\-_]+$')
    if (-not $compliant) { $badName += $s.Name }
    Add-Line("| $($s.Name) | $([math]::Round($s.Length / 1KB, 1)) KB | $(if ($compliant) { '✅' } else { '❌ 非中文/无意义命名' }) |")
}
Check '测试截图 ≥6 张' ($shots.Count -ge 6) "实得 $($shots.Count) 张"
Check '截图命名全部中文可读' ($badName.Count -eq 0) $(if ($badName.Count) { $badName -join ', ' } else { '全部合规' })
Add-Line('')

Add-Line('### 4.2 pytest 用例与数据')
Add-Line('')
$tests = @(Get-ChildItem (Join-Path $repoRoot '测试') -Filter 'test_*.py' -File -ErrorAction SilentlyContinue)
Add-Line("| 用例文件 | 大小 |")
Add-Line("| --- | --- |")
foreach ($t in $tests | Sort-Object Name) { Add-Line("| $($t.Name) | $([math]::Round($t.Length / 1KB, 1)) KB |") }
Check 'pytest 用例存在' ($tests.Count -ge 5) "$($tests.Count) 个用例文件"
Check '测试报告存在' (Test-Path (Join-Path $repoRoot '测试\测试报告.md')) '测试/测试报告.md'

$dataFiles = @(Get-ChildItem (Join-Path $repoRoot '测试\测试数据') -File -ErrorAction SilentlyContinue)
$jsonl = @($dataFiles | Where-Object Extension -eq '.jsonl')
$jsons = @($dataFiles | Where-Object Extension -eq '.json')
$txts = @($dataFiles | Where-Object Extension -eq '.txt')
Add-Line('')
Add-Line("测试数据：$($dataFiles.Count) 个文件（.jsonl $($jsonl.Count) / .json $($jsons.Count) / .txt $($txts.Count)）")
Check '压测明细 jsonl 存在' ($jsonl.Count -ge 4) "$($jsonl.Count) 个 .jsonl"
Add-Line('')

# ------------------------------------------------------------------ 优化 --
Add-Line('## 5. 优化')
Add-Line('')
Add-Line('| 产物 | 状态 | 说明 |')
Add-Line('| --- | --- | --- |')
$report = Join-Path $repoRoot '优化\瓶颈排查与修复报告.md'
Check '优化/瓶颈排查与修复报告.md' (Test-Path $report) `
    $(if (Test-Path $report) { "$([math]::Round((Get-Item $report).Length / 1KB, 1)) KB（核心交付物）" } else { '缺失' })
foreach ($f in '基准_优化前.json', '优化后.json', '对比.csv', '对比.md') {
    $p = Join-Path $repoRoot "优化\评估结果\$f"
    $ok = Test-Path $p
    Check "优化/评估结果/$f" $ok $(if ($ok) { "$([math]::Round((Get-Item $p).Length / 1KB, 1)) KB" } else { '缺失' })
}
$optScripts = @(Get-ChildItem (Join-Path $repoRoot '优化\脚本') -File -ErrorAction SilentlyContinue)
Check '优化/脚本/ 非空' ($optScripts.Count -gt 0) "$($optScripts.Count) 个脚本"
Add-Line('')

# ------------------------------------------------------------------ 部署 --
Add-Line('## 6. 部署')
Add-Line('')
Add-Line('| 产物 | 状态 | 说明 |')
Add-Line('| --- | --- | --- |')
foreach ($f in 'DEPLOYMENT.md', 'OPERATION.md', 'README.md', '交付物核验.md',
               'docker\docker-compose.yml', 'docker\Dockerfile', 'docker\prometheus.yml',
               'docker\healthcheck.py', 'docker\entrypoint.sh',
               '脚本\起服务.ps1', '脚本\本地停止.ps1', '脚本\health.ps1',
               '配置\.env.example') {
    $p = Join-Path $repoRoot "部署\$f"
    $ok = Test-Path $p
    Check "部署/$f" $ok $(if ($ok) { "$([math]::Round((Get-Item $p).Length / 1KB, 1)) KB" } else { '缺失' })
}
Add-Line('')

# ------------------------------------------------------------- 资源纪律 --
Add-Line('## 7. 资源纪律核验（共享约定 §5.5 + 工单17 附加）')
Add-Line('')
Add-Line('| 项 | 状态 | 实测 |')
Add-Line('| --- | --- | --- |')
$procs = @(Get-Process python -ErrorAction SilentlyContinue)
$heavy = @($procs | Where-Object { ($_.PrivateMemorySize64 / 1MB) -gt 200 })
Check 'python 进程回归会话基线（无 >200MB 残留）' ($heavy.Count -eq 0) `
    "当前 $($procs.Count) 个 python；超 200MB 的 $($heavy.Count) 个"
$port8200 = $false
try { if ((Invoke-WebRequest http://127.0.0.1:8200/healthz -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200) { $port8200 = $true } } catch { }
Check '8200 端口已释放（若仍在跑压测则忽略此项）' (-not $port8200) `
    $(if ($port8200) { '仍在响应（服务在跑）' } else { '无响应' })
$caches = @(Get-ChildItem $repoRoot -Recurse -Directory -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in @('__pycache__', '.pytest_cache') })
Check '__pycache__ / .pytest_cache 清零' ($caches.Count -eq 0) "$($caches.Count) 个残留"
$hasData = Test-Path (Join-Path $repoRoot '研发\data')
Check '未复制索引 data/（只读引用工单7）' (-not $hasData) `
    $(if ($hasData) { '存在研发/data（应删除）' } else { '不存在，索引走绝对路径引用' })
Add-Line('')

# ---------------------------------------------------------------- 汇总 --
Add-Line('## 8. 汇总')
Add-Line('')
if ($issues.Count -eq 0) {
    Add-Line('**全部核验项通过。**')
} else {
    Add-Line("**未通过 $($issues.Count) 项**：")
    Add-Line('')
    foreach ($i in $issues) { Add-Line("- $i") }
}
Add-Line('')
$total = $lines.Count
Add-Line("> 本文件由脚本自动生成，共 $total 行；核验时间 $now。")

Set-Content -LiteralPath $outFile -Value ($lines -join "`n") -Encoding utf8
Write-Host "已写出：$outFile"
Write-Host "未通过项：$($issues.Count)"
if ($issues.Count) { $issues | ForEach-Object { Write-Host "  - $_" } }
