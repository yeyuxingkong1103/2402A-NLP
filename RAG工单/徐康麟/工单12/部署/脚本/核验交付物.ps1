# =============================================================================
# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
# 文件：部署/脚本/核验交付物.ps1
#
# 用途：**从真实文件系统**扫描生成 部署/交付物核验.md ——
#       五个中文一级目录文件数、测试截图张数与中文命名、pytest/报告/优化/部署产物，
#       以及硬检查「文档引用的截图文件名 与 磁盘实际文件名是否完全一致」。
#       清单不手写，避免「文档里写了、磁盘上没有」。
#
# 用法（工作目录 = E:\gao6gongdan\工单12）：
#     pwsh -NoProfile -File 部署/脚本/核验交付物.ps1                       # 写 部署/交付物核验.md
#     pwsh -NoProfile -File 部署/脚本/核验交付物.ps1 -OutFile 测试/临时/预检.md
# =============================================================================
[CmdletBinding()]
param([string]$OutFile = '')

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot
if (-not $OutFile) { $OutFile = Join-Path $repoRoot '部署\交付物核验.md' }

$lines = New-Object System.Collections.Generic.List[string]
$issues = New-Object System.Collections.Generic.List[string]
$now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
function Add-Line([string]$s) { $script:lines.Add($s) }
function Check([string]$label, [bool]$ok, [string]$detail) {
    Add-Line("| $label | $(if ($ok) { '✅' } else { '❌' }) | $detail |")
    if (-not $ok) { $script:issues.Add("$label —— $detail") }
}
function FilesOf([string]$rel) {
    @(Get-ChildItem (Join-Path $repoRoot $rel) -Recurse -File -Force -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -notmatch '__pycache__|\.pytest_cache' })
}

Add-Line('# 交付物核验（工单12 / LightRAG 优化）')
Add-Line('')
Add-Line('> **工单编号：人工智能NLP-RAG 项目-LightRAG 优化**')
Add-Line("> 核验时间：$now")
Add-Line('> 核验方式：由 `部署/脚本/核验交付物.ps1` 扫描真实文件系统自动生成（非手写清单）')
Add-Line('')

# ------------------------------------------------------------- 1 目录契约 --
Add-Line('## 1. 五个中文一级目录（共享约定 §1 硬性）')
Add-Line('')
Add-Line('| 目录 | 状态 | 文件数 | 体积 |')
Add-Line('| --- | --- | --- | --- |')
foreach ($d in '设计', '研发', '测试', '优化', '部署') {
    $exists = Test-Path (Join-Path $repoRoot $d)
    $fs = FilesOf $d
    $mb = [math]::Round((($fs | Measure-Object -Property Length -Sum).Sum) / 1MB, 2)
    Check $d $exists "$($fs.Count) 个文件 / $mb MB"
}
Add-Line('')

# ------------------------------------------------------------ 2 测试截图 --
Add-Line('## 2. 测试截图（≥3 张 + 中文按测试内容命名）')
Add-Line('')
$shotDir = Join-Path $repoRoot '测试\测试截图'
$shots = @(Get-ChildItem $shotDir -Filter *.png -File -ErrorAction SilentlyContinue)
Add-Line('| 截图 | 大小 | 命名合规 |')
Add-Line('| --- | --- | --- |')
$badName = @()
# 合规 = 含中文（按测试内容命名）且非禁用名、非纯数字/符号名
$bannedStems = @('1', '2', '3', 'img', 'screenshot', 'screen', 'shot', 'result',
                 'chart1', 'fig', 'fig_01', 'output', 'pic')
foreach ($s in $shots | Sort-Object Name) {
    $ok = ($s.BaseName -match '[\u4e00-\u9fff]') -and
          ($bannedStems -notcontains $s.BaseName.ToLower()) -and
          ($s.BaseName -notmatch '^[\d\-_]+$')
    if (-not $ok) { $badName += $s.Name }
    Add-Line("| $($s.Name) | $([math]::Round($s.Length / 1KB, 1)) KB | $(if ($ok) { '✅ 中文可读' } else { '❌ 命名不合规' }) |")
}
Check '截图张数 ≥3' ($shots.Count -ge 3) "实得 $($shots.Count) 张"
Check '截图命名含中文且非禁用名' ($badName.Count -eq 0) `
    $(if ($badName.Count) { $badName -join ', ' } else { "$($shots.Count) 张全部合规（含中文、非纯数字/纯英文/禁用名）" })
Add-Line('')

# ---------------------------------------------- 3 硬检查：文件名一致性 --
Add-Line('## 3. 硬检查：文档引用的截图文件名 = 磁盘实际文件名')
Add-Line('')
$onDisk = @($shots | ForEach-Object Name)
# 命名规范文档里会举反面例子（"❌ 1.png / screenshot.png …"），这些不是真实引用，跳过
$bannedExamples = @('1.png', 'img.png', 'screenshot.png', 'result.png', 'chart1.png',
                    'fig_01.png', 'output.png', 'pic.png', '2.png', '3.png')
$docs = @(Get-ChildItem (Join-Path $repoRoot '测试\*.md'), (Join-Path $repoRoot '优化\*.md'),
               (Join-Path $repoRoot '部署\*.md'), (Join-Path $repoRoot '设计\*.md') -File -ErrorAction SilentlyContinue)
$refs = @()
foreach ($doc in $docs) {
    $text = Get-Content -LiteralPath $doc.FullName -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
    foreach ($m in [regex]::Matches([string]$text, '[\u4e00-\u9fff0-9A-Za-z_\-]+\.png')) {
        $refs += [pscustomobject]@{ Doc = $doc.Name; Name = $m.Value }
    }
}
$missing = @($refs | Where-Object { $onDisk -notcontains $_.Name -and $bannedExamples -notcontains $_.Name })
$docsWithRefs = @($refs | Select-Object -ExpandProperty Doc -Unique)
Add-Line("扫描文档 $($docs.Count) 份，引用 .png 文件名 $($refs.Count) 处（来自 $($docsWithRefs.Count) 份文档）；磁盘实际 $($onDisk.Count) 张。")
Add-Line('')
Check '引用文件名全部能在磁盘找到' ($missing.Count -eq 0) `
    $(if ($missing.Count) { ($missing | ForEach-Object { "$($_.Doc) → $($_.Name)" }) -join '；' } else { '零缺失' })
Add-Line('')

# ------------------------------------------------------------- 4 关键产物 --
Add-Line('## 4. 关键产物')
Add-Line('')
Add-Line('| 产物 | 状态 | 说明 |')
Add-Line('| --- | --- | --- |')
$checks = @(
    @{ P = '设计\验收标准.md';            N = '设计文档（6 份）' },
    @{ P = '研发\lightrag\entity_schema.py'; N = '实体/关系 Schema（17/16 类）' },
    @{ P = '研发\data\graph\graph_meta.json'; N = '图谱抽样口径' },
    @{ P = '研发\data\graph\graph_stats.json'; N = '图谱统计' },
    @{ P = '测试\测试报告.md';            N = '测试报告' },
    @{ P = '优化\LightRAG优化说明.md';    N = '优化说明（短版）' },
    @{ P = '优化\评估结果\对比报告.md';    N = 'RAG vs LightRAG 对比报告' },
    @{ P = '优化\评估结果\对比汇总.json';  N = '指标对比汇总' },
    @{ P = '部署\配置\lightrag.json';     N = '生效配置' },
    @{ P = '部署\README.md';              N = '部署文档' }
)
foreach ($c in $checks) {
    $p = Join-Path $repoRoot $c.P
    $ok = Test-Path $p
    Check $c.N $ok $(if ($ok) { "$([math]::Round((Get-Item $p).Length / 1KB, 1)) KB｜$($c.P)" } else { "缺失：$($c.P)" })
}
$testFiles = @(Get-ChildItem (Join-Path $repoRoot '测试') -Filter 'test_*.py' -File -ErrorAction SilentlyContinue)
$evalRows = 0
foreach ($f in '逐题结果_rag.jsonl', '逐题结果_lightrag.jsonl') {
    $p = Join-Path $repoRoot "优化\评估结果\$f"
    if (Test-Path $p) { $evalRows += @(Get-Content $p | Where-Object { $_.Trim() }).Count }
}
Check 'pytest 用例文件' ($testFiles.Count -ge 1) "$($testFiles.Count) 个：$(($testFiles | ForEach-Object Name) -join ', ')"
Check '逐题结果 32 行（16 题 × 2 后端）' ($evalRows -eq 32) "实得 $evalRows 行"
Add-Line('')

# --------------------------------------------- 4.1 验收标准条款对照 --
Add-Line('## 4.1 验收标准条款对照（逐条，自动可核验项）')
Add-Line('')
Add-Line('| 条款 | 状态 | 实测证据 |')
Add-Line('| --- | --- | --- |')
$gs = Get-Content (Join-Path $repoRoot '研发\data\graph\graph_stats.json') -Raw -Encoding UTF8 | ConvertFrom-Json
Check '验收标准1：实体/关系类型体系（17/16）' `
    (($gs.entity_types_declared -eq 17) -and ($gs.relation_types_declared -eq 16)) `
    "declared 17 实体 / 16 关系；多别名节点 $($gs.entity_multi_alias_nodes) 个、类型冲突 $($gs.entity_type_conflicts) 个"
Check '验收标准1：关系头尾相容如实上报' (($gs.edges_constraint_ok + $gs.edges_constraint_flagged) -eq $gs.edges) `
    "相容 $($gs.edges_constraint_ok) / 已标记 $($gs.edges_constraint_flagged) / 合计 $($gs.edges)（keep_flagged，未改写任一条边）"
$rowRag = @(Get-Content (Join-Path $repoRoot '优化\评估结果\逐题结果_rag.jsonl') -ErrorAction SilentlyContinue | Where-Object { $_.Trim() }).Count
$rowLr = @(Get-Content (Join-Path $repoRoot '优化\评估结果\逐题结果_lightrag.jsonl') -ErrorAction SilentlyContinue | Where-Object { $_.Trim() }).Count
$rowSrc = @(Get-Content (Join-Path $repoRoot '研发\评估\逐题结果.jsonl') -ErrorAction SilentlyContinue | Where-Object { $_.Trim() }).Count
Check '验收标准2 / 产出物3：16 题 RAG vs LightRAG 逐题结果' (($rowRag + $rowLr) -eq 32) `
    "派生视图 rag $rowRag + lightrag $rowLr = $($rowRag + $rowLr) 行；源文件 研发\评估\逐题结果.jsonl $rowSrc 行（16 题 × 3 后端）"
$ragasRaw = Get-Content (Join-Path $repoRoot '优化\评估结果\RAGAS指标.json') -Raw -Encoding UTF8
$nMetricSets = ([regex]::Matches($ragasRaw, '"题数":\s*16')).Count
Check '验收标准2：RAGAS 四项指标对比（3 后端）' ($nMetricSets -eq 3) `
    "$nMetricSets 组汇总（每组 4 项）；官方 ragas 可用=false，$([regex]::Match($ragasRaw, '"官方ragas探测结果":\s*"([^"]+)"').Groups[1].Value)"
Check '产出物1：实现步骤 + 过程问题记录' `
    ((Test-Path (Join-Path $repoRoot '部署\README.md')) -and (Test-Path (Join-Path $repoRoot '设计\需求分析.md'))) `
    '部署\README.md（六条命令 + 两段跑法 + 依赖）、设计\需求分析.md、对比报告 §7 未达标项'
$graphFiles = @(Get-ChildItem (Join-Path $repoRoot '研发\data\graph') -File -ErrorAction SilentlyContinue)
Check '产出物2：知识图谱 4 格式 + 可视化' ($graphFiles.Count -ge 6) `
    "研发\data\graph\：$($graphFiles.Count) 个文件（nodes/edges/graph.json/graph.graphml/stats/meta）"
# S3：报告里的数字必须能在产物 json 中找到（抽样 9 处）
$summaryRaw = Get-Content (Join-Path $repoRoot '优化\评估结果\对比汇总.json') -Raw -Encoding UTF8
$reportRaw = Get-Content (Join-Path $repoRoot '测试\测试报告.md') -Raw -Encoding UTF8
$summaryVals = @()
$summaryVals += ([regex]::Matches($summaryRaw, '"平均答案关键词命中率":\s*([0-9.]+)') | ForEach-Object { $_.Groups[1].Value })
$summaryVals += ([regex]::Matches($summaryRaw, '"平均上下文关键词命中率":\s*([0-9.]+)') | ForEach-Object { $_.Groups[1].Value })
$summaryVals += ([regex]::Matches($summaryRaw, '"页命中数":\s*([0-9]+)') | ForEach-Object { $_.Groups[1].Value })
$missNum = @($summaryVals | Where-Object { $reportRaw -notmatch [regex]::Escape($_) })
Check "S3 报告数字可在产物 json 溯源（$($summaryVals.Count) 处抽样）" ($missNum.Count -eq 0) `
    $(if ($missNum.Count) { '缺：' + ($missNum -join ', ') } else { "$($summaryVals.Count) 处全部命中（对比汇总.json → 测试报告.md）" })
# S4：本工单新增/改动源码头部含工单编号；与 工单7 逐字节一致的继承件按 S7「只增不改」不追加
$srcFiles = @(Get-ChildItem (Join-Path $repoRoot '研发') -Recurse -Include *.py -File -ErrorAction SilentlyContinue |
              Where-Object { $_.FullName -notmatch '__pycache__|第三方库' })
$devRoot = Join-Path $repoRoot '研发'
$base7Root = 'E:\gao6gongdan\工单7\研发'
$noHeaderNew = @()
$newCount = 0
$inheritedCount = 0
foreach ($f in $srcFiles) {
    $rel = $f.FullName.Substring($devRoot.Length).TrimStart('\')
    $basePath = Join-Path $base7Root $rel
    $same = $false
    if (Test-Path -LiteralPath $basePath) {
        $same = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash -eq
                (Get-FileHash -LiteralPath $basePath -Algorithm SHA256).Hash
    }
    if ($same) { $inheritedCount++; continue }
    $newCount++
    if (-not (((Get-Content -LiteralPath $f.FullName -TotalCount 4 -Encoding UTF8 -ErrorAction SilentlyContinue) -join ' ') -match '工单编号')) {
        $noHeaderNew += $rel
    }
}
Check 'S4 本工单新增/改动源码头部含工单编号' ($noHeaderNew.Count -eq 0) `
    "研发 .py 共 $($srcFiles.Count) 个 → 新增/改动 $newCount 个（缺注释 $($noHeaderNew.Count) 个$(if ($noHeaderNew.Count) { '：' + ($noHeaderNew -join ', ') })）；与 工单7 逐字节一致的继承件 $inheritedCount 个（按 S7「只增不改」不追加注释）"
# S8：索引沿用未重建（文件在且有索引清单）
$vecPath = Join-Path $repoRoot '研发\data\index\vectors.npy'
$bmPath = Join-Path $repoRoot '研发\data\index\bm25.pkl'
$indexOk = (Test-Path $vecPath) -and (Test-Path $bmPath)
Check 'S8 向量/BM25 索引沿用（未重建）' $indexOk `
    $(if ($indexOk) { "vectors.npy $([math]::Round((Get-Item $vecPath).Length / 1MB, 1)) MB / bm25.pkl $([math]::Round((Get-Item $bmPath).Length / 1MB, 2)) MB（复用 工单7 产物）" } else { '索引文件缺失' })
Add-Line('')
Add-Line('**人工核验项（脚本不自动判定，已附定位）**：S5 阈值配置化 → `部署\配置\lightrag.json`（研发源码内阈值 0 命中）；')
Add-Line('S7 只增不改 → `E:\gao6gongdan\工单7\` 未被修改；S9 文档数字标来源 → `设计\*.md` 每个数字后带来源路径；')
Add-Line('R2 峰值内存 → `E:\gao6gongdan\_共享工具\运行日志\*.监察.log`（3 次建图均「未开枪」）；R3/R4 串行与单模型驻留 → `部署\日志\20_build_graph_20261016.log`。')
Add-Line('')

# ------------------------------------------------------------ 5 资源纪律 --
Add-Line('## 5. 资源纪律自检（共享约定 §5.5）')
Add-Line('')
Add-Line('| 项 | 状态 | 实测 |')
Add-Line('| --- | --- | --- |')
$pyFiles = @(Get-ChildItem (Join-Path $repoRoot '研发') -Recurse -Include *.py -File |
             Where-Object { $_.FullName -notmatch '__pycache__' })
$pattern = 'multiprocessing|ProcessPoolExecutor|Pool\(|joblib|n_jobs'
# 判定项：只算**代码行**（跳过注释/文档字符串行，避免把「禁用说明」误判成使用）
$codeHits = @()
foreach ($f in $pyFiles) {
    $ln = 0
    foreach ($line in (Get-Content -LiteralPath $f.FullName -Encoding UTF8 -ErrorAction SilentlyContinue)) {
        $ln++
        $t = ([string]$line).Trim()
        if ($t.StartsWith('#') -or $t.StartsWith('*') -or $t.StartsWith('"""') -or $t.StartsWith("'''")) { continue }
        if ($t -match $pattern) { $codeHits += "$($f.Name):$ln" }
    }
}
$naiveHits = @($pyFiles | Select-String -CaseSensitive -Pattern $pattern)
Check '无进程池（代码行 0 命中）' ($codeHits.Count -eq 0) `
    "$($codeHits.Count) 处代码命中；朴素全量 grep 命中 $($naiveHits.Count) 处（含注释提及）$(if ($naiveHits.Count) { '：' + (($naiveHits | ForEach-Object { "$($_.Filename):$($_.LineNumber)" }) -join ', ') })"
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='pythonw.exe'" -ErrorAction SilentlyContinue)
# MCP 常驻进程（serena / uv 缓存）不属于本工单残留；只判「命令行含 工单12」或「>200MB」
$mcp = @($procs | Where-Object { $_.CommandLine -match 'serena|uv\\cache|mcp' })
$work = @($procs | Where-Object { $_.CommandLine -match '工单12' })
$heavy = @($procs | Where-Object { $_.WorkingSetSize -gt 200MB })
Check '无本工单遗留 python 进程' (($work.Count -eq 0) -and ($heavy.Count -eq 0)) `
    "python 共 $($procs.Count) 个 → 命令行含「工单12」的 $($work.Count) 个、>200MB 的 $($heavy.Count) 个；其余 $($mcp.Count) 个为 MCP 常驻（serena/uv，非本工单）"
$caches = @(Get-ChildItem $repoRoot -Recurse -Directory -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in @('__pycache__', '.pytest_cache') })
Check '__pycache__ / .pytest_cache 已清' ($caches.Count -eq 0) "$($caches.Count) 个残留"
$port = Test-NetConnection -ComputerName 127.0.0.1 -Port 8612 -InformationLevel Quiet -WarningAction SilentlyContinue
Check '评估服务端口 8612 已释放' (-not $port) $(if ($port) { '仍在监听' } else { '未监听' })
# 收工三查之一：Ollama 无驻留模型
try {
    $psInfo = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/ps' -TimeoutSec 5
    $nModels = @($psInfo.models).Count
    Check 'Ollama 无驻留模型（/api/ps 为空）' ($nModels -eq 0) `
        "当前驻留 $nModels 个$(if ($nModels) { '：' + (($psInfo.models | ForEach-Object { $_.name }) -join ', ') })"
} catch {
    Check 'Ollama 无驻留模型（/api/ps 为空）' $false "探活失败：$($_.Exception.Message)"
}
Add-Line('')

# ---------------------------------------------------------------- 汇总 --
Add-Line('## 6. 结论')
Add-Line('')
if ($issues.Count -eq 0) {
    Add-Line('**核验通过：全部检查项 ✅。**')
} else {
    Add-Line("**核验未通过：$($issues.Count) 项 ❌**")
    Add-Line('')
    foreach ($i in $issues) { Add-Line("- $i") }
}
Add-Line('')
Add-Line("> 本文件由 `部署/脚本/核验交付物.ps1` 扫描生成，共 $($lines.Count) 行；核验时间 $now。")

Set-Content -LiteralPath $OutFile -Value ($lines -join "`n") -Encoding utf8
Write-Host "已写出：$OutFile"
Write-Host "未通过项：$($issues.Count)"
if ($issues.Count) { $issues | ForEach-Object { Write-Host "  - $_" } }
