# =============================================================================
# 工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件：部署/脚本/evaluate.ps1 —— 一键评测（部署后自证：解析/索引/召回/答案/测试）
#
# 实测口径（全部来自真实运行，禁止口头结论）：
#   ① 基座校验   部署/脚本/verify_config_env.py（解释器/依赖/PDF/Ollama 四段）
#   ② 重建索引   研发/scripts/parse_corpus.py + 研发/scripts/build_index.py（仅 -Rebuild）
#   ③ 召回评测   测试/离线/eval_retrieval.py（14 题命中 top-5：全库 + 按文件过滤两套口径）
#   ④ 答案评测   测试/离线/eval_answers.py（14 题答案 + 引用可回溯 + 首字 ms）
#   ⑤ 离线测试   pytest 测试/离线（锚点/索引/引擎契约/界面共用回归）
#   ⑥ 在线+用户  测试/在线 + 测试/用户/simulate_user.py（仅 -WithOnline；需要 Ollama 可用）
#
# **产物落位纪律**：本脚本把新的评测产物写到自己的目录（默认 部署/日志/eval/），
#   **不覆盖** 优化/评估结果/ 下的权威留痕（那里是 T5/T6/T7 的既有证据，只读引用）。
#
# 每个步骤的真实退出码、耗时、stdout 尾部与产物路径都写入
#   部署/日志/evaluate_<stamp>.json（机器可读）与 部署/日志/deploy.log（结构化 JSON Lines）。
#
# 用法（工作目录 = E:\gao6gongdan\工单3）：
#   pwsh -NoProfile -File 部署/脚本/evaluate.ps1                        # 全套（不重建索引）
#   pwsh -NoProfile -File 部署/脚本/evaluate.ps1 -Level smoke           # 冒烟：只跑 ①③④
#   pwsh -NoProfile -File 部署/脚本/evaluate.ps1 -Rebuild               # 先重建索引再评测
#   pwsh -NoProfile -File 部署/脚本/evaluate.ps1 -WithOnline            # 追加在线/用户级测试
#   pwsh -NoProfile -File 部署/脚本/evaluate.ps1 -EnvFile 部署/配置/config.example.env
#
# 退出码：0=全部步骤通过；1=存在失败步骤（禁止「部分通过也算过」）；127=解释器缺失。
# =============================================================================
[CmdletBinding()]
param(
    [ValidateSet('smoke', 'full')]
    [string] $Level = 'full',
    [switch] $Rebuild,
    [switch] $WithOnline,
    [string] $EnvFile = '',
    [string] $Python = '',
    [string] $EvalOutDir = '部署/日志/eval',
    [string[]] $KnownRedDeselect = @(),
    [int]    $TailLines = 12
)

$ErrorActionPreference = 'Stop'
$WorkOrder = '人工智能NLP-RAG-PDF文档的表格解析及检索优化'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $RepoRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'

$LogDir = Join-Path $RepoRoot '部署\日志'
$DeployLog = Join-Path $LogDir 'deploy.log'
$Entry = Join-Path $RepoRoot 'run_py.ps1'
$OutDirPath = if ([System.IO.Path]::IsPathRooted($EvalOutDir)) { $EvalOutDir } else { Join-Path $RepoRoot $EvalOutDir }
if (-not (Test-Path -LiteralPath $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $OutDirPath)) { New-Item -ItemType Directory -Path $OutDirPath -Force | Out-Null }
$OutDirRel = $EvalOutDir.Replace('\', '/').TrimEnd('/')

function Write-DeployLog {
    param(
        [Parameter(Mandatory)][string] $Event,
        [string] $Level = 'INFO',
        [string] $Func = '',
        $Inputs = $null,
        $Outputs = $null,
        $ElapsedMs = $null,
        $Error = $null
    )
    $record = [ordered]@{
        ts = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss.fffzzz'); level = $Level; event = $Event; func = $Func
        work_order = $WorkOrder; module = '部署/脚本/evaluate.ps1'; pid = $PID
        inputs = $Inputs; outputs = $Outputs; elapsed_ms = $ElapsedMs; error = $Error
    }
    Add-Content -LiteralPath $DeployLog -Value ($record | ConvertTo-Json -Compress -Depth 6) -Encoding utf8
    if ($Level -in @('ERROR', 'CRITICAL')) { Write-Host "[$Level] $($record.event) :: $($record.func) :: $($record.error.message)" }
}

function Import-EnvFile {
    param([string] $Path)
    if (-not $Path) { return @{} }
    $resolved = $Path
    if (-not [System.IO.Path]::IsPathRooted($resolved)) { $resolved = Join-Path $RepoRoot $Path }
    if (-not (Test-Path -LiteralPath $resolved)) { throw "env 文件不存在：$resolved" }
    $applied = @{}
    foreach ($raw in [System.IO.File]::ReadAllLines($resolved, [System.Text.Encoding]::UTF8)) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith('#')) { continue }
        $idx = $line.IndexOf('=')
        if ($idx -le 0) { continue }
        $key = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        if ($value -eq '') { continue }
        Set-Item -Path "Env:$key" -Value $value
        $applied[$key] = $value
    }
    return $applied
}

function Resolve-Python {
    param([string] $Override)
    if ($Override) { return $Override }
    if ($env:RAG_SCHEDULER_PYTHON) { return $env:RAG_SCHEDULER_PYTHON.Trim().Trim('"').Trim("'") }
    return 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
}

# 单步执行：跑 run_py.ps1（统一解释器入口），完整 stdout 落盘，返回结构化结果（不污染返回值）
function Invoke-EvalStep {
    param(
        [Parameter(Mandatory)][string] $Name,
        [Parameter(Mandatory)][string[]] $Arguments,
        [string] $Description = ''
    )
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $stepLog = Join-Path $LogDir "evaluate_${Name}_$stamp.log"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-DeployLog -Event 'func.enter' -Func "Invoke-EvalStep/$Name" -Inputs @{ arguments = ($Arguments -join ' '); description = $Description }
    Write-Host ''
    Write-Host ("===== [{0}] {1} =====" -f $Name, $Description)
    Write-Host ("     命令：pwsh -NoProfile -File run_py.ps1 {0}" -f ($Arguments -join ' '))
    $output = @(& pwsh -NoProfile -File $Entry @Arguments 2>&1)
    $code = $LASTEXITCODE
    $output | Out-File -FilePath $stepLog -Encoding utf8
    $sw.Stop()
    # 只把尾部打到控制台（完整输出见 stepLog），避免刷屏
    $tail = @($output | Select-Object -Last $TailLines)
    $tail | ForEach-Object { Write-Host "     | $_" }
    if ($code -eq 0) {
        Write-DeployLog -Event 'func.exit' -Func "Invoke-EvalStep/$Name" -Inputs @{ arguments = ($Arguments -join ' ') } `
            -Outputs @{ exit_code = 0; log = $stepLog; tail = ($tail -join "`n") } -ElapsedMs $sw.ElapsedMilliseconds
    }
    else {
        Write-DeployLog -Event 'func.error' -Level 'ERROR' -Func "Invoke-EvalStep/$Name" -Inputs @{ arguments = ($Arguments -join ' ') } `
            -Outputs @{ log = $stepLog; tail = ($tail -join "`n") } -ElapsedMs $sw.ElapsedMilliseconds `
            -Error @{ type = 'StepFailed'; message = "步骤 $Name 退出码 $code"; stack = ($tail -join "`n") }
    }
    return [ordered]@{
        name = $Name; description = $Description; arguments = ($Arguments -join ' ')
        exit_code = $code; elapsed_ms = $sw.ElapsedMilliseconds; log = $stepLog
        tail = ($tail -join "`n"); passed = ($code -eq 0)
    }
}

# 读取 JSON 产物的 summary（读不到就显式 WARN 并返回 $null，不静默）
function Read-JsonSummary {
    param([string] $Path)
    $full = if ([System.IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $RepoRoot $Path }
    if (-not (Test-Path -LiteralPath $full)) {
        Write-DeployLog -Event 'evaluate.artifact_missing' -Level 'WARNING' -Func 'Read-JsonSummary' -Inputs @{ path = $full }
        return $null
    }
    try { return (Get-Content -LiteralPath $full -Raw -Encoding utf8 | ConvertFrom-Json) }
    catch {
        Write-DeployLog -Event 'evaluate.artifact_unreadable' -Level 'WARNING' -Func 'Read-JsonSummary' -Inputs @{ path = $full } `
            -Error @{ type = $_.Exception.GetType().Name; message = $_.Exception.Message; stack = '' }
        return $null
    }
}

$exitCode = 0
$stampAll = Get-Date -Format 'yyyyMMdd_HHmmss'
try {
    $py = Resolve-Python -Override $Python
    if (-not (Test-Path -LiteralPath $py)) {
        Write-DeployLog -Event 'func.error' -Level 'ERROR' -Func 'Resolve-Python' -Inputs @{ python = $py } `
            -Error @{ type = 'FileNotFoundError'; message = '找不到 Python 解释器'; stack = '' }
        Write-Host "❌ 找不到 Python 解释器：$py" -ForegroundColor Red
        exit 127
    }

    if ($EnvFile) {
        $applied = Import-EnvFile -Path $EnvFile
        Write-DeployLog -Event 'func.exit' -Func 'Import-EnvFile' -Inputs @{ env_file = $EnvFile } `
            -Outputs @{ applied_keys = @($applied.Keys).Count; keys = (@($applied.Keys) -join ',') } -ElapsedMs 0
    }

    Write-Host '================================================================'
    Write-Host "工单：$WorkOrder"
    Write-Host ("一键评测：级别 {0}；重建索引={1}；含在线级={2}；产物目录={3}" -f $Level, [bool]$Rebuild, [bool]$WithOnline, $OutDirRel)
    Write-Host 'RAGAS 未运行（依赖不可用，本机断网）—— 本脚本只产出确定性指标，绝不伪造 RAGAS 数值'
    Write-Host "解释器：$py"
    Write-Host '================================================================'

    $steps = @()

    # ① 基座校验：解释器 / 依赖（真实 import）/ PDF / Ollama
    $steps += Invoke-EvalStep -Name 'verify_env' -Description '基座校验（解释器/依赖/PDF/Ollama）' `
        -Arguments @('部署/脚本/verify_config_env.py')

    # ② 可选重建索引：解析 → 建索引（幂等，重跑安全）
    if ($Rebuild) {
        $steps += Invoke-EvalStep -Name 'parse_corpus' -Description '全量解析（逐页扫描 + 表格归一化 + 分块）' `
            -Arguments @('研发/scripts/parse_corpus.py')
        $steps += Invoke-EvalStep -Name 'build_index' -Description '建索引（嵌入 + 自实现 BM25）' `
            -Arguments @('研发/scripts/build_index.py')
    }

    # ③ 召回评测：14 题 top-5 命中（全库 + 按文件过滤）
    $steps += Invoke-EvalStep -Name 'retrieval_full' -Description '14 题召回评测（全库）' `
        -Arguments @('测试/离线/eval_retrieval.py',
                     '--json', "$OutDirRel/retrieval_eval_deploy_full.json",
                     '--md', "$OutDirRel/retrieval_eval_deploy_full.md")
    $steps += Invoke-EvalStep -Name 'retrieval_filtered' -Description '14 题召回评测（按文件过滤，越界块必须为 0）' `
        -Arguments @('测试/离线/eval_retrieval.py', '--file-filter',
                     '--json', "$OutDirRel/retrieval_eval_deploy_filtered.json",
                     '--md', "$OutDirRel/retrieval_eval_deploy_filtered.md")

    # ④ 答案评测：14 题答案 + 引用可回溯 + 首字 ms
    $steps += Invoke-EvalStep -Name 'answer_eval' -Description '14 题端到端答案评测（引用/首字）' `
        -Arguments @('测试/离线/eval_answers.py',
                     '--json', "$OutDirRel/answer_eval_deploy.json",
                     '--md', "$OutDirRel/answer_eval_deploy.md")

    if ($Level -eq 'full') {
        # ⑤ 离线测试套件（锚点/索引/引擎契约/界面共用）
        #    -KnownRedDeselect：只用于**已在 测试/测试报告.md §9 登记的「预期为红」用例**，
        #    显式 --deselect 并在日志/报告里留痕（WARNING），绝不静默改绿色
        $pytestArgs = @('-m', 'pytest', '测试/离线', '-q')
        foreach ($knownRed in @($KnownRedDeselect)) {
            if ($knownRed) { $pytestArgs += @('--deselect', $knownRed) }
        }
        if (@($KnownRedDeselect).Count -gt 0) {
            Write-DeployLog -Event 'evaluate.known_red_deselected' -Level 'WARNING' -Func 'evaluate' `
                -Inputs @{ deselected = @($KnownRedDeselect) } `
                -Outputs @{ reason = '登记在 测试/测试报告.md §9 的「预期为红」用例（判分尺子已知特性，captain 裁定不修判分器）' }
            Write-Host "⚠️ 已显式排除 $($KnownRedDeselect.Count) 个「预期为红」用例（登记见 测试/测试报告.md §9）：$($KnownRedDeselect -join '; ')" -ForegroundColor Yellow
        }
        $steps += Invoke-EvalStep -Name 'pytest_offline' -Description '离线测试（pytest 测试/离线）' `
            -Arguments $pytestArgs
    }

    if ($WithOnline) {
        # ⑥ 在线级 + 用户级（需要 Ollama 可用；不可用会失败而不是静默跳过）
        $steps += Invoke-EvalStep -Name 'pytest_online' -Description '在线级测试（pytest 测试/在线）' `
            -Arguments @('-m', 'pytest', '测试/在线', '-q')
        $steps += Invoke-EvalStep -Name 'simulate_user' -Description '用户级端到端模拟（测试/用户/simulate_user.py）' `
            -Arguments @('测试/用户/simulate_user.py')
    }

    # ---------------- 汇总（每个数字都来自落盘产物） ----------------
    $answerFresh = Read-JsonSummary -Path "$OutDirRel/answer_eval_deploy.json"
    $retrievalFull = Read-JsonSummary -Path "$OutDirRel/retrieval_eval_deploy_full.json"
    $retrievalFiltered = Read-JsonSummary -Path "$OutDirRel/retrieval_eval_deploy_filtered.json"
    $answerAuthoritative = Read-JsonSummary -Path '优化/评估结果/answer_eval_t6.json'
    $failed = @($steps | Where-Object { -not $_.passed })
    if ($failed.Count -gt 0) { $exitCode = 1 }

    Write-Host ''
    Write-Host '================================ 汇总 ================================'
    foreach ($s in $steps) {
        $mark = if ($s.passed) { '✅' } else { '❌' }
        Write-Host ("  {0} {1,-18} 退出码 {2,-3} 耗时 {3,9} ms  {4}" -f $mark, $s.name, $s.exit_code, $s.elapsed_ms, $s.log)
    }
    if ($answerFresh) {
        $sum = $answerFresh.summary
        Write-Host ''
        Write-Host ("  本次答案评测：作答 {0}/{1}；语义正确 {2}/{1}；引用可回溯 {3}/{4}；首字 max {5} ms（预算 3000）；后端 {6}/{7}" -f `
            $sum.answered, $sum.total, $sum.correct_count, $sum.citation_supported, $sum.citation_total,
            $sum.first_token_warm.max, $sum.backend.name, $sum.backend.model)
        Write-Host ("  如实披露：错题 {0}；误拒答 {1}；文件归属错误 {2}" -f `
            $(if (@($sum.incorrect_ids).Count -gt 0) { @($sum.incorrect_ids) -join ',' } else { '无' }), `
            $(if (@($sum.unknown_ids).Count -gt 0) { @($sum.unknown_ids) -join ',' } else { '无' }), `
            $(if (@($sum.wrong_file_questions).Count -gt 0) { @($sum.wrong_file_questions) -join ',' } else { '无' }))
    }
    if ($retrievalFull) {
        $rs = $retrievalFull.summary
        Write-Host ("  本次召回评测（全库）：hit@{0} {1}/{2} = {3}；严格单证据 {4}/{2}" -f `
            $rs.top_k, $rs.hit, $rs.total, $rs.hit_rate, $rs.hit_strict_single_evidence)
    }
    if ($retrievalFiltered) {
        $rf = $retrievalFiltered.summary
        # filter_violations 是「按题记录」对象（空对象 = 无越界块），不能按数组计数
        $violationCount = if ($null -eq $rf.filter_violations) { 0 }
                          else { @($rf.filter_violations.PSObject.Properties).Count }
        Write-Host ("  本次召回评测（按文件过滤）：hit@{0} {1}/{2}；越界块记录（filter_violations）{3} 条；未命中 {4}" -f `
            $rf.top_k, $rf.hit, $rf.total, $violationCount, (@($rf.miss_ids) -join ','))
    }
    if ($answerAuthoritative) {
        $auth = $answerAuthoritative.summary
        Write-Host ("  参考（优化/评估结果/answer_eval_t6.json，T7/t12 权威留痕）：作答 {0}/{1}；语义正确 {2}/{1}；首字 max {3} ms" -f `
            $auth.answered, $auth.total, $auth.correct_count, $auth.first_token_warm.max)
    }
    Write-Host '  RAGAS 未运行（依赖不可用，本机断网）'

    $report = [ordered]@{
        work_order = $WorkOrder
        generated_at = (Get-Date).ToString('o')
        level = $Level
        rebuild = [bool]$Rebuild
        with_online = [bool]$WithOnline
        python = $py
        ragas = '未运行（依赖不可用，本机断网）'
        env_file = $EnvFile
        eval_out_dir = $OutDirRel
        known_red_deselected = @($KnownRedDeselect)
        failed_steps = @($failed | ForEach-Object { $_.name })
        exit_code = $exitCode
        steps = @($steps)
        artifacts = [ordered]@{
            answer_eval_fresh = "$OutDirRel/answer_eval_deploy.json"
            retrieval_eval_full_fresh = "$OutDirRel/retrieval_eval_deploy_full.json"
            retrieval_eval_filtered_fresh = "$OutDirRel/retrieval_eval_deploy_filtered.json"
            answer_eval_authoritative = '优化/评估结果/answer_eval_t6.json'
            env_report = '部署/配置/verify_env_report.json'
        }
        answer_metrics = $(if ($answerFresh) { $answerFresh.summary } else { $null })
        retrieval_metrics_full = $(if ($retrievalFull) { $retrievalFull.summary } else { $null })
        retrieval_metrics_filtered = $(if ($retrievalFiltered) { $retrievalFiltered.summary } else { $null })
    }
    $reportPath = Join-Path $LogDir "evaluate_$stampAll.json"
    [System.IO.File]::WriteAllText($reportPath, ($report | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding($false)))
    Write-DeployLog -Event 'evaluate.done' -Level $(if ($exitCode -eq 0) { 'INFO' } else { 'ERROR' }) -Func 'evaluate' `
        -Inputs @{ level = $Level; rebuild = [bool]$Rebuild; with_online = [bool]$WithOnline } `
        -Outputs @{ report = $reportPath; failed_steps = @($failed | ForEach-Object { $_.name }); exit_code = $exitCode } -ElapsedMs 0
    Write-Host ''
    Write-Host ("  评测报告：{0}" -f $reportPath)
    Write-Host ("  部署日志：{0}" -f $DeployLog)
    exit $exitCode
}
catch {
    Write-DeployLog -Event 'evaluate.fatal' -Level 'CRITICAL' -Func 'evaluate' `
        -Error @{ type = $_.Exception.GetType().Name; message = $_.Exception.Message; stack = $_.ScriptStackTrace }
    Write-Host "❌ 评测流程异常：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace
    exit 1
}
