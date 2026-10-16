# =============================================================================
# 工单：人工智能NLP-RAG-图像内容解析及检索优化
# 文件：部署/脚本/rotate_logs.ps1 —— 日志归档/轮转（**归档，不是删除**）
#
# 背景（实测）：结构化日志是交付要求（可审计性），但追加式写入会让
#   部署/日志/app.log 长到 40 MB（75,670 行 / 12.5 小时），影响打包与检视。
#   研发侧 logging_conf.py 是**纯追加**实现（O_APPEND + 每行单次 write + 跨进程锁，
#   为多进程安全而不用缓冲），**没有内置轮转** —— 故本脚本做**离线轮转**（部署环节负责）。
#
# 策略（按大小触发 + 按时间命名归档 + 保留最近可审计样例）：
#   1. 触发：文件 > -ThresholdMB（默认 8 MB）或显式 -Force；
#   2. 归档：整份**流式复制并 gzip** 到 部署/日志/归档/<基名>.<yyyyMMdd-HHmmss>.<扩展名>.gz，
#      复制过程中计数；随后**回读 .gz 校验**（字节数 + 行数必须与源一致）才算归档成功；
#   3. 保留样例：把**最近 -KeepLines 行**（默认 3000）写回活动文件（逐行校验是合法 JSONL），
#      于是活动文件既小、又保留最近一段真实可审计链路；
#   4. 归档清理：每类日志最多保留 -MaxArchives 份（默认 5，0 = 不删），删除的**文件名逐一打印**；
#   5. 留痕：写 部署/日志/轮转记录.json（处理前后体积、字节/行数、归档路径、校验结果）。
#
# 两种落地方式（脚本自动选择，并如实报告用了哪种）：
#   A) 重命名法（首选，无竞态）：把活动文件 Move 走 → 放回「保留样例」→ 删除被 Move 的整份副本。
#   B) 就地截断法（copytruncate 兜底：有进程仍持有追加句柄、Move 失败时）：
#      以 FileShare.ReadWrite 打开活动文件，把保留样例字节写回偏移 0 并 SetLength —— 无需停服务。
#      ⚠️ 该方式存在「复制与截断之间写入的行可能丢失」的固有竞态（与 Linux logrotate copytruncate 相同），
#      故脚本**默认先检测 8123/8600/8125 是否有服务在监听并给出警告**；建议先停服务再轮转。
#
# 用法：
#   pwsh -NoProfile -File 部署/脚本/rotate_logs.ps1                 # 按默认阈值轮转
#   pwsh -NoProfile -File 部署/脚本/rotate_logs.ps1 -DryRun         # 只看会做什么，不改任何文件
#   pwsh -NoProfile -File 部署/脚本/rotate_logs.ps1 -ThresholdMB 4 -KeepLines 2000 -Force
#
# 退出码：0 成功（含「未达阈值、无需轮转」）/ 1 有文件处理失败（其余文件继续处理，逐一报告）
# =============================================================================
param(
    [int]$ThresholdMB = 8,
    [int]$KeepLines = 3000,
    [int]$MaxArchives = 5,
    [string]$LogDir = '',
    [string[]]$Files = @('app.log', 'error.log', 'rag_trace.jsonl'),
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)      # 部署/脚本 → 部署 → 工单4
if (-not $LogDir) { $LogDir = Join-Path $root '部署/日志' }
if (-not (Test-Path -LiteralPath $LogDir)) { Write-Host "[失败] 日志目录不存在：$LogDir"; exit 1 }
$LogDir = (Resolve-Path -LiteralPath $LogDir).Path
$archiveDir = Join-Path $LogDir '归档'
$recordPath = Join-Path $LogDir '轮转记录.json'
$thresholdBytes = [int64]$ThresholdMB * 1MB

function Format-Size([int64]$bytes) {
    if ($bytes -ge 1MB) { '{0:N2} MiB' -f ($bytes / 1MB) }
    elseif ($bytes -ge 1KB) { '{0:N1} KiB' -f ($bytes / 1KB) }
    else { '{0} B' -f $bytes }
}

# 归档：流式复制 + gzip + 回读校验（返回 @{ ok; bytes; lines; path; note }）
function Save-Archive([string]$path, [string]$base) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $ext = [IO.Path]::GetExtension($path)                 # .log / .jsonl
    $target = Join-Path $archiveDir ("{0}.{1}{2}.gz" -f $base, $stamp, $ext)
    $src = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    $srcLen = $src.Length
    $bytes = [int64]0; $lines = [int64]0
    try {
        $out = [IO.File]::Create($target)
        try {
            $gz = New-Object System.IO.Compression.GZipStream($out, [System.IO.Compression.CompressionMode]::Compress)
            try {
                $buf = New-Object byte[] (1MB)
                while (($n = $src.Read($buf, 0, $buf.Length)) -gt 0) {
                    $gz.Write($buf, 0, $n)
                    $bytes += $n
                    for ($i = 0; $i -lt $n; $i++) { if ($buf[$i] -eq 10) { $lines++ } }
                }
                $gz.Flush()
            } finally { $gz.Dispose() }
        } finally { $out.Dispose() }
    } finally { $src.Dispose() }

    # 回读校验：解压后字节数与行数必须与复制时计数一致
    $readBytes = [int64]0; $readLines = [int64]0
    $in = [IO.File]::OpenRead($target)
    try {
        $gz = New-Object System.IO.Compression.GZipStream($in, [System.IO.Compression.CompressionMode]::Decompress)
        try {
            $buf = New-Object byte[] (1MB)
            while (($n = $gz.Read($buf, 0, $buf.Length)) -gt 0) {
                $readBytes += $n
                for ($i = 0; $i -lt $n; $i++) { if ($buf[$i] -eq 10) { $readLines++ } }
            }
        } finally { $gz.Dispose() }
    } finally { $in.Dispose() }

    $ok = ($readBytes -eq $bytes) -and ($readLines -eq $lines)
    $note = if ($ok) { '回读校验一致（字节数+行数）' } else {
        "回读校验不一致：写 ${bytes}B/${lines}行 vs 读 ${readBytes}B/${readLines}行" }
    if (-not $ok) { Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue }
    return @{ ok = $ok; bytes = $bytes; lines = $lines; path = $target
              archiveBytes = (Get-Item -LiteralPath $target -ErrorAction SilentlyContinue).Length
              srcLen = $srcLen; note = $note }
}

# 保留样例：取最近 N 行，逐行校验 JSONL，写到临时文件（返回 @{ ok; lines; bytes; temp; note }）
function Save-Tail([string]$path, [int]$keep) {
    $tail = @(Get-Content -LiteralPath $path -Tail $keep -Encoding utf8 -ErrorAction Stop)
    $bad = 0
    foreach ($line in $tail) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        try { $null = $line | ConvertFrom-Json } catch { $bad++ }
    }
    if ($bad -gt 0) {
        return @{ ok = $false; lines = $tail.Count; bytes = 0; temp = $null
                  note = "保留样例中有 $bad 行不是合法 JSON —— 已放弃改写活动文件（不动原文件）" }
    }
    $temp = "$path.rotating"
    Set-Content -LiteralPath $temp -Value $tail -Encoding utf8NoBOM
    return @{ ok = $true; lines = ($tail | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count
              bytes = (Get-Item -LiteralPath $temp).Length; temp = $temp
              note = '保留样例逐行 JSON 校验通过' }
}

Write-Host ""
Write-Host ("=" * 78)
Write-Host ("工单4 日志归档/轮转：目录={0} 阈值={1} MB 保留最近={2} 行 归档上限={3} 份 DryRun={4}" -f `
    $LogDir, $ThresholdMB, $KeepLines, $MaxArchives, [bool]$DryRun)
Write-Host ("=" * 78)

# 有服务在监听时给出警告（copytruncate 竞态只在这种情况下存在）
$listening = @()
foreach ($port in 8123, 8600, 8125) {
    if (netstat -ano | Select-String (":{0}\s.*LISTENING" -f $port)) { $listening += $port }
}
if ($listening.Count -gt 0) {
    Write-Host ("[警告] 端口 {0} 有服务在监听：若走「就地截断」方式，复制与截断之间写入的行可能丢失。" -f ($listening -join '/'))
    Write-Host "        建议先停服务再轮转；本脚本会优先尝试无竞态的「重命名」方式。"
} else {
    Write-Host "[前置] 8123/8600/8125 均无监听：可用无竞态的「重命名」方式轮转。"
}

if (-not $DryRun) { if (-not (Test-Path -LiteralPath $archiveDir)) { New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null } }

$results = New-Object System.Collections.ArrayList
$failed = 0

foreach ($name in $Files) {
    $path = Join-Path $LogDir $name
    $base = [IO.Path]::GetFileNameWithoutExtension($name)
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Host ("  [SKIP] {0} —— 不存在" -f $name)
        [void]$results.Add([ordered]@{ file = $name; action = 'skip'; reason = '不存在' })
        continue
    }
    $before = (Get-Item -LiteralPath $path).Length
    if ($before -le $thresholdBytes -and -not $Force) {
        Write-Host ("  [KEEP] {0} —— {1} ≤ 阈值 {2} MB，未触发（保留原样，不改动）" -f $name, (Format-Size $before), $ThresholdMB)
        [void]$results.Add([ordered]@{ file = $name; action = 'keep'; before_bytes = $before
                                       reason = "未达阈值 $ThresholdMB MB" })
        continue
    }
    if ($DryRun) {
        Write-Host ("  [DRY ] {0} —— {1} > 阈值，将归档为 归档/{2}.<时间戳>.gz 并保留最近 {3} 行" -f `
            $name, (Format-Size $before), $base, $KeepLines)
        [void]$results.Add([ordered]@{ file = $name; action = 'dry-run'; before_bytes = $before
                                       plan = "归档 + 保留最近 $KeepLines 行" })
        continue
    }

    Write-Host ""
    Write-Host ("  [轮转] {0} —— 处理前 {1}（{2} 字节）" -f $name, (Format-Size $before), $before)
    $arch = Save-Archive $path $base
    if (-not $arch.ok) {
        Write-Host ("     [失败] 归档未通过校验：{0}（活动文件未改动）" -f $arch.note)
        [void]$results.Add([ordered]@{ file = $name; action = 'failed'; before_bytes = $before; reason = $arch.note })
        $script:failed++
        continue
    }
    Write-Host ("     归档：{0}（{1}，压缩后 {2}；源 {3} 字节 / {4} 行；{5}）" -f `
        (Split-Path -Leaf $arch.path), (Format-Size $arch.bytes), (Format-Size $arch.archiveBytes),
        $arch.bytes, $arch.lines, $arch.note)

    $tail = Save-Tail $path $KeepLines
    if (-not $tail.ok) {
        Write-Host ("     [失败] {0}（活动文件未改动；归档已保留）" -f $tail.note)
        [void]$results.Add([ordered]@{ file = $name; action = 'failed'; before_bytes = $before; reason = $tail.note
                                       archive = $arch.path })
        $script:failed++
        continue
    }

    # 方式 A：重命名（无竞态）；失败则方式 B：就地截断（copytruncate）
    $method = 'rename'
    $backup = "$path.rotated-$([DateTime]::Now.ToString('yyyyMMdd-HHmmss'))"
    $moved = $false
    try { Move-Item -LiteralPath $path -Destination $backup -ErrorAction Stop; $moved = $true }
    catch { $moved = $false }

    if ($moved) {
        try {
            Move-Item -LiteralPath $tail.temp -Destination $path -ErrorAction Stop
        } catch {
            Move-Item -LiteralPath $backup -Destination $path -ErrorAction SilentlyContinue   # 回滚
            Write-Host ("     [失败] 放回保留样例失败：{0}（已回滚，活动文件未丢）" -f $_.Exception.Message)
            [void]$results.Add([ordered]@{ file = $name; action = 'failed'; before_bytes = $before
                                           reason = $_.Exception.Message; archive = $arch.path })
            $script:failed++
            continue
        }
        Remove-Item -LiteralPath $backup -Force -ErrorAction SilentlyContinue
    } else {
        $method = 'copytruncate'
        Write-Host "     活动文件被占用（有进程持有句柄）→ 改用就地截断方式（copytruncate）"
        # 竞态防护：截断前先采样文件大小，2 秒内仍在增长说明有活跃写入 → 放弃改写活动文件（归档已保留）
        $s1 = (Get-Item -LiteralPath $path).Length
        Start-Sleep -Seconds 2
        $s2 = (Get-Item -LiteralPath $path).Length
        if ($s1 -ne $s2) {
            Remove-Item -LiteralPath $tail.temp -Force -ErrorAction SilentlyContinue
            $msg = ("就地截断前检测到活跃写入（{0} → {1} 字节/2s），为避免丢失在途日志，本次不改写活动文件；" -f $s1, $s2) +
                   "整份已归档（归档保留），活动文件保持原样。建议先停服务再轮转。"
            Write-Host ("     [放弃] {0}" -f $msg) -ForegroundColor Yellow
            [void]$results.Add([ordered]@{ file = $name; action = 'aborted-active-writer'; before_bytes = $s2
                                           reason = $msg; archive = $arch.path })
            continue
        }
        try {
            $tailBytes = [IO.File]::ReadAllBytes($tail.temp)
            $fs = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite)
            try {
                $fs.Position = 0
                $fs.Write($tailBytes, 0, $tailBytes.Length)
                $fs.SetLength($tailBytes.Length)
                $fs.Flush()
            } finally { $fs.Dispose() }
            Remove-Item -LiteralPath $tail.temp -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Host ("     [失败] 就地截断失败：{0}（归档已保留；活动文件可能未被改写）" -f $_.Exception.Message)
            Remove-Item -LiteralPath $tail.temp -Force -ErrorAction SilentlyContinue
            [void]$results.Add([ordered]@{ file = $name; action = 'failed'; before_bytes = $before
                                           reason = $_.Exception.Message; archive = $arch.path })
            $script:failed++
            continue
        }
    }

    $after = (Get-Item -LiteralPath $path).Length
    Write-Host ("     保留样例：最近 {0} 行（逐行 JSON 校验通过）→ 处理后 {1}（{2} 字节），方式={3}" -f `
        $tail.lines, (Format-Size $after), $after, $method)

    # 归档清理（逐一打印被删除的文件名）
    $deleted = New-Object System.Collections.ArrayList
    if ($MaxArchives -gt 0) {
        $old = Get-ChildItem -LiteralPath $archiveDir -File -Filter "$base.*.gz" -ErrorAction SilentlyContinue |
               Sort-Object LastWriteTime -Descending | Select-Object -Skip $MaxArchives
        foreach ($f in $old) {
            Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue
            [void]$deleted.Add($f.Name)
            Write-Host ("     归档清理：删除最旧归档 {0}（超过 {1} 份上限）" -f $f.Name, $MaxArchives)
        }
    }

    [void]$results.Add([ordered]@{
        file = $name; action = 'rotated'; method = $method
        before_bytes = $before; before_human = (Format-Size $before)
        after_bytes = $after; after_human = (Format-Size $after)
        archived_bytes = $arch.bytes; archived_lines = $arch.lines
        archive_path = $arch.path; archive_gz_bytes = $arch.archiveBytes
        archive_verify = $arch.note; kept_lines = $tail.lines; kept_verify = $tail.note
        deleted_archives = @($deleted)
    })
}

# ---------------------------------------------------------------- 留痕
if (-not $DryRun) {
    $record = [ordered]@{
        generated_by = '部署/脚本/rotate_logs.ps1'
        run_at       = (Get-Date).ToString('s')
        log_dir      = $LogDir
        policy       = [ordered]@{ threshold_mb = $ThresholdMB; keep_lines = $KeepLines
                                   max_archives = $MaxArchives; force = [bool]$Force
                                   strategy = '归档为 gzip（整份）+ 活动文件保留最近 N 行；归档超上限时删最旧' }
        services_listening = $listening
        exit_code    = $script:failed
        files        = $script:results
        note         = '处理前后体积均为本次实跑实测；归档为「保留」而非删除（超上限的最旧归档单独列出）。'
    }
    try {
        $json = $record | ConvertTo-Json -Depth 8
        Set-Content -LiteralPath $recordPath -Encoding UTF8 -Value $json
        # 另外**按次留档**：同一份记录同时写进 归档/轮转记录_<时间戳>.json，
        # 这样多次轮转（例如先 copytruncate 后 rename）的记录都留得下来，不被最新一次覆盖。
        if (Test-Path -LiteralPath $archiveDir) {
            $histPath = Join-Path $archiveDir ("轮转记录_{0}.json" -f ([DateTime]::Now.ToString('yyyyMMdd-HHmmss')))
            Set-Content -LiteralPath $histPath -Encoding UTF8 -Value $json
            Write-Host ("     轮转留痕（按次）：{0}" -f (Split-Path -Leaf $histPath))
        }
        Write-Host ""
        Write-Host ("[OK] 已写轮转留痕 {0}" -f $recordPath)
    } catch { Write-Host ("[警告] 轮转留痕写出失败：{0}" -f $_.Exception.Message) }
}

Write-Host ""
Write-Host ("=" * 78)
Write-Host "轮转结果汇总"
Write-Host ("=" * 78)
foreach ($r in $results) {
    if ($r.action -eq 'rotated') {
        Write-Host ("  {0,-16} {1,10} → {2,-10} （归档 {3}，方式 {4}）" -f `
            $r.file, $r.before_human, $r.after_human, (Split-Path -Leaf $r.archive_path), $r.method)
    } elseif ($r.action -eq 'keep') {
        Write-Host ("  {0,-16} {1,10}   未触发（{2}）" -f $r.file, (Format-Size $r.before_bytes), $r.reason)
    } elseif ($r.action -eq 'dry-run') {
        Write-Host ("  {0,-16} {1,10}   将轮转（{2}）" -f $r.file, (Format-Size $r.before_bytes), $r.plan)
    } elseif ($r.action -eq 'failed') {
        Write-Host ("  {0,-16} 失败：{1}" -f $r.file, $r.reason) -ForegroundColor Red
    } else {
        Write-Host ("  {0,-16} 跳过（{1}）" -f $r.file, $r.reason)
    }
}
if ($DryRun) { Write-Host "  （DryRun：未修改任何文件）" }
if ($script:failed -gt 0) { Write-Host ("结论：{0} 个文件处理失败，退出码 1" -f $script:failed) -ForegroundColor Red; exit 1 }
Write-Host "结论：日志归档/轮转完成，退出码 0" -ForegroundColor Green
exit 0
