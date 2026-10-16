# =============================================================================
# 工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单
# 文件：研发/剖析/探GPU.ps1 —— GPU / Ollama 驻留观测（nvidia-smi + /api/ps）
#
# 工单原文任务二-2-①：「使用 nvidia-smi 和 docker stats 持续监控 GPU 与容器内存变化」。
#   实测结论（设计/环境事实.md §2、§4）：
#     * nvidia-smi        ✅ 可用（NVIDIA GeForce RTX 2060, 6144 MiB）
#     * docker stats      ❌ 不可用（docker CLI 走 npipe，沙箱拒绝，退出码 1）
#   ⇒ GPU 用 nvidia-smi 真实测；「容器内存」用宿主进程 RSS/私有内存代替并注明口径。
#
# 用法：
#     pwsh -NoProfile -File 研发/剖析/探GPU.ps1                 # 单次快照
#     pwsh -NoProfile -File 研发/剖析/探GPU.ps1 -Watch 60 5     # 每 5 秒采一次，持续 60 秒
# =============================================================================
[CmdletBinding()]
param(
    [int]$Watch = 0,
    [int]$IntervalSec = 5,
    [string]$OutDir = ''
)

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $OutDir) { $OutDir = Join-Path $repoRoot '测试\测试数据' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$outFile = Join-Path $OutDir 'GPU与Ollama驻留观测.jsonl'

function Get-Snapshot {
    $row = [ordered]@{ ts = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') }
    try {
        $csv = (& nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu `
                --format=csv,noheader,nounits 2>$null) -join ''
        if ($csv) {
            $f = $csv -split ','
            $row.gpu_name = $f[0].Trim()
            $row.gpu_mem_total_mib = [int]$f[1].Trim()
            $row.gpu_mem_used_mib = [int]$f[2].Trim()
            $row.gpu_mem_free_mib = [int]$f[3].Trim()
            $row.gpu_util_pct = [int]$f[4].Trim()
            $row.nvidia_smi = 'ok'
        } else {
            $row.nvidia_smi = 'no-output'
        }
    } catch {
        $row.nvidia_smi = "failed: $($_.Exception.Message)"
    }
    try {
        $ps = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/ps' -TimeoutSec 5
        $models = @($ps.models)
        $row.ollama_models = @($models | ForEach-Object { $_.name })
        $row.ollama_count = $models.Count
        $row.ollama_vram_mib = [math]::Round((($models | Measure-Object -Property size_vram -Sum).Sum) / 1MB, 1)
        $row.ollama_expires = @($models | ForEach-Object { $_.expires_at })
    } catch {
        $row.ollama_models = @()
        $row.ollama_count = 0
        $row.ollama_error = "failed: $($_.Exception.Message)"
    }
    # 宿主进程口径（docker stats 替代）
    $row.api_private_mb = $null
    $pidFile = Join-Path $repoRoot '部署\日志\本地进程.json'
    if (Test-Path $pidFile) {
        try {
            $info = Get-Content $pidFile -Raw -Encoding utf8 | ConvertFrom-Json
            $target = if ($info.api_real_pid) { $info.api_real_pid } else { $info.api_pid }
            $p = Get-Process -Id $target -ErrorAction SilentlyContinue
            if ($p) {
                $row.api_pid = $p.Id
                $row.api_private_mb = [math]::Round($p.PrivateMemorySize64 / 1MB, 1)
                $row.api_rss_mb = [math]::Round($p.WorkingSet64 / 1MB, 1)
            }
        } catch { }
    }
    return [pscustomobject]$row
}

Write-Host "===== GPU / Ollama 驻留观测 ====="
Write-Host "nvidia-smi 可用性：$((Get-Command nvidia-smi -ErrorAction SilentlyContinue) ? '可用' : '不可用')"
Write-Host "docker stats      ：不可用（docker CLI 走 npipe，沙箱拒绝；见 设计/环境事实.md §2.1）"
Write-Host "容器内存口径      ：宿主进程 WorkingSet(RSS) / PrivateMemorySize（已注明）"
Write-Host "输出：$outFile"
Write-Host ""

$deadline = (Get-Date).AddSeconds($Watch)
$n = 0
do {
    $snap = Get-Snapshot
    $json = $snap | ConvertTo-Json -Compress -Depth 4
    Add-Content -LiteralPath $outFile -Value $json -Encoding UTF8
    $n++
    Write-Host ("[{0}] GPU 已用 {1}/{2} MiB ({3}%) util={4}%  |  Ollama 驻留 {5} 个 {6}  |  API 私有 {7} MB" -f `
        $snap.ts, $snap.gpu_mem_used_mib, $snap.gpu_mem_total_mib,
        ([math]::Round(100 * $snap.gpu_mem_used_mib / [math]::Max($snap.gpu_mem_total_mib, 1), 1)),
        $snap.gpu_util_pct, $snap.ollama_count, ($snap.ollama_models -join ','), $snap.api_private_mb)
    if ($Watch -le 0) { break }
    Start-Sleep -Seconds $IntervalSec
} while ((Get-Date) -lt $deadline)

Write-Host "`n采样 $n 次，已追加到 $outFile"
