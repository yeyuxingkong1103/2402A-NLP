# =============================================================================
# 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
# 文件：部署/脚本/logs.ps1 —— 收集容器日志（验收项②「无异常日志输出」的证据来源）
#
# 用法：
#     pwsh -NoProfile -File 部署/脚本/logs.ps1                    # 每个服务最近 200 行
#     pwsh -NoProfile -File 部署/脚本/logs.ps1 -Tail 1000
#     pwsh -NoProfile -File 部署/脚本/logs.ps1 -Follow             # 实时跟随（Ctrl+C 退出）
#     pwsh -NoProfile -File 部署/脚本/logs.ps1 -ScanErrors         # 扫描 ERROR/CRITICAL/Traceback
#
# 产出：
#     * 部署/日志/容器日志-<时间戳>.log      全量日志快照
#     * 部署/日志/异常扫描-<时间戳>.log      异常关键字扫描结果（供验收逐条核对）
# =============================================================================
[CmdletBinding()]
param(
    [string]$EnvFile = '部署/配置/.env.example',
    [int]$Tail = 200,
    [switch]$Follow,
    [switch]$ScanErrors,
    [string[]]$Services = @('rag-api', 'rag-ui', 'redis')
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$logDir = Join-Path $repoRoot '部署\日志'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$snapFile = Join-Path $logDir "容器日志-$stamp.log"
$scanFile = Join-Path $logDir "异常扫描-$stamp.log"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[错误] 找不到 docker CLI"
    exit 127
}

$compose = @('compose', '--env-file', $EnvFile, '-f', '部署/docker/docker-compose.yml')

Write-Host "===== 容器日志采集 ====="
Write-Host "时间：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

if ($Follow) {
    & docker @compose logs -f --tail $Tail
    exit $LASTEXITCODE
}

"# 容器日志快照 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Set-Content -LiteralPath $snapFile -Encoding utf8

foreach ($svc in $Services) {
    Write-Host "`n---------- $svc ----------"
    $lines = & docker @compose logs --tail $Tail --no-color $svc 2>&1
    ($lines | Out-String).TrimEnd() | Add-Content -LiteralPath $snapFile -Encoding utf8
    $lines | Select-Object -Last 40 | ForEach-Object { Write-Host $_ }
}

# 异常关键字扫描：容器运行期「无异常日志输出」这条验收靠它给证据
$patterns = 'ERROR|CRITICAL|Traceback|Exception|CRIT|FATAL|panic|failed|refused|denied'
$hits = Select-String -LiteralPath $snapFile -Pattern $patterns -AllMatches -ErrorAction SilentlyContinue
$header = @(
    "# 异常关键字扫描 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "# 扫描文件：$snapFile",
    "# 关键字：$patterns",
    "# 命中行数：$(if ($hits) { @($hits).Count } else { 0 })",
    ""
)
$header | Set-Content -LiteralPath $scanFile -Encoding utf8
if ($hits) {
    $hits | ForEach-Object { "L$($_.LineNumber): $($_.Line.Trim())" } | Add-Content -LiteralPath $scanFile -Encoding utf8
    Write-Host "`n[注意] 命中 $(@($hits).Count) 行可疑日志 → $scanFile" -ForegroundColor Yellow
} else {
    "（未命中任何异常关键字 —— 运行期日志干净）" | Add-Content -LiteralPath $scanFile -Encoding utf8
    Write-Host "`n[OK] 未命中任何异常关键字" -ForegroundColor Green
}
Write-Host "日志快照：$snapFile"
Write-Host "扫描结果：$scanFile"
if ($ScanErrors -and $hits) { exit 1 }
exit 0
