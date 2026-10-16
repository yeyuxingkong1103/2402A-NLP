# =============================================================================
# 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
# 文件：部署/脚本/down.ps1 —— 停止服务（**默认保留卷**，数据不丢）
#
# 用法：
#     pwsh -NoProfile -File 部署/脚本/down.ps1                 # 停容器，卷保留 ← 默认
#     pwsh -NoProfile -File 部署/脚本/down.ps1 -RemoveVolumes  # ⚠️ 连卷一起删（会话/日志全没）
#     pwsh -NoProfile -File 部署/脚本/down.ps1 -RemoveImages   # 顺带删镜像
#
# 回滚语义（部署文档 §回滚）：
#    * 停服务 → 卷在 → 重新 up 后会话/日志/索引仍在（这是"数据不丢失"的正面验证）；
#    * -RemoveVolumes 只在确认要重置环境时使用，执行前脚本会要求二次确认。
# =============================================================================
[CmdletBinding()]
param(
    [string]$EnvFile = '部署/配置/.env.example',
    [switch]$RemoveVolumes,
    [switch]$RemoveImages,
    [switch]$WithMonitoring,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$logDir = Join-Path $repoRoot '部署\日志'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logFile = Join-Path $logDir "停止-$stamp.log"

function Write-Both([string]$Text) {
    Write-Host $Text
    Add-Content -LiteralPath $logFile -Value $Text -Encoding utf8
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Both "[错误] 找不到 docker CLI"
    exit 127
}

if ($RemoveVolumes -and -not $Force) {
    Write-Host "⚠️  -RemoveVolumes 会删除全部具名卷（索引/日志/会话/Redis 数据）。" -ForegroundColor Yellow
    $answer = Read-Host "确认继续？输入 yes 继续，其它任意键取消"
    if ($answer -ne 'yes') {
        Write-Both "[取消] 未执行任何操作"
        exit 0
    }
}

$compose = @('compose', '--env-file', $EnvFile, '-f', '部署/docker/docker-compose.yml')
$profile = @()
if ($WithMonitoring) { $profile = @('--profile', 'monitoring') }

Write-Both "===== 停止服务 ====="
Write-Both "时间：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Both "删卷：$(if ($RemoveVolumes) { '是（⚠️ 数据会丢）' } else { '否（默认，数据保留）' })"

$args = @('down', '--remove-orphans')
if ($RemoveVolumes) { $args += '--volumes' }
if ($RemoveImages) { $args += '--rmi', 'local' }

Write-Both "执行：docker $($compose -join ' ') $($profile -join ' ') $($args -join ' ')"
& docker @compose @profile @args 2>&1 | Tee-Object -FilePath $logFile -Append
$code = $LASTEXITCODE

Write-Both "`n当前容器："
& docker @compose @profile ps -a 2>&1 | Tee-Object -FilePath $logFile -Append
Write-Both "`n剩余卷（应保留数据卷）："
& docker volume ls --filter "name=rag-finqa" 2>&1 | Tee-Object -FilePath $logFile -Append
Write-Both "日志：$logFile"
exit $code
