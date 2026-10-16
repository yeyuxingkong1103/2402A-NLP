# =============================================================================
# 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
# 文件：部署/脚本/build.ps1 —— 构建金融问答系统镜像（多阶段构建）
#
# 用法（在 工单10/ 目录下）：
#     pwsh -NoProfile -File 部署/脚本/build.ps1
#     pwsh -NoProfile -File 部署/脚本/build.ps1 -Tag 1.0.0 -NoCache
#
# 参数：
#     -Tag      镜像标签（默认 rag-finqa:1.0.0）
#     -EnvFile  环境文件（默认 部署/配置/.env.example）
#     -NoCache  不使用构建缓存（测真实冷构建耗时）
#
# ⚠️ 前置条件：Docker Desktop 已启动，且**当前 shell 有权限访问 Docker 命名管道**
#    （npipe:////./pipe/dockerDesktopLinuxEngine）。若报
#    "permission denied while trying to connect to the docker API"，
#    说明当前沙箱/账户无管道权限，详见 部署/过程问题记录.md 问题1。
#
# 退出码：0 成功 / 1 构建失败 / 127 找不到 docker CLI
# =============================================================================
[CmdletBinding()]
param(
    [string]$Tag = 'rag-finqa:1.0.0',
    [string]$EnvFile = '部署/配置/.env.example',
    [switch]$NoCache,
    [string]$HttpProxy = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # 部署/脚本 → 部署 → 工单10
Set-Location $repoRoot

$logDir = Join-Path $repoRoot '部署\日志'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logFile = Join-Path $logDir "构建-$stamp.log"

function Write-Both([string]$Text) {
    Write-Host $Text
    Add-Content -LiteralPath $logFile -Value $Text -Encoding utf8
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Both "[错误] 找不到 docker CLI，请确认 Docker Desktop 已安装并在 PATH 中"
    exit 127
}

Write-Both "===== 镜像构建 ====="
Write-Both "工单编号：人工智能NLP-RAG 项目-金融问答系统部署"
Write-Both "时间       ：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Both "工作目录   ：$repoRoot"
Write-Both "镜像标签   ：$Tag"
Write-Both "Dockerfile ：部署/docker/Dockerfile"
Write-Both "环境文件   ：$EnvFile"
Write-Both "构建缓存   ：$(if ($NoCache) { '禁用（-NoCache）' } else { '启用' })"

$started = Get-Date
$args = @(
    'build',
    '--file', '部署/docker/Dockerfile',
    '--tag', $Tag,
    '--build-arg', 'PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple',
    '--build-arg', 'PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn',
    '--label', 'work_order=人工智能NLP-RAG 项目-金融问答系统部署',
    '.'
)
if ($NoCache) { $args += '--no-cache' }
if ($HttpProxy) {
    $args += @('--build-arg', "HTTP_PROXY=$HttpProxy", '--build-arg', "HTTPS_PROXY=$HttpProxy")
}

Write-Both "执行命令   ：docker $($args -join ' ')"
Write-Both ('-' * 78)

# 2>&1 合并 stderr：构建日志必须**完整**落盘（验收要看真实构建日志）
& docker @args 2>&1 | Tee-Object -FilePath $logFile -Append
$code = $LASTEXITCODE
$elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 2)

Write-Both ('-' * 78)
if ($code -eq 0) {
    Write-Both "[OK] 构建成功，耗时 $elapsed 秒"
    & docker images $Tag 2>&1 | Tee-Object -FilePath $logFile -Append
    Write-Both "镜像大小（真实数字，写入 优化/评估结果/）："
    & docker image inspect $Tag --format '{{.Size}}' 2>&1 | Tee-Object -FilePath $logFile -Append
} else {
    Write-Both "[FAIL] 构建失败，退出码 $code，耗时 $elapsed 秒"
    Write-Both "排查建议：① Docker 命名管道权限（过程问题记录 问题1）；② 镜像源超时（问题3）"
}
Write-Both "日志：$logFile"
exit $code
