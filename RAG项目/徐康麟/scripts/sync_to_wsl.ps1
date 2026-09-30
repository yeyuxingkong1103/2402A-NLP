<#
.SYNOPSIS
    从 Windows 一条命令把工作区代码同步到 WSL（Ubuntu），并可选跑部署自检 / 测试。

.DESCRIPTION
    只是 scripts/wsl_sync.sh 的薄封装：真正的同步逻辑（rsync + 校验）在 bash 侧，
    且那套逻辑已在 WSL 上实测通过。

    为什么用「PowerShell 调 wsl.exe 跑 bash 脚本」而不是在 PowerShell 里搬文件：
      * WSL 的文件系统（ext4）从 Windows 侧不可直接写，必须经由 WSL 内部操作；
      * 更重要的：**PowerShell 5.1 传原生参数时不转义内层双引号**，凡是把带引号的
        命令拼成一行的写法都会被截断（本项目已因此踩坑四次）。这里全程用参数数组
        （`& wsl.exe @Args`），命令行里一个双引号都不出现。

.PARAMETER Distro
    WSL 发行版名，默认 Ubuntu。

.PARAMETER Deploy
    同步后运行 scripts/deploy.sh（建/刷 .venv + 依赖 + 五项导入自检）。

.PARAMETER Test
    同步后运行 pytest tests -q（隐含 Deploy）。

.PARAMETER Dst
    WSL 内的目标目录，默认 WSL 的 ~/legal-rag。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\sync_to_wsl.ps1
    powershell -ExecutionPolicy Bypass -File scripts\sync_to_wsl.ps1 -Deploy
    powershell -ExecutionPolicy Bypass -File scripts\sync_to_wsl.ps1 -Test

.NOTES
    退出码：0 成功；2 参数/发行版名非法；3 找不到 wsl.exe；
            其余为 bash 侧脚本的退出码（1 工作区结构不对、3 同步失败、4 校验失败、5 部署失败、6 测试失败）。
#>
[CmdletBinding()]
param(
    [string]$Distro = 'Ubuntu',
    [switch]$Deploy,
    [switch]$Test,
    [string]$Dst = ''
)

$ErrorActionPreference = 'Stop'

function Write-Info([string]$m) { Write-Host $m -ForegroundColor Cyan }
function Write-Err2([string]$m) { Write-Host $m -ForegroundColor Red }

# --- 定位工作区与 bash 脚本 -------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Workspace = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$wslScriptPath = Join-Path $ScriptDir 'wsl_sync.sh'

if (-not (Test-Path -LiteralPath $wslScriptPath -PathType Leaf)) {
    Write-Err2 "找不到 $wslScriptPath"
    exit 2
}

# 发行版名只允许安全字符 —— 它会被拼进 wsl.exe 的参数，避免任何注入/引号问题
if ($Distro -notmatch '^[A-Za-z0-9._-]+$') {
    Write-Err2 "发行版名非法：$Distro（只允许字母数字 . _ -）"
    exit 2
}

function ConvertTo-WslPath([string]$WindowsPath) {
    <# E:\a\b -> /mnt/e/a/b #>
    $full = [IO.Path]::GetFullPath($WindowsPath)
    $drive = $full.Substring(0, 1).ToLower()
    $rest = $full.Substring(2).Replace('\', '/')
    return "/mnt/$drive$rest"
}

$wslScript = ConvertTo-WslPath $wslScriptPath

# --- 检查 wsl.exe ----------------------------------------------------------
$wslCmd = Get-Command 'wsl.exe' -ErrorAction SilentlyContinue
if (-not $wslCmd) {
    Write-Err2 '找不到 wsl.exe。修复建议：以管理员身份运行 `wsl --install`，或确认 Windows 功能「适用于 Linux 的 Windows 子系统」已启用。'
    exit 3
}

# --- 组装参数（全程数组，不拼命令行字符串）---------------------------------
$wslArgs = @('-d', $Distro, '-e', 'bash', $wslScript)
if (-not [string]::IsNullOrWhiteSpace($Dst)) { $wslArgs += @('--dst', $Dst) }
if ($Deploy) { $wslArgs += '--deploy' }
if ($Test) { $wslArgs += '--test' }

Write-Info '法律 RAG 同步到 WSL'
Write-Info ("  工作区   : {0}" -f $Workspace)
Write-Info ("  发行版   : {0}" -f $Distro)
Write-Info ("  WSL 脚本 : {0}" -f $wslScript)
if (-not [string]::IsNullOrWhiteSpace($Dst)) { Write-Info ("  目标目录 : {0}" -f $Dst) } else { Write-Info '  目标目录 : （默认）WSL 内的 ~/legal-rag' }
Write-Info ("  附加动作 : {0}" -f (@(@(); if ($Deploy) { '部署自检' }; if ($Test) { 'pytest' }) -join ' + '))
Write-Host ''

# --- 执行 ------------------------------------------------------------------
& wsl.exe @wslArgs
$code = $LASTEXITCODE

Write-Host ''
if ($code -eq 0) {
    Write-Info '同步完成（exit 0）'
} else {
    Write-Err2 "失败，退出码 $code"
    Write-Host '  排查建议：' -ForegroundColor Yellow
    Write-Host '    1) 发行版名不对：wsl -l -v 看实际名字，用 -Distro 指定' -ForegroundColor Yellow
    Write-Host '    2) WSL 进不去：wsl -d <发行版> -e echo ok' -ForegroundColor Yellow
    Write-Host '    3) 想看 bash 侧细节：wsl -d <发行版> -e bash <上面那个 WSL 脚本路径> --help' -ForegroundColor Yellow
}
exit $code
