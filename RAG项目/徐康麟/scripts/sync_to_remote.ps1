<#
.SYNOPSIS
    Windows 侧「同步到远端 + 自检」脚本（法律 RAG / VM-DEPLOY.md §2 的固化版）

.DESCRIPTION
    把 VM-DEPLOY.md「§2 一条命令完成『同步 + 刷依赖 + 自检』」里那条已被实测复跑过的
    命令固化成脚本，并补上：排除清单、逐字节（sha256）自检、四类差异计数。

    四步：
      [1/4] 本地清理 __pycache__ / *.pyc
      [2/4] scp -r 同步源码目录（必须带 -r；不带会直接失败，VM-DEPLOY.md §7 发现 1）
      [3/4] 远端解字节码 + 刷新依赖 + 五项导入自检
      [4/4] 逐字节自检：两侧各自生成「相对路径|大小|sha256」清单并比对，
            打印 identical / 缺失 / 不同 / 多余 四个数字

    本脚本刻意不传 data/（原始语料，RAG 用不到），也不动 Redis / Milvus。

.PARAMETER RemoteUser
    远端 SSH 用户名。默认 dshagent（VM-DEPLOY.md §1 记录的目标机用户）。
    换机器时显式传：-RemoteUser ubuntu

.PARAMETER RemoteHost
    远端主机名或 IP。默认 192.168.188.128 只是 VM-DEPLOY.md 里那台机器的历史值，
    不是唯一选项 —— 换机器时显式传：-RemoteHost 10.0.0.7

.PARAMETER Port
    SSH 端口，默认 22。

.PARAMETER KeyPath
    SSH 私钥路径。默认 <工作区根>\.ssh\dsh_ubuntu_ed25519
    （VM-DEPLOY.md §1：私钥就在工作区 .ssh\ 下，沙箱不允许写用户 ~\.ssh）。

.PARAMETER KnownHosts
    已知主机文件。默认 <工作区根>\.ssh\known_hosts
    （VM-DEPLOY.md §1：必须显式 -o UserKnownHostsFile，否则 BatchMode 下会直接失败）。
    文件不存在时会自动创建空文件并提示：首次连接请先手动 ssh 一次确认指纹。

.PARAMETER RemoteDir
    远端项目目录（相对远端家目录）。默认 legal-rag（VM-DEPLOY.md §1：~/legal-rag）。

.PARAMETER WorkspaceRoot
    本地工作区根目录。默认取本脚本所在 scripts/ 的上一级。

.PARAMETER ReportPath
    自检报告输出路径。默认 <工作区根>\.tmp\sync_report_<时间戳>.txt

.PARAMETER WhatIf
    只打印将要执行的步骤，不做任何同步 / 不改动远端，**也不产生任何本地副作用**
    （不创建 .ssh\known_hosts、不创建 .tmp\、不写报告文件）。

.EXAMPLE
    # 最常用：用默认参数同步到 VM-DEPLOY.md 里那台 VM
    powershell -ExecutionPolicy Bypass -File scripts\sync_to_remote.ps1

.EXAMPLE
    # 换一台机器：所有连接参数都显式给
    powershell -ExecutionPolicy Bypass -File scripts\sync_to_remote.ps1 `
        -RemoteUser ubuntu -RemoteHost 10.10.0.21 -Port 2222 `
        -KeyPath C:\keys\id_ed25519 -RemoteDir projects\legal-rag

.EXAMPLE
    # 先看要做什么，不动远端
    powershell -ExecutionPolicy Bypass -File scripts\sync_to_remote.ps1 -WhatIf

.NOTES
    实测坑（都写在 VM-DEPLOY.md §7 发现 1 里，本脚本已规避）：
      1) scp 同步目录必须带 -r，否则 scp.exe 报 "is not a regular file" 直接失败。
      2) PowerShell 5.1 会把 ssh "… python -c \"…\"" 里的内层 \" 剥掉，bash 收到残缺命令
         → 语法错误。所以远端命令一律「外层双引号 + 内层单引号」。
      3) 含中文的脚本体不要用管道直接喂 ssh（PowerShell 会加 CRLF / 按 GBK 误解），
         要用 base64 传输再 `base64 -d | bash -s`。
      4) 清单里的文件先按**严格 UTF-8**（`Text.UTF8Encoding($false, $true)` + `GetString`）
         解码一遍：非法字节会抛 `DecoderFallbackException` 并带上文件路径，而不是被静默
         替换成 U+FFFD（早期用 `[Text.Encoding]::UTF8` 或 `ReadAllText($p, $strict)` 时
         那句「不是合法 UTF-8 会抛错」**不成立** —— 前者是替换回退，后者带 BOM 嗅探会把
         `FF FE` 当 UTF-16LE）。**已知二进制后缀例外**（见 $BinaryExtensions）：
         `tests/fixtures/sample_cn.pdf` 实测就不是合法 UTF-8，只做 sha256、不做解码校验。
         哈希一律用 Get-FileHash 取「原始字节」的 sha256 —— 必须与远端 sha256sum 同口径
         （含 BOM 也要一致）。
      5) 历史坑（函数 New-CmdLine 已删除，教训保留）：参数名**不能**叫 $Args —— 那是
         PowerShell 的自动变量（存放未绑定参数），会遮蔽同名参数、拼出空命令行，
         真机跑才暴露（AST 解析与辅助函数单测都发现不了）。现在两个调用点都直接
         `& $exe @args`，由 PowerShell 逐个传参，不再手写 cmd.exe 引用规则。
#>
[CmdletBinding()]
param(
    [string]$RemoteUser = 'dshagent',
    [string]$RemoteHost = '192.168.188.128',
    [int]$Port = 22,
    [string]$KeyPath = '',
    [string]$KnownHosts = '',
    [string]$RemoteDir = 'legal-rag',
    [string]$WorkspaceRoot = '',
    [string]$ReportPath = '',
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

# ===========================================================================
# 0. 常量：同步范围 / 排除清单 / 导入自检模块
#    同步范围严格照抄 VM-DEPLOY.md §2.1：7 个目录 + 7 个顶层文件
#    项数**不写死**：由脚本扫描后打印（见下面的「本地项数 / 远端项数」）。
#    历史记录：VM-DEPLOY.md 那轮实测 94 项；此后加入 scripts/deploy.sh、
#    scripts/sync_to_remote.ps1、tests/test_openai_compat.py、
#    docs/{CLOUD,WSL,ARCHITECTURE}.md 等文件，按同一扫描口径在本机实测为 **100 项** ——
#    所以任何硬编码的数字都会过时，判断以脚本打印的实测值为准。
# ===========================================================================
$SyncDirs = @('legal_rag', 'scripts', 'tests', 'knowledge', 'docs', 'sql', 'data_pipeline')
$SyncFiles = @(
    'requirements.txt', 'requirements-full.txt', 'README.md', '.env.example',
    'run.sh', 'shutdown.sh', '.gitignore'
)

# 必须排除的目录名（按目录名匹配，任意层级）
#   data/        用户明确要求不上传（94MB 原始语料，RAG 用不到）
#   .venv/       Windows 的虚拟环境不能传到 Linux
#   __pycache__/ 字节码
#   .tmp/        VM-DEPLOY.md 里放临时清单/脚本的地方
#   logs/ run/   运行时产物
#   index/ uploads/  入库登记与上传落盘
#   .pytest_cache/
# 注：因为按「目录名」匹配，仓库里任何一层叫 data 的目录都会被排除 —— 这是刻意的，
#     宁可少传也不能把 data/ 传上去。
$ExcludeDirNames = @(
    'data', '.venv', '__pycache__', '.tmp', 'logs', 'run',
    'index', 'uploads', '.pytest_cache'
)
$ExcludeNamePattern = '(^|/)(' + (($ExcludeDirNames | ForEach-Object { [Regex]::Escape($_) }) -join '|') + ')(/|$)'

$ImportMods = @(
    'legal_rag.metrics',
    'legal_rag.observability',
    'legal_rag.logging_setup',
    'legal_rag.config',
    'legal_rag.system_metrics'
)

#: 明确按「二进制」处理的后缀：它们**只做逐字节 sha256 比对**，不做 UTF-8 解码校验。
#: 为什么必须留这个口子：同步范围里有 `tests/fixtures/sample_cn.pdf`（PDF 夹具，
#: 实测确为非 UTF-8 —— 字节 0xE2 处 continuation byte 非法）。严格解码会直接抛错、
#: 让第 [4/4] 步中断；而二进制文件本来就不该按文本读。
#: 这一清单只影响「文本解码校验」，不影响 sha256（字节级一致性照旧）。
$BinaryExtensions = @(
    '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp',
    '.zip', '.gz', '.tar', '.7z', '.xz', '.bz2',
    '.woff', '.woff2', '.ttf', '.otf', '.eot',
    '.xlsx', '.xls', '.docx', '.doc', '.pptx', '.parquet', '.sqlite', '.db',
    '.bin', '.exe', '.dll', '.so', '.onnx', '.pt', '.pth', '.safetensors'
)

#: 远端输出里的分隔标记：清单在标记之前，摘要块在标记之后
$ManifestMarker = '===DSH-MANIFEST==='

# ===========================================================================
# 辅助函数
# ===========================================================================
function Write-Step {
    param([string]$Text)
    Write-Host ''
    Write-Host ('=' * 74) -ForegroundColor DarkGray
    Write-Host $Text -ForegroundColor Cyan
    Write-Host ('=' * 74) -ForegroundColor DarkGray
}

function Write-Info  { param([string]$Text) Write-Host "      $Text" -ForegroundColor Gray }
function Write-Ok    { param([string]$Text) Write-Host "      OK  $Text" -ForegroundColor Green }
function Write-Warn2 { param([string]$Text) Write-Host "      警告  $Text" -ForegroundColor Yellow }
function Write-Err2  { param([string]$Text) Write-Host "      错误  $Text" -ForegroundColor Red }

function New-B64File {
    <# 把文本按 UTF-8 + LF 写文件，并返回该文件的 base64 文本。
       含中文的脚本/清单都走这条路（VM-DEPLOY.md §7 发现 1：不要用管道喂 ssh）。 #>
    param([string]$Text)
    $path = Join-Path $script:TempDir ([Guid]::NewGuid().ToString('N') + '.b64')
    $norm = $Text -replace "`r`n", "`n"
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($norm))
    [IO.File]::WriteAllText($path, $b64, (New-Object Text.ASCIIEncoding))
    return $path
}

function Invoke-RemoteBash {
    <# 把 bash 脚本 base64 后交给远端执行：
         ssh … "base64 -d | bash -s"   ← base64 通过 stdin 送进 ssh
       这里**不**用 cmd.exe 做管道：base64 是纯 ASCII，PowerShell 管道送它没有编码风险；
       而经 cmd + 手写引用传远端命令会多一层引号，远端会把整条管道当成一个命令名
       （实测：`bash: 行 1: base64 -d | bash -s: 未找到命令`，退出码 127）。 #>
    param(
        [Parameter(Mandatory = $true)][string]$ScriptText,
        [Parameter(Mandatory = $true)][string]$LogPath,
        [string]$Label = 'deploy'
    )
    $b64Path = New-B64File -Text $ScriptText
    $remoteB64 = '/tmp/' + [IO.Path]::GetFileName($b64Path)
    try {
        # 两步都只用「已实测可用」的直接调用：
        #   1) 先把 base64 **文件**用 scp 送过去 —— 纯 ASCII，逐字节可靠；
        #   2) 再 ssh 让远端 `base64 -d <文件> | bash -s`。
        # 为什么不用 stdin 管道：PowerShell 5.1 往原生命令 stdin 送字符串时会改动字节，
        #   实测远端报 `base64: 无效的输入`（GNU base64 -d 对多余的 CR 是严格的）。
        # 为什么不用 cmd.exe / Start-Process：前者多一层引号，实测远端把整条管道当成一个
        #   命令名（`bash: 行 1: base64 -d | bash -s: 未找到命令`，退出码 127）；
        #   后者在受限环境会被拒（Access is denied）。
        & $script:ScpExe @($script:ScpArgs + @($b64Path, "$($script:Target):$remoteB64")) *> ($LogPath + '.xfer')
        $remoteCmd = "base64 -d $remoteB64 | bash -s; rm -f $remoteB64"
        $sshArgs = $script:SshArgs + @($script:Target, $remoteCmd)
        & $script:SshExe @sshArgs *> $LogPath
        $code = $LASTEXITCODE
    } finally {
        if (Test-Path -LiteralPath $b64Path) { Remove-Item -LiteralPath $b64Path -Force -ErrorAction SilentlyContinue }
    }
    Write-Info ("远端 {$Label} 退出码 = $code ；原始输出见 $LogPath")
    return $code
}

function Test-IsBinaryByExtension {
    <# 该文件是否属于「已知二进制后缀」（只做 sha256，不做 UTF-8 解码校验）。
       见 $BinaryExtensions 的说明：同步范围内的 tests/fixtures/sample_cn.pdf 是二进制。 #>
    param([string]$Path)
    $ext = [IO.Path]::GetExtension($Path).ToLowerInvariant()
    return ($BinaryExtensions -contains $ext)
}

function Get-Utf8Checked {
    <# VM-DEPLOY.md §2.2 的口径：清单里的文件按 UTF-8 读一遍，避免按本地码页(GBK)误解。
       **这里是真正的严格解码**：先取**原始字节**，再用 `Text.UTF8Encoding($false, $true)`
       （throwOnInvalidBytes = $true）的 `GetString()` 解码 —— 非法 UTF-8 字节会抛
       DecoderFallbackException，于是「哪个文件、什么问题」被直接暴露，而不是被静默
       替换成 U+FFFD 后算出一个错的哈希。
       注意两点（都实测过）：
         * `[Text.Encoding]::UTF8` 是**替换回退**（不抛错），用它达不到校验目的；
         * 不能写成 `[IO.File]::ReadAllText($p, $strict)` —— 它带 BOM 嗅探，遇到
           `FF FE` 开头会按 UTF-16LE 解码（既不抛错、内容也不对）。
           所以必须自己 ReadAllBytes + GetString 才真的严格。 #>
    param([string]$Path)
    $strict = New-Object Text.UTF8Encoding($false, $true)
    $bytes = [IO.File]::ReadAllBytes($Path)
    try {
        $text = $strict.GetString($bytes)
    } catch [System.Text.DecoderFallbackException] {
        throw ("文件不是合法 UTF-8（存在无法解码的字节）：{0} —— {1}" -f $Path, $_.Exception.Message)
    }
    # 与原先 ReadAllText 的行为对齐：剥掉 UTF-8 BOM（调用方目前只用于「按 UTF-8 读一遍」校验）
    if ($text.Length -gt 0 -and $text[0] -eq [char]0xFEFF) { $text = $text.Substring(1) }
    return $text
}

function Get-Sha256Hex {
    <# 算 sha256：必须与远端 `sha256sum` 的口径完全一致 —— 即「文件原始字节」。
       本机实测（很重要）：
         Get-FileHash / 远端 sha256sum  → 原始字节（含 UTF-8 BOM）
         ReadAllText + UTF8.GetBytes    → 已剥掉 BOM（差 3 字节 EF BB BF）
       同一个带 BOM 的文件实测：Get-FileHash=aeb8bc1f… 而 ReadAllText+GetBytes=b78f0a46…，
       若本地用后者、远端用前者，就会把「只多了 BOM」的文件误报成「不同」。
       所以哈希用 Get-FileHash 取原始字节；「按 UTF-8 读」由 Get-Utf8Checked 单独负责。
       两者结合：既满足按 UTF-8 读的防乱码要求，又保证与 Linux 逐字节可比。 #>
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Invoke-Scanned {
    <# 走一个目录树，返回符合条件的文件（排除 排除清单 里的目录名） #>
    param(
        [Parameter(Mandatory = $true)][string]$Root
    )
    $result = New-Object 'System.Collections.Generic.List[string]'
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
        Write-Warn2 "目录不存在，跳过扫描：$Root"
        return $result
    }
    $files = Get-ChildItem -LiteralPath $Root -Recurse -File -Force -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        $rel = $f.FullName.Substring($Root.Length).TrimStart('\', '/') -replace '\\', '/'
        if ($rel -match $ExcludeNamePattern) { continue }
        if ($f.Extension -eq '.pyc' -or $f.Extension -eq '.pyo') { continue }
        [void]$result.Add($rel)
    }
    return $result
}

# ===========================================================================
# 1. 参数解析与前置校验
# ===========================================================================
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $WorkspaceRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
} else {
    $WorkspaceRoot = (Resolve-Path -LiteralPath $WorkspaceRoot).Path
}
$TempDir = Join-Path $WorkspaceRoot '.tmp'
$RemoteDir = $RemoteDir.Replace('\', '/').Trim('/')

$sshdExe = (Get-Command 'ssh' -ErrorAction SilentlyContinue)
$scpdExe = (Get-Command 'scp' -ErrorAction SilentlyContinue)
if (-not $sshdExe -or -not $scpdExe) {
    Write-Err2 '找不到 ssh / scp 命令。'
    Write-Info '修复建议：安装 Windows 的 OpenSSH 客户端（设置 → 应用 → 可选功能 → 添加功能 → OpenSSH 客户端），'
    Write-Info '          或确认 C:\Windows\System32\OpenSSH 在 PATH 里（Get-Command ssh）。'
    exit 2
}
$script:SshExe = $sshdExe.Source
$script:ScpExe = $scpdExe.Source

if ([string]::IsNullOrWhiteSpace($KeyPath))     { $KeyPath = Join-Path $WorkspaceRoot '.ssh\dsh_ubuntu_ed25519' }
if ([string]::IsNullOrWhiteSpace($KnownHosts))  { $KnownHosts = Join-Path $WorkspaceRoot '.ssh\known_hosts' }
$KeyPath = [IO.Path]::GetFullPath($KeyPath)
$KnownHosts = [IO.Path]::GetFullPath($KnownHosts)

if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $ReportPath = Join-Path $TempDir ('sync_report_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.txt')
}

if (-not (Test-Path -LiteralPath $KeyPath -PathType Leaf)) {
    Write-Err2 "SSH 私钥不存在：$KeyPath"
    Write-Info '修复建议：用 -KeyPath 显式指定私钥；VM-DEPLOY.md §1 记录默认位置是 <工作区>\.ssh\dsh_ubuntu_ed25519。'
    exit 2
}
if (-not (Test-Path -LiteralPath (Join-Path $WorkspaceRoot 'legal_rag') -PathType Container)) {
    Write-Err2 "工作区根目录不对：$WorkspaceRoot 下没有 legal_rag\"
    Write-Info '修复建议：用 -WorkspaceRoot 显式指定工作区根目录（例如 E:\deepseekharness\LLaMA-Factory）。'
    exit 2
}

# SSH / SCP 公共参数（全部照抄 VM-DEPLOY.md §1 / §2.1 的实测可用形式）
$CommonArgs = @(
    '-i', $KeyPath,
    '-o', "UserKnownHostsFile=$KnownHosts",
    '-o', 'BatchMode=yes',
    '-o', 'IdentitiesOnly=yes',
    '-o', 'ConnectTimeout=15'
)
$script:SshArgs = $CommonArgs + @('-p', "$Port")
$ScpArgs = $CommonArgs + @('-P', "$Port")
$Target = "$RemoteUser@$RemoteHost"

# 非 ASCII 参数会让 PowerShell 5.1 的内建参数编码（ANSI）出错。本机路径/主机名是
# 纯 ASCII 时正常；一旦出现中文路径，直接明确报错，而不是让 ssh 收到乱码。
$nonAscii = @()
foreach ($pair in @(
    @{ n = 'KeyPath'; v = $KeyPath }, @{ n = 'KnownHosts'; v = $KnownHosts },
    @{ n = 'Target'; v = $Target }, @{ n = 'WorkspaceRoot'; v = $WorkspaceRoot },
    @{ n = 'RemoteDir'; v = $RemoteDir }, @{ n = 'ReportPath'; v = $ReportPath }
)) {
    if ($pair.v -match '[^\x00-\x7F]') { $nonAscii += ($pair.n + '=' + $pair.v) }
}
if ($nonAscii.Count -gt 0) {
    Write-Err2 '以下参数含非 ASCII 字符，PowerShell 5.1 会把它们按 ANSI 传给 ssh/scp，导致乱码或连接失败：'
    foreach ($x in $nonAscii) { Write-Host "        $x" -ForegroundColor Red }
    Write-Info '修复建议：把工作区/私钥放到纯 ASCII 路径下（例如 E:\work\LLaMA-Factory），'
    Write-Info '          或用 -WorkspaceRoot / -KeyPath 指到不含中文的路径。'
    exit 2
}
if ($RemoteDir -match '[\s":\\]') {
    Write-Err2 "-RemoteDir 不能含空格、引号、反斜杠或冒号：$RemoteDir"
    Write-Info '修复建议：远端目录用不含特殊字符的相对路径，例如 -RemoteDir legal-rag 或 -RemoteDir projects/legal-rag。'
    exit 2
}

Write-Host ''
Write-Host '法律 RAG 同步 + 自检（Windows → 远端）' -ForegroundColor White
Write-Host ("  工作区      : {0}" -f $WorkspaceRoot)
Write-Host ("  远端        : {0}:{1}  (端口 {2})" -f $Target, $RemoteDir, $Port)
Write-Host ("  私钥        : {0}" -f $KeyPath)
Write-Host ("  已知主机    : {0}" -f $KnownHosts)
Write-Host ("  报告        : {0}" -f $ReportPath)
Write-Host ("  同步范围    : {0} 个目录 + {1} 个顶层文件" -f $SyncDirs.Count, $SyncFiles.Count)
Write-Host ("  排除        : {0}" -f ($ExcludeDirNames -join ' / '))
if ($WhatIf) { Write-Host '  模式        : -WhatIf（只打印，不执行）' -ForegroundColor Yellow }

# -WhatIf 必须在这里**早退**：此时还没有创建 .ssh\known_hosts、也没有创建 .tmp\，
# 因此「只打印，不执行」是真的零副作用（此前该判断在两处创建之后，会留下本地文件）。
if ($WhatIf) {
    Write-Host ''
    Write-Host '按 -WhatIf：以下命令不会真正执行。' -ForegroundColor Yellow
    Write-Host ("  [2/4] scp -r {0} … {1}:{2}/" -f ($SyncDirs -join ' '), $Target, $RemoteDir)
    Write-Host ("  [3/4] ssh … {0} `"cd ~/{1} && … base64 -d | bash -s`"" -f $Target, $RemoteDir)
    Write-Host '  [4/4] 两侧生成 sha256 清单并比对（identical / 缺失 / 不同 / 多余）'
    Write-Host '  未创建任何本地文件（.ssh\known_hosts / .tmp\ 都留给真正执行时再建）。' -ForegroundColor Yellow
    exit 0
}

# ---------------------------------------------------------------------------
# 已知主机文件（**放在 -WhatIf 早退之后**，避免「只打印」模式产生本地副作用）
# ---------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $KnownHosts)) {
    Write-Warn2 "known_hosts 不存在，正在创建空文件：$KnownHosts"
    Write-Info '首次连这台机器请先在普通终端手动 ssh 一次确认指纹，或确认主机已在别的 known_hosts 里；'
    Write-Info 'BatchMode=yes 下遇到未知主机指纹会直接失败（这是有意的，避免静默接受陌生指纹）。'
    $khDir = Split-Path -Parent $KnownHosts
    if (-not (Test-Path -LiteralPath $khDir)) { New-Item -ItemType Directory -Path $khDir -Force | Out-Null }
    [IO.File]::WriteAllText($KnownHosts, '', (New-Object Text.UTF8Encoding($false)))
}

# .tmp\（清单 / 日志 / 报告用）由第 [1/4] 步创建，此处不再提前创建。
Write-Host ''

# ===========================================================================
# 2. [1/4] 本地清理 __pycache__ / *.pyc
# ===========================================================================
Write-Step '[1/4] 本地清理 __pycache__ / *.pyc'

# 必须先建 .tmp：后面第 [4/4] 步要用它放清单/日志
if (-not (Test-Path -LiteralPath $TempDir)) {
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
}

$pycDirsRemoved = 0
$pycFilesRemoved = 0
foreach ($d in $SyncDirs) {
    $full = Join-Path $WorkspaceRoot $d
    if (-not (Test-Path -LiteralPath $full)) { Write-Warn2 "跳过不存在的目录：$d"; continue }
    $dirs = Get-ChildItem -LiteralPath $full -Recurse -Directory -Filter '__pycache__' -Force -ErrorAction SilentlyContinue
    foreach ($p in $dirs) {
        Remove-Item -LiteralPath $p.FullName -Recurse -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path -LiteralPath $p.FullName)) { $pycDirsRemoved++ }
    }
    $stray = Get-ChildItem -LiteralPath $full -Recurse -File -Force -ErrorAction SilentlyContinue |
             Where-Object { $_.Extension -eq '.pyc' -or $_.Extension -eq '.pyo' }
    foreach ($p in $stray) {
        Remove-Item -LiteralPath $p.FullName -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path -LiteralPath $p.FullName)) { $pycFilesRemoved++ }
    }
}
Write-Ok "已删除 __pycache__ 目录 $pycDirsRemoved 个、散落 .pyc/.pyo $pycFilesRemoved 个"
Write-Info '只清理要同步的 7 个目录（VM-DEPLOY.md §2.1 的做法：别 -Recurse 扫全仓，.tmp/.pytest-tmp 会刷一屏 Access Denied）'

# 同步源清单自证
$missingLocal = @()
foreach ($d in $SyncDirs)  { if (-not (Test-Path -LiteralPath (Join-Path $WorkspaceRoot $d) -PathType Container)) { $missingLocal += "$d\" } }
foreach ($f in $SyncFiles) { if (-not (Test-Path -LiteralPath (Join-Path $WorkspaceRoot $f) -PathType Leaf))      { $missingLocal += $f } }
if ($missingLocal.Count -gt 0) {
    Write-Warn2 ('本地缺少这些同步源：' + ($missingLocal -join '、'))
    Write-Info '缺失项不会被同步，第 [4/4] 步不会把它算成差异；若属于预期外的缺失，请先补上再重跑。'
} else {
    Write-Ok ('同步源齐全：{0} 个目录 + {1} 个顶层文件' -f $SyncDirs.Count, $SyncFiles.Count)
}

# ===========================================================================
# 3. [2/4] scp -r 同步
# ===========================================================================
Write-Step '[2/4] scp -r 同步源码目录'

$scpSources = @()
foreach ($d in $SyncDirs)  { if (Test-Path -LiteralPath (Join-Path $WorkspaceRoot $d) -PathType Container) { $scpSources += $d } }
foreach ($f in $SyncFiles) { if (Test-Path -LiteralPath (Join-Path $WorkspaceRoot $f) -PathType Leaf)      { $scpSources += $f } }

$scpLog = Join-Path $TempDir 'sync_scp.log'
# 注意 -r 必须有：VM-DEPLOY.md §7 发现 1 实测，缺 -r 时 scp.exe 直接
#   "local \"legal_rag\" is not a regular file" 并以 exit=1 失败。
$scpAll = $ScpArgs + @('-r') + $scpSources + @("${Target}:${RemoteDir}/")
Write-Info ("执行：scp -r … {0}:{1}/  （共 {2} 项）" -f $Target, $RemoteDir, $scpSources.Count)

# 直接调用 scp（不用 Start-Process：它在受限环境会被拒；& 的参数由 PowerShell 逐个传递，
# 因此也不需要手写 Windows 命令行引用规则）。
Push-Location $WorkspaceRoot
try {
    & $script:ScpExe @scpAll > $scpLog 2> ($scpLog + '.err')
    $scpCode = $LASTEXITCODE
} finally { Pop-Location }
if ($scpCode -ne 0) {
    Write-Err2 "scp 失败，退出码 $scpCode"
    foreach ($lf in @($scpLog, $scpLog + '.err')) {
        if (Test-Path -LiteralPath $lf) { Get-Content -LiteralPath $lf -Tail 20 | ForEach-Object { Write-Host "        $_" -ForegroundColor DarkGray } }
    }
    Write-Host '      修复建议（按可能性排序）：' -ForegroundColor Yellow
    Write-Host '        1) 确认远端目录存在且可写：ssh … "mkdir -p ~/legal-rag && test -w ~/legal-rag && echo writable"' -ForegroundColor Yellow
    Write-Host '        2) 私钥/权限：ssh 用 -v 复跑一次看认证阶段（BatchMode=yes 不会提示密码，失败就是失败）' -ForegroundColor Yellow
    Write-Host '        3) 首次连接指纹未知：先在普通终端手动 ssh 一次，或检查 -KnownHosts 指向的文件' -ForegroundColor Yellow
    Write-Host '        4) 主机不可达：Test-NetConnection <host> -Port 22' -ForegroundColor Yellow
    exit 3
}
Write-Ok 'scp 完成（exit 0）'
Write-Info 'OpenSSH(SFTP 模式) 是「合并」语义：不会生成 legal_rag/legal_rag 嵌套，也不会删除远端多余文件（所以第 [4/4] 步要看「多余」计数）'

# ===========================================================================
# 4. [3/4] 远端：解字节码 + 刷依赖 + 五项导入自检
# ===========================================================================
Write-Step '[3/4] 远端解字节码 + 刷新依赖 + 五项导入自检'

# 远端命令一律「外层双引号 + 内层单引号」（VM-DEPLOY.md §7 发现 1：PowerShell 5.1
# 会把 \" 剥掉，bash 收到残缺命令直接语法错误）。
$importList = ($ImportMods -join ', ')
$remoteStep3 = @"
set -euo pipefail
cd ~/$RemoteDir

echo '[远端 1/3] 清理字节码'
find . -path ./.venv -prune -o -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
echo '          已清理 __pycache__（.venv 保持原样）'

echo '[远端 2/3] 刷新依赖'
.venv/bin/python -m pip install -q -r requirements.txt
echo '          pip install -r requirements.txt 完成'

echo '[远端 3/3] 五项导入自检'
.venv/bin/python -c 'import $importList; print(123456)'

echo 'REMOTE_STEP3_OK'
"@

$step3Log = Join-Path $TempDir 'sync_step3.log'
$step3Rc = Invoke-RemoteBash -ScriptText $remoteStep3 -LogPath $step3Log -Label 'step3'
$step3Text = ''
if (Test-Path -LiteralPath $step3Log) { $step3Text = [IO.File]::ReadAllText($step3Log, [Text.Encoding]::UTF8) }
$step3Text -split "`r?`n" | Where-Object { $_ -ne '' } | ForEach-Object { Write-Host "        $_" -ForegroundColor DarkGray }

if ($step3Rc -ne 0 -or $step3Text -notmatch 'REMOTE_STEP3_OK') {
    Write-Err2 "远端第 [3/4] 步失败（退出码 $step3Rc，或没有看到 REMOTE_STEP3_OK 标记）"
    Write-Host '      修复建议：' -ForegroundColor Yellow
    Write-Host '        1) 若 cd 失败 → 远端没有该目录：ssh … "mkdir -p ~/legal-rag" 后重跑（或 -RemoteDir 传对）' -ForegroundColor Yellow
    Write-Host '        2) 若是 ModuleNotFoundError: legal_rag.xxx → 源码没同步全，看第 [4/4] 步的「缺失」数' -ForegroundColor Yellow
    Write-Host '        3) 若是 ModuleNotFoundError: pydantic/redis/pymilvus → 远端 .venv 没建好：' -ForegroundColor Yellow
    Write-Host '           ssh … "cd ~/legal-rag && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"' -ForegroundColor Yellow
    Write-Host '        4) 上方 log 里 pip 的报错原样贴给归属人（本脚本刻意不隐藏 pip 输出）' -ForegroundColor Yellow
    exit 4
}
Write-Ok '远端解字节码 / 刷依赖 / 五项导入自检 全部通过（打印了 123456）'

# ===========================================================================
# 5. 本地清单（相对路径|大小|sha256）
# ===========================================================================
Write-Step '[4/4] 逐字节自检（sha256 清单比对）'

$localManifestPath = Join-Path $TempDir 'manifest_local.txt'
$localMap = @{}
$localBytes = New-Object 'System.Collections.Generic.List[byte]'

foreach ($d in $SyncDirs) {
    $root = Join-Path $WorkspaceRoot $d
    $rels = Invoke-Scanned -Root $root
    foreach ($rel in $rels) {
        $full = Join-Path $root ($rel -replace '/', '\')
        # 按 UTF-8 读一遍：不是合法 UTF-8 就在这里报出来（已知二进制后缀除外，见常量段）
        if (-not (Test-IsBinaryByExtension -Path $full)) { [void](Get-Utf8Checked -Path $full) }
        $item = Get-Item -LiteralPath $full -Force
        $rr = "$d/$rel"
        $localMap[$rr] = [pscustomobject]@{ Size = [int64]$item.Length; Sha = (Get-Sha256Hex -Path $full) }
        $line = '{0}|{1}|{2}' -f $rr, $item.Length, $localMap[$rr].Sha
        $localBytes.AddRange([Text.Encoding]::UTF8.GetBytes($line + "`n"))
    }
}
foreach ($f in $SyncFiles) {
    $full = Join-Path $WorkspaceRoot $f
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { continue }
    if (-not (Test-IsBinaryByExtension -Path $full)) { [void](Get-Utf8Checked -Path $full) }
    $item = Get-Item -LiteralPath $full -Force
    $localMap[$f] = [pscustomobject]@{ Size = [int64]$item.Length; Sha = (Get-Sha256Hex -Path $full) }
    $line = '{0}|{1}|{2}' -f $f, $item.Length, $localMap[$f].Sha
    $localBytes.AddRange([Text.Encoding]::UTF8.GetBytes($line + "`n"))
}
[IO.File]::WriteAllBytes($localManifestPath, $localBytes.ToArray())
$localCount = $localMap.Count
Write-Info "本地清单：$localManifestPath（$localCount 项）"
Write-Info '清单按「相对路径|大小|sha256」排列，行尾固定 LF（避免 CRLF 污染比对）'

# ===========================================================================
# 6. 远端清单
# ===========================================================================
$remoteManifestPath = Join-Path $TempDir 'manifest_vm.txt'
$remoteMap = @{}
$remoteSummary = @()

# 远端脚本：读文件用 sha256sum，路径一律相对项目根，分隔符与本地一致
# 中文文件名按 UTF-8 原样输出（不经过任何本地码页），再由 PowerShell 用 UTF-8 解读。
$remoteStep4 = @'
set -uo pipefail
cd ~/REMOTE_DIR_PLACEHOLDER || { echo "CD_FAIL"; exit 9; }

HASHBIN=""
if command -v sha256sum >/dev/null 2>&1; then
  HASHBIN="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  HASHBIN="shasum -a 256"
else
  echo "NO_SHA256_TOOL"
  exit 8
fi

DIRS="legal_rag scripts tests knowledge docs sql data_pipeline"
FILES="requirements.txt requirements-full.txt README.md .env.example run.sh shutdown.sh .gitignore"

{
  for d in $DIRS; do
    if [ -d "$d" ]; then
      find "$d" -type f ! -path '*__pycache__*' ! -name '*.pyc' ! -name '*.pyo'
    fi
  done
  for f in $FILES; do
    [ -f "$f" ] && echo "$f"
  done
} | LC_ALL=C sort | while IFS= read -r p; do
  sz=$(stat -c '%s' "$p" 2>/dev/null || echo 0)
  hx=$($HASHBIN "$p" | cut -d' ' -f1)
  printf '%s|%s|%s\n' "$p" "$sz" "$hx"
done

echo "===DSH-MANIFEST==="
echo "SUMMARY hash_tool=$HASHBIN"
echo "SUMMARY python=$(.venv/bin/python -V 2>&1)"
echo "SUMMARY git_head=$(git -C . rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "SUMMARY cwd_has_legal_rag=$([ -d legal_rag ] && echo 1 || echo 0)"
'@

$remoteStep4 = $remoteStep4.Replace('REMOTE_DIR_PLACEHOLDER', $RemoteDir)
$step4Log = Join-Path $TempDir 'sync_manifest_raw.txt'
$step4Rc = Invoke-RemoteBash -ScriptText $remoteStep4 -LogPath $step4Log -Label 'manifest'

$rawOut = ''
if (Test-Path -LiteralPath $step4Log) { $rawOut = [IO.File]::ReadAllText($step4Log, [Text.Encoding]::UTF8) }
$rawOut = $rawOut -replace "`r`n", "`n"

if ($rawOut -match 'CD_FAIL') {
    Write-Err2 "远端 cd ~/$RemoteDir 失败：目录不存在。"
    Write-Info "修复建议：ssh $Target `"mkdir -p ~/$RemoteDir`" 后重跑本脚本。"
    exit 4
}
if ($rawOut -match 'NO_SHA256_TOOL') {
    Write-Err2 '远端既没有 sha256sum 也没有 shasum，无法做逐字节比对。'
    Write-Info '修复建议：sudo apt-get install -y coreutils（sha256sum 属 coreutils）；'
    Write-Info '          或把第 [4/4] 步改用 md5sum（本轮任务要求 sha256，故不自动降级）。'
    exit 6
}

$idx = $rawOut.IndexOf($ManifestMarker)
if ($idx -lt 0) {
    Write-Err2 "远端清单输出里没有分隔标记 $ManifestMarker（退出码 $step4Rc）"
    Write-Info "原始输出（末 25 行）在：$step4Log"
    $rawOut -split "`n" | Select-Object -Last 25 | ForEach-Object { Write-Host "        $_" -ForegroundColor DarkGray }
    exit 6
}

$manifestText = $rawOut.Substring(0, $idx)
$tailText = $rawOut.Substring($idx)
$tailText -split "`n" | Where-Object { $_ -match '^SUMMARY ' } | ForEach-Object {
    $remoteSummary += $_.Substring(8)
}
foreach ($s in $remoteSummary) { Write-Info "远端 $s" }

[IO.File]::WriteAllText($remoteManifestPath, $manifestText, (New-Object Text.UTF8Encoding($false)))

$parseErrs = 0
foreach ($ln in ($manifestText -split "`n")) {
    if ([string]::IsNullOrWhiteSpace($ln)) { continue }
    $p = $ln.Split('|')
    if ($p.Count -ne 3) { $parseErrs++; continue }
    $remoteMap[$p[0]] = [pscustomobject]@{ Size = [int64]$p[1]; Sha = $p[2].ToLowerInvariant() }
}
if ($parseErrs -gt 0) { Write-Warn2 "远端清单里有 $parseErrs 行格式不对，已跳过" }
$remoteCount = $remoteMap.Count
Write-Info "远端清单：$remoteManifestPath（$remoteCount 项）"

# 远端哈希工具若不是 sha256sum，说明可能不是 sha256 —— 明确警告而不是假装通过
foreach ($s in $remoteSummary) {
    if ($s -like 'hash_tool=*' -and $s -notlike 'hash_tool=sha256sum*') {
        Write-Warn2 "远端哈希工具是 $($s.Substring(10))，与本地 SHA256 不是同一算法，比对结果不可信！"
    }
}

# ===========================================================================
# 7. 比对 + 四类差异
# ===========================================================================
$identical = 0
$missingRemote = New-Object 'System.Collections.Generic.List[string]'
$hashDiff = New-Object 'System.Collections.Generic.List[string]'
$sizeDiff = New-Object 'System.Collections.Generic.List[string]'
$extraRemote = New-Object 'System.Collections.Generic.List[string]'

foreach ($k in ($localMap.Keys | Sort-Object)) {
    if (-not $remoteMap.ContainsKey($k)) { $missingRemote.Add($k); continue }
    $l = $localMap[$k]; $r = $remoteMap[$k]
    if ($l.Sha -eq $r.Sha) {
        $identical++
    } elseif ($l.Size -ne $r.Size) {
        $sizeDiff.Add(("{0}（本地 {1}B / 远端 {2}B）" -f $k, $l.Size, $r.Size))
    } else {
        $hashDiff.Add($k)
    }
}
foreach ($k in ($remoteMap.Keys | Sort-Object)) {
    if (-not $localMap.ContainsKey($k)) { $extraRemote.Add($k) }
}

# 「不同」= 大小不同 + 内容（sha256）不同
$different = $sizeDiff.Count + $hashDiff.Count
$verdict = ($missingRemote.Count -eq 0 -and $different -eq 0 -and $extraRemote.Count -eq 0)

# ===========================================================================
# 8. 报告
# ===========================================================================
$lines = New-Object 'System.Collections.Generic.List[string]'
function Add-Line {
    param([string]$Text)
    [void]$lines.Add($Text)
    switch -Regex ($Text) {
        '^===' { Write-Host $Text -ForegroundColor DarkGray }
        '^结论' { if ($verdict) { Write-Host $Text -ForegroundColor Green } else { Write-Host $Text -ForegroundColor Red } }
        '检查项数|一致|清单' { Write-Host $Text -ForegroundColor Gray }
        '^\s+\[缺失\]|^\s+\[不同\]|^\s+\[多余\]|^\s+\[本地独有\]' { Write-Host $Text -ForegroundColor Yellow }
        default { Write-Host $Text }
    }
}

Write-Host ''
Write-Host ('=' * 74) -ForegroundColor DarkGray
Write-Host ' 逐字节自检结果（sha256）' -ForegroundColor Cyan
Write-Host ('=' * 74) -ForegroundColor DarkGray
$now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Line '=== 同步 + 自检报告 ==='
Add-Line ("时间          : {0}" -f $now)
Add-Line ("目标          : {0}:{1}" -f $Target, $RemoteDir)
Add-Line ("工作区        : {0}" -f $WorkspaceRoot)
Add-Line ("本地清单      : {0}" -f $localManifestPath)
Add-Line ("远端清单      : {0}" -f $remoteManifestPath)
Add-Line ("本地项数      : {0}" -f $localCount)
Add-Line ("远端项数      : {0}" -f $remoteCount)
Add-Line ("远端缺少(本地有) : {0}" -f $missingRemote.Count)
Add-Line ("远端独有(远端多出来的) : {0}" -f $extraRemote.Count)
Add-Line ''
Add-Line '--- 四类差异计数（任务要求的口径）---'
Add-Line ("identical（两侧逐字节一致） : {0}" -f $identical)
Add-Line ("缺失（本地有、远端没有）    : {0}" -f $missingRemote.Count)
Add-Line ("不同（大小或 sha256 不一致）: {0}" -f $different)
Add-Line ("多余（远端有、本地没有）    : {0}" -f $extraRemote.Count)
Add-Line ''
Add-Line '--- 明细 ---'
Add-Line ("远端缺少文件（{0}）:" -f $missingRemote.Count)
foreach ($x in $missingRemote) { Add-Line ("  [缺失] {0}" -f $x) }
Add-Line ("内容不同（{0}）:" -f $hashDiff.Count)
foreach ($x in $hashDiff) { Add-Line ("  [不同] {0}（大小一致、sha256 不一致）" -f $x) }
Add-Line ("大小不同（{0}）:" -f $sizeDiff.Count)
foreach ($x in $sizeDiff) { Add-Line ("  [不同] {0}" -f $x) }
Add-Line ("远端多出（{0}）:" -f $extraRemote.Count)
foreach ($x in $extraRemote) { Add-Line ("  [多余] {0}" -f $x) }
Add-Line ''
Add-Line '--- 远端环境摘要 ---'
foreach ($s in $remoteSummary) { Add-Line ("  " + $s) }
Add-Line ''
if ($verdict) {
    Add-Line ("结论          : PASS —— 远端与工作区逐字节一致（{0}/{0}）" -f $identical)
} else {
    Add-Line '结论          : FAIL —— 两侧不一致，见上面明细'
    if ($missingRemote.Count -gt 0) { Add-Line '  处置：有文件没同步上去。确认它在同步范围内（7 个目录 / 7 个顶层文件），然后重跑本脚本。' }
    if ($different -gt 0) {
        Add-Line '  处置：同路径内容不同。最常见两种原因 ——'
        Add-Line '        a) 远端是上一轮的旧版（scp 覆盖失败/被跳过）：重跑本脚本第 [2/4] 步。'
        Add-Line '        b) 行尾差异（CRLF vs LF）或编码被改：用 git diff --stat 看本地文件，'
        Add-Line '           再对比远端 `file <路径>` / `head -c 200 <路径> | xxd | head`。'
    }
    if ($extraRemote.Count -gt 0) { Add-Line '  处置：scp -r 不会删远端多余文件（VM-DEPLOY.md §2.2）。确认多余项可以删，再远端 rm 掉；本脚本不自动删除。' }
}
Add-Line ("检查项数      : {0}" -f $localCount)

[IO.File]::WriteAllText($ReportPath, (($lines -join "`r`n") + "`r`n"), (New-Object Text.UTF8Encoding($false)))
Write-Host ''
Write-Host ("报告已写入：{0}" -f $ReportPath) -ForegroundColor Gray

if ($verdict) { exit 0 } else { exit 5 }
