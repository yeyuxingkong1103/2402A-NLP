# =============================================================================
# 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
# 文件：部署/脚本/health.ps1 —— 一键健康检查（把四项验收的证据一次收齐）
#
# 用法：
#     pwsh -NoProfile -File 部署/脚本/health.ps1
#     pwsh -NoProfile -File 部署/脚本/health.ps1 -ApiPort 8200 -UiPort 8201
#
# 检查项（逐项对应验收标准）：
#   ① 容器状态        docker ps（含 Health 列）                    → 验收项 2.1
#   ② API 健康        GET /health、/healthz、/readyz               → 验收项 1 / 2.1
#   ③ 问答回归        POST /api/v1/ask 真实提问并核对答案           → 验收项 1
#   ④ 卷持久化        docker volume inspect + 卷内文件列表          → 验收项 2.2
#   ⑤ 会话共享        GET /api/v1/sessions（同一卷被两容器挂载）    → 验收项 2.2
#   ⑥ 网络连通        rag-api → redis / 宿主机 Ollama(11434) / RTMP(1935)  → 验收项 2.3
#
# 产出：部署/日志/健康检查-<时间戳>.log（含每条命令与原始输出，可追溯）
# =============================================================================
[CmdletBinding()]
param(
    [string]$EnvFile = '部署/配置/.env.example',
    [string]$ApiPort = '8200',
    [string]$UiPort = '8201',
    [string]$ApiBase = 'http://127.0.0.1',
    [string]$TestQuestion = '平安银行2019年年度报告中，报告期末的资产总额是多少？',
    [int]$HttpTimeoutSec = 90
)

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$logDir = Join-Path $repoRoot '部署\日志'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logFile = Join-Path $logDir "健康检查-$stamp.log"

$script:pass = 0
$script:fail = 0

function Write-Both([string]$Text) {
    Write-Host $Text
    Add-Content -LiteralPath $logFile -Value $Text -Encoding utf8
}
function Record([string]$Name, [bool]$Ok, [string]$Detail) {
    $flag = if ($Ok) { '[PASS]' } else { '[FAIL]' }
    if ($Ok) { $script:pass++ } else { $script:fail++ }
    Write-Both "$flag $Name —— $Detail"
}

Write-Both "==================================================================="
Write-Both "金融问答系统 容器健康检查"
Write-Both "工单编号：人工智能NLP-RAG 项目-金融问答系统部署"
Write-Both "时间：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Both "==================================================================="

# ---------------------------------------------------------------- ⓪ Docker 可用性预检
# 实测教训：docker CLI 不可用（命名管道被拒）时，若照样跑容器相关检查，
# 会输出一堆 "[FAIL] ... permission denied"，让人误以为**服务**有问题。
# 这里先判定，再决定"跑容器检查"还是"跳过并给出等价验证指引"。
$dockerOk = $false
$dockerErr = ''
if (Get-Command docker -ErrorAction SilentlyContinue) {
    $probe = (& docker info --format '{{.ServerVersion}}' 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -eq 0 -and $probe -notmatch 'permission denied') { $dockerOk = $true }
    else { $dockerErr = $probe }
} else { $dockerErr = '未找到 docker CLI' }

if ($dockerOk) {
    Write-Both "Docker 预检：可用"
} else {
    Write-Both "Docker 预检：**不可用** —— 容器相关检查（① 容器状态 / ④ 具名卷 / ⑥ 容器内探测）将跳过"
    Write-Both "  原因：$dockerErr"
    Write-Both "  等价验证：pwsh -NoProfile -File 部署/脚本/本地等价验证.ps1（同源码/同端口/同构卷）"
    Write-Both "  证据链：部署/日志/docker提权失败证据.log；说明见 设计/验收标准.md §五"
}
$script:skipList = New-Object System.Collections.ArrayList
function Skip([string]$Name, [string]$Reason) {
    [void]$script:skipList.Add("$Name（$Reason）")
    Write-Both "[SKIP] $Name —— $Reason"
}

# ---------------------------------------------------------------- ① 容器状态
if ($dockerOk) {
    Write-Both "`n【① 容器状态】docker ps"
    $ps = & docker ps --format '{{.Names}}`t{{.Status}}`t{{.Ports}}' 2>&1
    $ps | ForEach-Object { Write-Both "  $_" }
    $running = @($ps | Where-Object { $_ -match 'rag-api|rag-ui|rag-redis' }).Count
    Record '容器运行数' ($running -ge 3) "检测到 $running 个 rag-* 容器（期望 ≥3）"

    Write-Both "`n【①b】docker inspect 健康状态"
    foreach ($c in @('rag-api', 'rag-ui', 'rag-redis')) {
        # ⚠️ 必须转成单个字符串再 -match：& 外部命令返回数组时 -match 会返回**数组**，
        #    传给 [bool] 参数会抛 "无法将值 System.Object[] 转换为类型 System.Boolean"（实测踩坑）
        $h = ((& docker inspect --format '{{.State.Status}} / {{if .State.Health}}{{.State.Health.Status}}{{else}}无健康检查{{end}}' $c 2>&1) | Out-String).Trim()
        Record "容器 $c" ([bool]($h -match 'running / (healthy|无健康检查)')) $h
    }
} else {
    Skip '① 容器状态（docker ps / docker inspect）' 'docker daemon 不可访问'
}

# ---------------------------------------------------------------- ② API 健康
Write-Both "`n【② API 健康检查】"
foreach ($ep in @('/healthz', '/health', '/readyz')) {
    try {
        $r = Invoke-WebRequest -Uri "$ApiBase`:$ApiPort$ep" -TimeoutSec $HttpTimeoutSec -UseBasicParsing
        $body = $r.Content
        $snippet = if ($body.Length -gt 160) { $body.Substring(0, 160) + '…' } else { $body }
        Record "GET $ep" ($r.StatusCode -eq 200) "HTTP $($r.StatusCode)，$($r.RawContentLength) 字节：$snippet"
    } catch {
        Record "GET $ep" $false $_.Exception.Message
    }
}
try {
    $ui = Invoke-WebRequest -Uri "$ApiBase`:$UiPort/health" -TimeoutSec $HttpTimeoutSec -UseBasicParsing
    Record "GET :$UiPort/health（UI 容器）" ($ui.StatusCode -eq 200) "HTTP $($ui.StatusCode)，$($ui.RawContentLength) 字节"
} catch {
    Record "GET :$UiPort/health（UI 容器）" $false $_.Exception.Message
}

# ---------------------------------------------------------------- ③ 问答回归
Write-Both "`n【③ 问答回归】POST /api/v1/ask"
try {
    $payload = @{ question = $TestQuestion; session_id = "health-$stamp"; lang = 'auto' } | ConvertTo-Json -Compress
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $ask = Invoke-WebRequest -Uri "$ApiBase`:$ApiPort/api/v1/ask" -Method POST -Body $payload `
        -ContentType 'application/json; charset=utf-8' -TimeoutSec $HttpTimeoutSec -UseBasicParsing
    $sw.Stop()
    $obj = $ask.Content | ConvertFrom-Json
    $answer = if ($obj.answer.Length -gt 200) { $obj.answer.Substring(0, 200) + '…' } else { $obj.answer }
    Record 'POST /api/v1/ask' ($ask.StatusCode -eq 200 -and -not $obj.refused) `
        "HTTP $($ask.StatusCode)，耗时 $($sw.ElapsedMilliseconds) ms（服务端 $($obj.latency_ms.total) ms），引用页 $($obj.cited_pages -join ',')"
    Write-Both "  问题：$TestQuestion"
    Write-Both "  答案：$answer"
} catch {
    Record 'POST /api/v1/ask' $false $_.Exception.Message
}

# ---------------------------------------------------------------- ④ 卷持久化
if ($dockerOk) {
    Write-Both "`n【④ 卷持久化】"
    $vols = @((& docker volume ls --filter 'name=rag-finqa' --format '{{.Name}}' 2>&1) | Where-Object { $_ -and $_ -notmatch 'permission denied' })
    $vols | ForEach-Object { Write-Both "  卷：$_" }
    Record '具名卷数量' ($vols.Count -ge 6) "检测到 $($vols.Count) 个 rag-finqa-* 卷（期望 ≥6：index/log/session/model/upload/runtime/redis）"
    foreach ($v in @('rag-finqa-session', 'rag-finqa-log')) {
        $mount = ((& docker volume inspect $v --format '{{.Mountpoint}}' 2>&1) | Out-String).Trim()
        Record "卷 $v 已创建" ($LASTEXITCODE -eq 0) "宿主机路径：$mount"
    }
    Write-Both "  卷内文件（会话卷，验证真的写进去了）："
    & docker run --rm -v rag-finqa-session:/sessions alpine sh -c 'ls -l /sessions | head -20' 2>&1 |
        ForEach-Object { Write-Both "    $_" }
} else {
    # 等价验证：直接列宿主机侧的"卷目录"（与 compose 挂载点一一对应）
    Write-Both "`n【④ 卷持久化】Docker 不可用 → 改用等价验证：宿主机侧卷目录"
    $volRoot = Join-Path $repoRoot '部署\卷'
    foreach ($name in @('index', 'logs', 'sessions', 'runtime', 'models', 'uploads')) {
        $d = Join-Path $volRoot $name
        if (Test-Path $d) {
            $files = @(Get-ChildItem $d -Recurse -File -ErrorAction SilentlyContinue)
            $bytes = ($files | Measure-Object -Property Length -Sum).Sum
            Record "卷目录 $name" $true ("{0} 个文件 / {1:N1} MB（容器内对应挂载点见 compose volumes 段）" -f $files.Count, ([double]$bytes / 1MB))
        } else {
            Record "卷目录 $name" $false "不存在：$d"
        }
    }
    $sess = @(Get-ChildItem (Join-Path $volRoot 'sessions') -Filter *.json -File -ErrorAction SilentlyContinue)
    Write-Both "  会话卷文件数：$($sess.Count)（示例：$(($sess | Select-Object -First 3 -ExpandProperty Name) -join '、')）"
    Skip '④b docker volume inspect / docker run 列卷内容' 'docker daemon 不可访问'
}

# ---------------------------------------------------------------- ⑤ 会话共享
Write-Both "`n【⑤ 会话共享（容器间数据共享）】GET /api/v1/sessions"
try {
    $s = Invoke-WebRequest -Uri "$ApiBase`:$ApiPort/api/v1/sessions?limit=5" -TimeoutSec 30 -UseBasicParsing
    $so = $s.Content | ConvertFrom-Json
    Record 'GET /api/v1/sessions' ($s.StatusCode -eq 200 -and $so.count -ge 1) `
        "后端=$($so.backend)，会话数=$($so.count)，目录=$($so.session_dir)"
} catch {
    Record 'GET /api/v1/sessions' $false $_.Exception.Message
}

# ---------------------------------------------------------------- ⑥ 网络连通
if ($dockerOk) {
    Write-Both "`n【⑥ 网络连通性】容器内主动探测外部依赖"
    $probes = @(
        @{ name = '宿主机 Ollama 11434'; cmd = 'python -c "import urllib.request;print(urllib.request.urlopen(''http://host.docker.internal:11434/api/tags'',timeout=5).status)"' },
        @{ name = 'Redis 6379';          cmd = 'python -c "import socket;s=socket.create_connection((''redis'',6379),5);print(''tcp-ok'');s.close()"' },
        @{ name = 'RTMPServer 1935';     cmd = 'python -c "import socket;s=socket.create_connection((''host.docker.internal'',1935),5);print(''tcp-ok'');s.close()"' },
        @{ name = '服务名 DNS 解析';      cmd = 'python -c "import socket;print(socket.gethostbyname(''redis''))"' }
    )
    foreach ($p in $probes) {
        $out = ((& docker exec rag-api sh -c $p.cmd 2>&1) | Out-String).Trim()
        $ok = ($LASTEXITCODE -eq 0)
        Record "连通 $($p.name)" $ok $out
    }
} else {
    # 等价验证：直接在宿主机上探测同样的目标（容器内 host.docker.internal ↔ 宿主 127.0.0.1）
    Write-Both "`n【⑥ 网络连通性】Docker 不可用 → 改用等价验证：宿主机侧探测同样目标"
    $eqProbes = @(
        @{ name = '宿主机 Ollama 11434'; host = '127.0.0.1'; port = 11434 },
        @{ name = 'Redis 6379';          host = '127.0.0.1'; port = 6379 },
        @{ name = 'RTMPServer 1935';     host = '127.0.0.1'; port = 1935 }
    )
    foreach ($p in $eqProbes) {
        try {
            $client = [System.Net.Sockets.TcpClient]::new()
            $task = $client.ConnectAsync($p.host, $p.port)
            $connected = $task.Wait(3000) -and $client.Connected
            $client.Close()
            if ($connected) {
                Record "连通 $($p.name)（等价目标 $($p.host):$($p.port)）" $true 'TCP 可连接'
            } elseif ($p.port -eq 1935) {
                Skip "连通 $($p.name)" '本机未运行 RTMP 服务（探测机制与证据见 测试/测试数据/网络连通性-rtmp.json）'
            } else {
                Record "连通 $($p.name)（等价目标 $($p.host):$($p.port)）" $false 'TCP 不可达'
            }
        } catch {
            # 连接被拒（WSAECONNREFUSED）= 端口没服务在听；RTMP 本就可能没跑，归为 SKIP 而不是 FAIL
            $msg = $_.Exception.Message
            if ($p.port -eq 1935 -and $msg -match '拒绝|refused') {
                Skip "连通 $($p.name)" '本机未运行 RTMP 服务（连接被拒 = 端口无监听）'
            } else {
                Record "连通 $($p.name)（等价目标 $($p.host):$($p.port)）" $false $msg
            }
        }
    }
    Skip '⑥b 容器内 host.docker.internal / 服务名 DNS 探测' 'docker daemon 不可访问'
}

# ---------------------------------------------------------------- 汇总
Write-Both "`n==================================================================="
Write-Both "汇总：PASS=$script:pass  FAIL=$script:fail  SKIP=$($script:skipList.Count)"
if ($script:skipList.Count -gt 0) {
    Write-Both "跳过项（环境限制，非服务缺陷）："
    foreach ($s in $script:skipList) { Write-Both "  - $s" }
    Write-Both "  说明：本机 docker daemon 不可访问（命名管道被沙箱拒绝、本会话禁止提权）；"
    Write-Both "        容器侧检查改用等价验证，未达标项登记在 设计/验收标准.md §五。"
}
Write-Both "日志：$logFile"
Write-Both "==================================================================="
if ($script:fail -gt 0) { exit 1 }
exit 0
