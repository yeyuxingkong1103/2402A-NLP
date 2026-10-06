# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【Windows启动脚本 · start.ps1】启动 FastAPI(8000) + Streamlit(8501)，无需 Docker
# 用法：在 05-部署 目录下执行  ./start.ps1
# ======================================================================
$ErrorActionPreference = "Stop"

$DeployDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Split-Path -Parent $DeployDir
$CodeDir   = Join-Path $Root "02-研发"
$LogDir    = Join-Path $DeployDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ---------- 运行环境变量 ----------
$env:PYTHONDONTWRITEBYTECODE = "1"   # 避免用户级 site-packages 写 pyc 触发权限问题
$env:HF_HUB_OFFLINE           = "1"   # 强制使用本地模型缓存
$env:TOKENIZERS_PARALLELISM   = "false"

function Test-Port($p) {
    $c = New-Object Net.Sockets.TcpClient
    try { $c.Connect("127.0.0.1", $p); return $true }
    catch { return $false }
    finally { $c.Close() }
}

# ---------- 1. 启动 FastAPI ----------
if (Test-Port 8000) {
    Write-Host "[SKIP] FastAPI 端口 8000 已被占用" -ForegroundColor Yellow
} else {
    Write-Host "[START] FastAPI :8000 ..." -ForegroundColor Green
    $api = Start-Process python -ArgumentList '-m','uvicorn','main:app','--host','0.0.0.0','--port','8000' `
        -WorkingDirectory $CodeDir -PassThru -WindowStyle Minimized `
        -RedirectStandardOutput (Join-Path $LogDir "api.out.log") `
        -RedirectStandardError  (Join-Path $LogDir "api.err.log")
    $api.Id | Out-File (Join-Path $LogDir "api.pid") -Encoding ascii

    # 等待健康检查
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        try {
            $h = Invoke-RestMethod "http://127.0.0.1:8000/api/health" -TimeoutSec 2
            if ($h.data.status -eq "healthy") { break }
        } catch {}
    }
    Write-Host "[OK] FastAPI 就绪" -ForegroundColor Green
}

# ---------- 2. 启动 Streamlit ----------
if (Test-Port 8501) {
    Write-Host "[SKIP] Streamlit 端口 8501 已被占用" -ForegroundColor Yellow
} else {
    Write-Host "[START] Streamlit :8501 ..." -ForegroundColor Green
    $st = Start-Process python -ArgumentList '-m','streamlit','run','app.py','--server.port','8501','--server.address','0.0.0.0' `
        -WorkingDirectory $CodeDir -PassThru -WindowStyle Minimized `
        -RedirectStandardOutput (Join-Path $LogDir "streamlit.out.log") `
        -RedirectStandardError  (Join-Path $LogDir "streamlit.err.log")
    $st.Id | Out-File (Join-Path $LogDir "streamlit.pid") -Encoding ascii
    Start-Sleep -Seconds 4
    Write-Host "[OK] Streamlit 就绪" -ForegroundColor Green
}

# ---------- 3. 打开浏览器 ----------
Start-Process "http://127.0.0.1:8501"
Write-Host ""
Write-Host "服务地址：" -ForegroundColor Cyan
Write-Host "  Streamlit : http://127.0.0.1:8501"
Write-Host "  FastAPI   : http://127.0.0.1:8000/api/health"
Write-Host "  API 文档  : http://127.0.0.1:8000/docs"
Write-Host "  日志目录  : $LogDir"
Write-Host "停止服务请执行：./stop.ps1" -ForegroundColor Cyan
