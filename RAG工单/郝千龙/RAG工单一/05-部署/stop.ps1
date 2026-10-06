# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【Windows停止脚本 · stop.ps1】根据 PID 文件停止 FastAPI 与 Streamlit
# 用法：./stop.ps1
# ======================================================================
$DeployDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $DeployDir "logs"

foreach ($name in @("api", "streamlit")) {
    $pidFile = Join-Path $LogDir "$name.pid"
    if (Test-Path $pidFile) {
        $procId = (Get-Content $pidFile | Select-Object -First 1).Trim()
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host "[STOP] $name (PID=$procId)" -ForegroundColor Yellow
        } catch {
            Write-Host "[SKIP] $name 进程 $procId 已退出" -ForegroundColor DarkGray
        }
        Remove-Item $pidFile -Force
    }
}

# 兜底：清理仍占用 8000/8501 端口的 python 进程
foreach ($port in @(8000, 8501)) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "[STOP] 释放端口 $port (PID=$($c.OwningProcess))" -ForegroundColor Yellow
    }
}
Write-Host "[OK] 全部服务已停止" -ForegroundColor Green
