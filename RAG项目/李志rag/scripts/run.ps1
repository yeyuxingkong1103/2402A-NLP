$ErrorActionPreference = "Stop"
docker compose up -d --wait --wait-timeout 180
$apiProcess = Start-Process -WindowStyle Hidden -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -RedirectStandardOutput ".\api-output.log" -RedirectStandardError ".\api-error.log" -PassThru
$apiReady = $false
foreach ($attempt in 1..30) {
    Start-Sleep -Seconds 1
    try {
        Invoke-RestMethod "http://127.0.0.1:8000/health" -TimeoutSec 2 | Out-Null
        $apiReady = $true
        break
    } catch {
        if ($apiProcess.HasExited) { break }
    }
}
if (-not $apiReady) {
    Write-Host "API 启动失败，错误日志：" -ForegroundColor Red
    Get-Content ".\api-error.log" -Tail 80 -ErrorAction SilentlyContinue
    exit 1
}
$uiProcess = Start-Process -WindowStyle Hidden -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "streamlit", "run", "web_ui.py", "--server.port", "8501" -PassThru
$apiProcess.Id | Set-Content .api.pid
$uiProcess.Id | Set-Content .ui.pid
Write-Host "API: http://127.0.0.1:8000/docs"
Write-Host "UI:  http://127.0.0.1:8501"
