# t6 复核批（t6 与 t1 同一交付物；本轮只复核留证，不改实现）
$ErrorActionPreference = 'Continue'
$py = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
$out = 'E:\gao6gongdan\工单2\.tmp_review'

Write-Host "=== [1/5] pytest 测试/在线/test_fault_tolerance.py -v ==="
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/在线/test_fault_tolerance.py -v 2>&1 | Out-File -Encoding UTF8 "$out\t6_online_ft.txt"
Get-Content -Encoding UTF8 "$out\t6_online_ft.txt" | Select-Object -Last 4

Write-Host "=== [2/5] pytest 测试/在线 -v ==="
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/在线 -v 2>&1 | Out-File -Encoding UTF8 "$out\t6_online_full.txt"
Get-Content -Encoding UTF8 "$out\t6_online_full.txt" | Select-Object -Last 6

Write-Host "=== [3/5] pytest 测试/离线 -q ==="
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线 -q 2>&1 | Out-File -Encoding UTF8 "$out\t6_offline.txt"
Get-Content -Encoding UTF8 "$out\t6_offline.txt" | Select-Object -Last 3

Write-Host "=== [4/5] 稳定性自测：官方 12 条 + 扩展同类 6 条 × 3 轮（与服务端同构引擎） ==="
pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_answerability.py --engine production 2>&1 | Out-File -Encoding UTF8 "$out\t6_selftest_production.txt"
Get-Content -Encoding UTF8 "$out\t6_selftest_production.txt" | Select-String -Pattern '引擎|第 \d 次：|中文 10 题|多轮作答|三次拒答结果|结论' | ForEach-Object { $_.Line }

Write-Host "=== [5/5] 真实 HTTP 服务：官方 12 条 × 3 连跑 ==="
$proc = Start-Process -FilePath $py -ArgumentList '研发/app/ui/serve_fallback.py', '--host', '127.0.0.1', '--port', '8125' `
    -WorkingDirectory 'E:\gao6gongdan\工单2' -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$out\t6_server_out.txt" -RedirectStandardError "$out\t6_server_err.txt"
$ready = $false
for ($i = 0; $i -lt 240; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri 'http://127.0.0.1:8125/api/health' -TimeoutSec 5 -UseBasicParsing
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    Start-Sleep -Milliseconds 500
}
Write-Host "  服务就绪=$ready（pid=$($proc.Id)）"
if ($ready) {
    & $py "$out\online_3runs.py" 'http://127.0.0.1:8125' 2>&1 | Out-File -Encoding UTF8 "$out\t6_online_3runs.txt"
    Get-Content -Encoding UTF8 "$out\t6_online_3runs.txt" | Select-String -Pattern '第 \d 次：|三次合计' | ForEach-Object { $_.Line }
}
if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }

Write-Host "=== t6 复核批结束 ==="
