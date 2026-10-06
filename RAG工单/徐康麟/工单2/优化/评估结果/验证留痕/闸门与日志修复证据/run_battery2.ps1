# t1 冻结态最终验证批（临时取证脚本，不入交付物树）
$ErrorActionPreference = 'Continue'
$py = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
$out = 'E:\gao6gongdan\工单2\.tmp_review'

Write-Host "=== [1/6] 自测：抽取式引擎（官方 12 条 + 扩展 6 条 × 3 轮） ==="
pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_answerability.py 2>&1 | Out-File -Encoding UTF8 "$out\c_selftest_extractive.txt"
Get-Content -Encoding UTF8 "$out\c_selftest_extractive.txt" | Select-String -Pattern '引擎|第 \d 次：|中文 10 题|多轮作答|三次拒答结果|结论' | ForEach-Object { $_.Line }

Write-Host "=== [2/6] 自测：生产引擎（与服务端同构） ==="
pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_answerability.py --engine production 2>&1 | Out-File -Encoding UTF8 "$out\c_selftest_production.txt"
Get-Content -Encoding UTF8 "$out\c_selftest_production.txt" | Select-String -Pattern '引擎|第 \d 次：|中文 10 题|多轮作答|三次拒答结果|结论' | ForEach-Object { $_.Line }

Write-Host "=== [3/6] 离线套件（核心 62 项） ==="
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线 -q --ignore=测试/离线/test_deliverable_integrity.py 2>&1 | Out-File -Encoding UTF8 "$out\c_offline_core.txt"
Get-Content -Encoding UTF8 "$out\c_offline_core.txt" | Select-Object -Last 3

Write-Host "=== [4/6] 在线·容错套件（验收 2 靶子） ==="
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/在线/test_fault_tolerance.py -v 2>&1 | Out-File -Encoding UTF8 "$out\c_online_ft.txt"
Get-Content -Encoding UTF8 "$out\c_online_ft.txt" | Select-Object -Last 5

Write-Host "=== [5/6] 真实 HTTP 服务：官方 12 条 × 3 连跑（逐题留痕） ==="
$proc = Start-Process -FilePath $py -ArgumentList '研发/app/ui/serve_fallback.py', '--host', '127.0.0.1', '--port', '8124' `
    -WorkingDirectory 'E:\gao6gongdan\工单2' -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$out\c_server_out.txt" -RedirectStandardError "$out\c_server_err.txt"
$ready = $false
for ($i = 0; $i -lt 240; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri 'http://127.0.0.1:8124/api/health' -TimeoutSec 5 -UseBasicParsing
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    Start-Sleep -Milliseconds 500
}
Write-Host "  服务就绪=$ready（pid=$($proc.Id)）"
if ($ready) {
    & $py "$out\online_3runs.py" 'http://127.0.0.1:8124' 2>&1 | Out-File -Encoding UTF8 "$out\c_online_3runs.txt"
    Get-Content -Encoding UTF8 "$out\c_online_3runs.txt" | Select-String -Pattern '第 \d 次：|三次合计' | ForEach-Object { $_.Line }
}
if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }

Write-Host "=== [6/6] 在线套件（全量 32 项） ==="
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/在线 -v 2>&1 | Out-File -Encoding UTF8 "$out\c_online_full.txt"
Get-Content -Encoding UTF8 "$out\c_online_full.txt" | Select-Object -Last 6

Write-Host "=== 最终验证批结束 ==="
