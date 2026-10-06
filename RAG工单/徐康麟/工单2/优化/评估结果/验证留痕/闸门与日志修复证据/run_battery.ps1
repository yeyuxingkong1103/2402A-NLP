# t1 冻结态验证批（临时取证脚本，不入交付物树）
# 顺序执行、全程单写者：避免与 tester/verifier 争用 Ollama 与日志文件。
$ErrorActionPreference = 'Continue'
$py = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
$out = 'E:\gao6gongdan\工单2\.tmp_review'

Write-Host "=== [1/8] 离线套件（全量，含 designer 的交付完整性守卫） ==="
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线 -q 2>&1 | Out-File -Encoding UTF8 "$out\b_offline_full.txt"
Get-Content -Encoding UTF8 "$out\b_offline_full.txt" | Select-Object -Last 8

Write-Host "=== [2/8] 离线套件（核心 62 项，排除交付完整性守卫） ==="
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线 -q --ignore=测试/离线/test_deliverable_integrity.py 2>&1 | Out-File -Encoding UTF8 "$out\b_offline_core.txt"
Get-Content -Encoding UTF8 "$out\b_offline_core.txt" | Select-Object -Last 4

Write-Host "=== [3/8] 在线·容错套件（验收 2 靶子） ==="
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/在线/test_fault_tolerance.py -v 2>&1 | Out-File -Encoding UTF8 "$out\b_online_ft.txt"
Get-Content -Encoding UTF8 "$out\b_online_ft.txt" | Select-Object -Last 6

Write-Host "=== [4/8] 在线套件（全量 32 项） ==="
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/在线 -v 2>&1 | Out-File -Encoding UTF8 "$out\b_online_full.txt"
Get-Content -Encoding UTF8 "$out\b_online_full.txt" | Select-Object -Last 8

Write-Host "=== [5/8] 自测：抽取式引擎 3 连跑 ==="
pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_answerability.py 2>&1 | Out-File -Encoding UTF8 "$out\b_selftest_extractive.txt"
Get-Content -Encoding UTF8 "$out\b_selftest_extractive.txt" | Select-String -Pattern '引擎|第 \d 次：|中文 10 题|多轮作答|三次拒答|结论' | ForEach-Object { $_.Line }

Write-Host "=== [6/8] 自测：生产引擎（与服务端同构）3 连跑 ==="
pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_answerability.py --engine production 2>&1 | Out-File -Encoding UTF8 "$out\b_selftest_production.txt"
Get-Content -Encoding UTF8 "$out\b_selftest_production.txt" | Select-String -Pattern '引擎|第 \d 次：|中文 10 题|多轮作答|三次拒答|结论' | ForEach-Object { $_.Line }

Write-Host "=== [7/8] 真实 HTTP 服务：12 条无关问题 3 连跑（逐题留痕） ==="
$proc = Start-Process -FilePath $py -ArgumentList '研发/app/ui/serve_fallback.py', '--host', '127.0.0.1', '--port', '8123' `
    -WorkingDirectory 'E:\gao6gongdan\工单2' -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$out\b_server_out.txt" -RedirectStandardError "$out\b_server_err.txt"
$ready = $false
for ($i = 0; $i -lt 240; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri 'http://127.0.0.1:8123/api/health' -TimeoutSec 5 -UseBasicParsing
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    Start-Sleep -Milliseconds 500
}
Write-Host "  服务就绪=$ready（pid=$($proc.Id)）"
if ($ready) {
    & $py "$out\online_3runs.py" 'http://127.0.0.1:8123' 2>&1 | Out-File -Encoding UTF8 "$out\b_online_3runs.txt"
    Get-Content -Encoding UTF8 "$out\b_online_3runs.txt" | Select-Object -Last 6
}
if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }

Write-Host "=== [8/8] 闸门逐题诊断（含多轮与信号验证） ==="
pwsh -NoProfile -File run_py.ps1 研发/scripts/diag_answerability_gate.py 2>&1 | Out-File -Encoding UTF8 "$out\b_diag.txt"
Get-Content -Encoding UTF8 "$out\b_diag.txt" | Select-String -Pattern '汇总|无关集|误拒|多轮追问误拒|信号功能|结论' | ForEach-Object { $_.Line }

Write-Host "=== 批处理结束 ==="
