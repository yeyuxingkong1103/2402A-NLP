param(
    [ValidateRange(1, 50)][int]$Threads = 2,
    [ValidateRange(1, 20)][int]$Loops = 1,
    [ValidateRange(0, 300)][int]$RampSeconds = 2
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

try {
    Invoke-RestMethod "http://127.0.0.1:8000/health" -TimeoutSec 5 | Out-Null
} catch {
    throw "RAG 后端未启动，请先执行 .\scripts\run.ps1"
}

$envValues = @{}
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') { $envValues[$matches[1]] = $matches[2].Trim() }
}
$loginBody = @{
    username = $envValues["ADMIN_USERNAME"]
    password = $envValues["ADMIN_PASSWORD"]
} | ConvertTo-Json
$login = Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/auth/login" `
    -ContentType "application/json" -Body $loginBody -TimeoutSec 10

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$relativeResult = "output/jmeter/$stamp/results.jtl"
$relativeReport = "output/jmeter/$stamp/report"
New-Item -ItemType Directory -Force -Path (Split-Path $relativeResult) | Out-Null

$jmeter = "D:\model_JMeter\apache-jmeter-5.6.3\bin\jmeter.bat"
$javaHome = "D:\python\PyCharm 2026.1.1\jbr"
if (-not (Test-Path $jmeter)) { throw "JMeter 不存在：$jmeter" }
if (-not (Test-Path "$javaHome\bin\java.exe")) { throw "Java 不存在：$javaHome" }
$env:JAVA_HOME = $javaHome
$env:Path = "$javaHome\bin;$env:Path"

Write-Host "开始压力测试：并发=$Threads，每用户请求=$Loops，升压=$RampSeconds 秒"
& $jmeter `
    -n -t (Resolve-Path "tests/jmeter_chat.jmx").Path `
    -l (Join-Path $projectRoot $relativeResult) `
    -e -o (Join-Path $projectRoot $relativeReport) `
    "-JRAG_TOKEN=$($login.access_token)" `
    "-JRAG_HOST=127.0.0.1" `
    "-JRAG_THREADS=$Threads" `
    "-JRAG_LOOPS=$Loops" `
    "-JRAG_RAMP=$RampSeconds"
if ($LASTEXITCODE -ne 0) { throw "JMeter 执行失败，退出码：$LASTEXITCODE" }

$rows = Import-Csv $relativeResult
$elapsed = @($rows | ForEach-Object { [double]$_.elapsed } | Sort-Object)
$errors = @($rows | Where-Object { $_.success -ne "true" }).Count
$p95Index = [Math]::Max(0, [Math]::Ceiling($elapsed.Count * 0.95) - 1)
$summary = [ordered]@{
    requests = $rows.Count
    successes = $rows.Count - $errors
    errors = $errors
    error_rate_percent = [Math]::Round(100 * $errors / [Math]::Max(1, $rows.Count), 2)
    average_ms = [Math]::Round(($elapsed | Measure-Object -Average).Average, 0)
    p95_ms = if ($elapsed.Count) { $elapsed[$p95Index] } else { 0 }
    maximum_ms = if ($elapsed.Count) { $elapsed[-1] } else { 0 }
    report = (Resolve-Path $relativeReport).Path + "\index.html"
}
$summary | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path (Split-Path $relativeResult) "summary.json")
$summary | Format-List
