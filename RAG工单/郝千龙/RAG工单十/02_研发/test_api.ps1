# -*- coding: utf-8 -*-
# 【接口冒烟测试脚本 · test_api.ps1】对本机或容器中的金融问答服务做健康检查与问答冒烟
# 工单编号：人工智能NLP-RAG-金融问答系统部署
#
# 用法（默认打 http://127.0.0.1:8000）：
#   powershell -ExecutionPolicy Bypass -File .\test_api.ps1
# 指定服务地址：
#   powershell -ExecutionPolicy Bypass -File .\test_api.ps1 -BaseUrl http://127.0.0.1:8000

param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
# 控制台与管道统一按 UTF-8 输出，避免中文重定向到文件时按 GBK 乱码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
Write-Host "==== 金融问答服务接口冒烟测试 -> $BaseUrl ====" -ForegroundColor Cyan

# ---------- 1. 健康检查 GET /health ----------
Write-Host "`n[1/4] GET /health 健康检查" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get -TimeoutSec 10
    Write-Host ("  status={0} documents={1} chunks={2} index_built={3} load={4}s" -f `
        $health.status, $health.documents, $health.chunks, $health.index_built, $health.index_load_seconds)
    if ($health.status -ne "ok") { throw "健康检查 status 非 ok" }
} catch {
    Write-Host "  健康检查失败：$_" -ForegroundColor Red
    exit 1
}

# ---------- 2. 根路径人工探活 GET / ----------
Write-Host "`n[2/4] GET / 服务说明页" -ForegroundColor Yellow
# -UseBasicParsing 避免 Windows PowerShell 5.1 依赖 IE 引擎解析 HTML 报错
$root = Invoke-WebRequest -Uri "$BaseUrl/" -Method Get -TimeoutSec 10 -UseBasicParsing
Write-Host "  HTTP $($root.StatusCode)，返回 $($root.Content.Length) 字符"

# ---------- 3/4. 三个金融问题 POST /ask ----------
$questions = @(
    "发行人的注册资本是多少？",
    "发行人的法定代表人是谁？",
    "公司主营业务包括哪些产品？"
)
$results = @()
$i = 2
foreach ($q in $questions) {
    $i++
    Write-Host "`n[$i/4] POST /ask 问题：$q" -ForegroundColor Yellow
    $body = @{ question = $q } | ConvertTo-Json -Compress
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "$BaseUrl/ask" -Method Post -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec 30
    $sw.Stop()
    Write-Host "  答案：$($resp.answer)" -ForegroundColor Green
    Write-Host ("  证据：第 {0} 页 | 标题：{1} | 接口耗时：{2} ms" -f `
        ($resp.evidences[0].page_no -join ","), $resp.evidences[0].title, $resp.latency_ms)
    Write-Host "  客户端实测往返：$($sw.ElapsedMilliseconds) ms"
    $results += $resp
}

Write-Host "`n==== 冒烟测试全部通过（健康检查 + 根路径 + 3 个金融问题）====" -ForegroundColor Cyan
