# 一键部署 / 运行脚本
# 工单编号: 人工智能 NLP-RAG-Query 理解优化任务
# 用法: powershell -ExecutionPolicy Bypass -File run_all.ps1
#      追加 -SkipBuild 可跳过知识库重建（已构建时推荐）
param([switch]$SkipBuild)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dev  = Join-Path $root "研发"
Set-Location $dev
$env:PYTHONIOENCODING = "utf-8"

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " 招股说明书多轮检索问答系统 · 一键运行" -ForegroundColor Cyan
Write-Host " 工单编号: 人工智能 NLP-RAG-Query 理解优化任务" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

Write-Host "`n[1/6] 检查 Ollama 服务 ..." -ForegroundColor Yellow
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    Write-Host "  Ollama 在线，模型: $($tags.models.name -join ', ')" -ForegroundColor Green
} catch {
    Write-Host "  警告: Ollama 未就绪，请先执行 'ollama serve'" -ForegroundColor Red
}

$kbOk = (Test-Path "vector_db\chunks.jsonl") -and (Test-Path "vector_db\embeddings.npy")
if ($SkipBuild -or $kbOk) {
    Write-Host "`n[2/6] 知识库已存在，跳过重建（如需重建请去掉 -SkipBuild 并删除 vector_db）" -ForegroundColor Green
} else {
    Write-Host "`n[2/6] 构建知识库（解析 → 切片 → 向量化，约 8 分钟）..." -ForegroundColor Yellow
    python build_kb.py --rebuild
}

Write-Host "`n[3/6] 命令行多轮演示 ..." -ForegroundColor Yellow
python run_demo.py

Write-Host "`n[4/6] 功能 / 容错 / 稳定性测试 ..." -ForegroundColor Yellow
python tools\run_functional_tests.py

Write-Host "`n[5/6] 评测（准确率 / 响应时间）..." -ForegroundColor Yellow
python evaluate_multiturn.py

Write-Host "`n[6/6] 启动 Streamlit 界面 ..." -ForegroundColor Yellow
Write-Host "  浏览器访问 http://localhost:8501   （演示回放: /?demo=1）" -ForegroundColor Green
streamlit run app.py
