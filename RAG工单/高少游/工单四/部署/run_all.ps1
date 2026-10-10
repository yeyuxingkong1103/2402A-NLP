# -*- coding: utf-8 -*-
# 一键部署/运行脚本（图像内容解析及检索优化版）
# 工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File 部署\run_all.ps1            # 检查环境 + 建库 + 评测
#   powershell -ExecutionPolicy Bypass -File 部署\run_all.ps1 -Serve     # 建库后直接启动界面

param(
    [switch]$Serve,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Proj = Join-Path $Root "研发"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " 招股说明书图像内容解析问答系统 · 一键部署" -ForegroundColor Cyan
Write-Host " 工单编号: 人工智能 NLP-RAG-图像内容解析及检索优化" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ---- 1. 定位 Python（优先 langchain2 环境） ----
$Py = "H:\an\envs\langchain2\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
Write-Host "[1/4] Python: $Py" -ForegroundColor Green
& $Py -c "import sys; print('     ', sys.version)"

# ---- 2. 检查依赖 ----
Write-Host "[2/4] 检查依赖..." -ForegroundColor Green
& $Py -c "import torch, transformers, faiss, rank_bm25, pymupdf, streamlit, jieba; print('      依赖检查通过')"

# ---- 3. 检查 Ollama ----
Write-Host "[3/4] 检查 Ollama 服务..." -ForegroundColor Green
try {
    $models = & ollama list
    Write-Host ($models | Out-String)
} catch {
    Write-Warning "      未检测到 ollama，请先启动 ollama serve"
}

# ---- 4. 构建知识库 + 评测 ----
Push-Location $Proj
try {
    $flag = ""
    if ($Rebuild) { $flag = "--rebuild" }
    Write-Host "[4/4] 构建知识库..." -ForegroundColor Green
    & $Py build_kb.py $flag

    Write-Host "运行评测（16 题）..." -ForegroundColor Green
    & $Py evaluate.py

    if ($Serve) {
        Write-Host "启动 Streamlit 演示界面: http://localhost:8501" -ForegroundColor Green
        & $Py -m streamlit run app.py
    }
} finally {
    Pop-Location
}

Write-Host "完成。产出物见: $Proj\output" -ForegroundColor Cyan