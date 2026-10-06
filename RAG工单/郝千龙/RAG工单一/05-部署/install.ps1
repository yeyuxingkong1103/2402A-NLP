# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【环境安装脚本 · install.ps1】安装 Python 依赖
# 用法：./install.ps1
# ======================================================================
$ErrorActionPreference = "Stop"
$DeployDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $DeployDir
$Req = Join-Path $Root "02-研发\requirements.txt"

Write-Host "[1/2] 升级 pip ..." -ForegroundColor Green
python -m pip install --upgrade pip

Write-Host "[2/2] 安装依赖: $Req" -ForegroundColor Green
python -m pip install -r $Req

Write-Host "[OK] 依赖安装完成" -ForegroundColor Green
Write-Host "下一步：./start.ps1" -ForegroundColor Cyan
