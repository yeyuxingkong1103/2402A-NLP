# 工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统
param([string]$Python = 'D:\an\envs\rags_\python.exe')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..')
$env:PYTHONIOENCODING = 'utf-8'
& $Python -m unittest discover -s 测试 -p 'test_*.py' -v
if ($LASTEXITCODE -ne 0) { throw '自动化测试失败' }
& $Python 测试\evaluate.py
if ($LASTEXITCODE -ne 0) { throw '评估运行失败' }
