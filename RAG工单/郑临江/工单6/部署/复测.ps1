# 工单编号：人工智能NLP-RAG
param([string]$Python='D:\an\envs\rags_\python.exe')
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..')
$env:PYTHONIOENCODING='utf-8'
& $Python 测试\回归测试.py
if($LASTEXITCODE -ne 0){throw '回归失败'}
& $Python 测试\评估.py
