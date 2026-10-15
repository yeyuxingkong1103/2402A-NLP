# 工单编号：人工智能NLP-RAG-工单17
param([string]$Python='D:\an\envs\rags_\python.exe',[int]$Port=8817)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..\研发')
$env:PYTHONIOENCODING='utf-8'
try {
  $existing=Invoke-RestMethod -Uri ("http://127.0.0.1:"+$Port+"/health") -TimeoutSec 2
  if ($existing) { Write-Host ("本项目服务已运行，浏览器访问 http://127.0.0.1:"+$Port); exit 0 }
} catch { }
Write-Host ("浏览器访问 http://127.0.0.1:"+$Port)
& $Python -m uvicorn app.service:app --host 127.0.0.1 --port $Port
