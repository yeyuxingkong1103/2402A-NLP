# 工单编号：人工智能NLP-RAG-GraphRAG优化
param([string]$Python='D:\an\envs\rags_\python.exe')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..\研发')
$env:PYTHONIOENCODING='utf-8'
$env:LLM_PROVIDER='ollama'
$env:LLM_BASE_URL='http://127.0.0.1:11434'
$env:LLM_MODEL='Qwen3.5:4B'
$env:LLM_TIMEOUT='120'
$env:EMBEDDING_MODEL_PATH=Join-Path (Get-Location) 'models\bge-small-zh-v1.5'
if (-not (Test-Path -LiteralPath 'data\index\manifest.json')) { & $Python -m app.main build; if($LASTEXITCODE -ne 0){throw '建库失败'} }
Write-Host '浏览器访问：http://127.0.0.1:8773'
& $Python -m app.main serve --port 8773
