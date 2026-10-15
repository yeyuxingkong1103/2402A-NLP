# 工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统
param([string]$Python = '', [int]$Port = 8765)
$env:LLM_PROVIDER = 'ollama'
$env:LLM_BASE_URL = 'http://127.0.0.1:11434'
$env:LLM_MODEL = 'Qwen3.5:4B'
# 120 秒用于完整功能验证；不等于 3 秒性能目标已通过。
$env:LLM_TIMEOUT = '120'
& (Join-Path $PSScriptRoot '启动.ps1') -Python $Python -Port $Port
