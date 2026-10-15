# 工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
# 使用微调权重时重建本项目向量，避免用旧向量与新查询混检。
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..\研发')
$env:EMBEDDING_MODEL_PATH=Join-Path (Get-Location) 'models\bge-finance-finetuned'
$env:EMBEDDING_MODEL_NAME='bge-finance-finetuned'
$env:EMBEDDING_DEVICE='cuda'
$env:LLM_PROVIDER='ollama'
$env:LLM_BASE_URL='http://127.0.0.1:11434'
$env:LLM_MODEL='Qwen3.5:4B'
$env:LLM_TIMEOUT='120'
$env:PYTHONIOENCODING='utf-8'
& 'D:\an\envs\rags_\python.exe' -m app.main build
if ($LASTEXITCODE -ne 0) { throw '微调模型索引构建失败' }
& 'D:\an\envs\rags_\python.exe' -m app.main serve --port 8775
