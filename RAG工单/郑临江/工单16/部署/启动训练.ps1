# 工单编号：人工智能NLP-RAG-微调专用视觉语言模型工单
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..\研发')
$env:PYTHONIOENCODING='utf-8'
& 'D:\an\envs\rags_\python.exe' train_lora.py
