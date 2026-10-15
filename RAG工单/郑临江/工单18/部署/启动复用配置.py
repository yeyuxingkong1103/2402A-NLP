"""工单14-18：一键入口复用本项目已运行服务，避免端口冲突。"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for n in range(14,19):
    module='app.service' if n==17 else 'app.dashboard' if n==16 else 'app.server'
    endpoint='/api/report' if n==16 else '/health' if n==17 else '/api/health'
    text=f'''# 工单编号：人工智能NLP-RAG-工单{n}
param([string]$Python='D:\\an\\envs\\rags_\\python.exe',[int]$Port={8800+n})
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..\\研发')
$env:PYTHONIOENCODING='utf-8'
try {{
  $existing=Invoke-RestMethod -Uri ("http://127.0.0.1:"+$Port+"{endpoint}") -TimeoutSec 2
  if ($existing) {{ Write-Host ("本项目服务已运行，浏览器访问 http://127.0.0.1:"+$Port); exit 0 }}
}} catch {{ }}
Write-Host ("浏览器访问 http://127.0.0.1:"+$Port)
& $Python -m uvicorn {module}:app --host 127.0.0.1 --port $Port
'''
    (ROOT/f'工单{n}/部署/启动.ps1').write_text(text,encoding='utf-8-sig')
print('simple starters checked')
