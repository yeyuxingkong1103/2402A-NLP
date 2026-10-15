# 工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统
param([string]$Python = '', [int]$Port = 8765)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..\研发')
if (-not $Python) {
    $venv = Join-Path (Get-Location) '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venv) { $Python = $venv }
    elseif (Test-Path -LiteralPath 'D:\an\envs\rags_\python.exe') { $Python = 'D:\an\envs\rags_\python.exe' }
    else { $Python = 'python' }
}
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path -LiteralPath 'data\index\manifest.json')) {
    & $Python -m app.main build
    if ($LASTEXITCODE -ne 0) { throw '知识库构建失败' }
}
& $Python -m app.main serve --port $Port
