$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:HF_HOME = "D:\工单models\huggingface"
$env:HUGGINGFACE_HUB_CACHE = "D:\工单models\huggingface\hub"
$env:TRANSFORMERS_CACHE = "D:\工单models\huggingface\transformers"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
& python (Join-Path $project "main.py") @args
if ($LASTEXITCODE -ne 0) { throw "Execution failed" }
