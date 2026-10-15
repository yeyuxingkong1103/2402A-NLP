# 工单编号：人工智能NLP-RAG-金融问答系统部署
$ErrorActionPreference='Stop'
$linuxDirectory = (& wsl.exe -d Ubuntu -e wslpath -a $PSScriptRoot).Trim()
& wsl.exe -d Ubuntu -e bash ($linuxDirectory + '/容器启动.sh')
if ($LASTEXITCODE -ne 0) { throw 'Docker启动失败，请确认Docker和WSL Ubuntu可用' }
Write-Host '浏览器访问：http://127.0.0.1:8774'
