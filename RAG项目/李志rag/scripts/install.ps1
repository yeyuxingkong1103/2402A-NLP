$ErrorActionPreference = "Stop"
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose pull
Write-Host "安装完成。请修改 .env 中的密钥后运行 scripts/run.ps1"

