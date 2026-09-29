$ErrorActionPreference = "SilentlyContinue"
foreach ($file in @(".api.pid", ".ui.pid")) {
    if (Test-Path $file) {
        $processId = Get-Content $file
        Stop-Process -Id $processId
        Remove-Item $file
    }
}
# 只停止容器，不删除容器；Docker Desktop 中会继续显示 medical-rag。
docker compose stop
