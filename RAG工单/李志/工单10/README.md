# 工单10 - 金融问答系统部署

提供无第三方框架的 HTTP API、Dockerfile、健康检查和自动重启配置。

```powershell
docker compose -f .\工单10\docker-compose.yml up --build -d
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post http://localhost:8000/ask -ContentType application/json -Body '{"query":"公司的主营业务是什么？"}'
```

停止：`docker compose -f .\工单10\docker-compose.yml down`。
