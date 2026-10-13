# 工单10 - 金融问答部署

目录中的 `docker-compose.yml`、`.env.example` 和 `deploy_check.py` 给出 API、Redis、向量库的部署骨架。`deploy_check.py` 只做配置和健康检查，不会伪造服务已启动。

```powershell
python deploy_check.py --base-url http://localhost:9380
```

完整部署需要 Docker、数据库、对象存储和模型服务，具体变量按目标云环境填写。

