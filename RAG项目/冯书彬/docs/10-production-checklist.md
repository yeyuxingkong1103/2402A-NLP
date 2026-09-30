# 生产部署验收清单

本清单用于内部测试、准生产和正式上线前的人工签字，不替代安全审查或法律专业验收。

## 配置与发布

- [ ] `ENVIRONMENT=production` 已设置。
- [ ] `AUTH_STORE_BACKEND=sql`、`OTP_STORE_BACKEND=redis`、`REQUEST_CONTROL_BACKEND=redis` 已设置。
- [ ] `SERVE_FRONTEND=false`，`CORS_ORIGINS` 仅包含正式前端来源。
- [ ] `RAG_FRAMEWORK=langchain`，并确认 `legacy` 回退开关由值班人员知晓。
- [ ] 生产镜像、依赖锁定和数据库迁移版本已记录。
- [ ] 未使用开发身份头、固定测试手机号或默认 Compose 凭据。

## 服务与网络

- [ ] `/health/live` 在反向代理和容器编排层通过。
- [ ] `/health/ready` 在 MySQL、Redis、Milvus、Celery、模型和密钥均就绪后返回 200。
- [ ] `/metrics` 仅对内部采集网络开放；未输出用户文本、密钥或连接串。
- [ ] TLS、反向代理超时、请求体大小和访问日志脱敏规则已验证。
- [ ] Celery worker 常驻，并验证 `inspect().ping()`。
- [ ] 依赖数据卷、权限和容量告警已验证。

## 数据保护与灾备

- [ ] 使用 `python -m backend.scripts.backup_database --output <文件>` 完成一致性备份。
- [ ] 备份文件已加密、短期访问控制，并排除临时导出密文。
- [ ] 使用独立临时数据库运行 `python -m backend.scripts.verify_database_backup --database-url <独立库> --backup <文件>`。
- [ ] 恢复后核对 `alembic_version`、知识材料、文档块、会话和消息数量。
- [ ] 已记录 RPO/RTO、备份保留期和恢复责任人。
- [ ] 演练结束后已清理临时库和临时备份文件。

## 性能与安全

- [ ] 已执行 `python -m backend.scripts.health_load_test --requests 1000 --concurrency 20` 并保存 JSON 结果。
- [ ] 已对真实聊天 API、RAG 检索、DeepSeek、Milvus 和 Celery 做独立多实例压测。
- [ ] 已验证限流、并发控制、超时、队列堆积和下游故障降级。
- [ ] 已完成依赖漏洞、安全配置和日志脱敏审查。
- [ ] 已完成高风险法律场景人工验收和应急联系方式核验。

## 结论

- [ ] 内部测试通过。
- [ ] 准生产通过。
- [ ] 正式生产批准人：________________
- [ ] 批准日期：________________
