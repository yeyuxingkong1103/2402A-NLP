# RAGFlow 运维手册

---

## 一、日常监控

### 容器状态

```bash
docker compose ps
```

### 资源占用

```bash
docker stats --no-stream
```

### 日志

```bash
docker logs ragflow-server --tail 100
docker logs ragflow-es-01 --tail 100
```

### Prometheus 指标（如已集成）

```bash
curl http://localhost:9380/metrics
```

---

## 二、常见故障处理

| 现象 | 排查方向 | 处理 |
| --- | --- | --- |
| 响应变慢 | docker stats CPU | 检查外网 API 限流 |
| 内存增长 | 会话历史 | 清理旧会话 |
| 端口冲突 | netstat -ano | 关闭冲突程序 |
| 容器 Restarting | docker logs | 看具体报错 |

---

## 三、性能优化清单

- Embedding 本地化（bge-m3）
- Chat 换本地或付费模型
- 会话历史限制（`MAX_HISTORY=20`）
- 请求队列限流（Nginx/网关层）

---

## 四、备份恢复

```bash
docker exec ragflow-mysql mysqldump -uroot -pinfini_rag_flow rag_flow > backup.sql
docker cp ragflow-minio:/data C:\backup\minio-data
```
