# 工单10 · 运维手册（OPERATION）

> 📌 **端口提醒**：本机 **8000 端口已被另一个项目占用**
> （`curl http://127.0.0.1:8000/` 会返回 `{"service":"知愈医疗RAG",...}`），
> 所以工单10 部署时请把宿主机端口换成 **8010**：
> `docker run -p 8010:8000 ...`，或 compose 里写成 `"8010:8000"`，
> 对应的访问地址也换成 `http://127.0.0.1:8010/`。


## 一、日常巡检

| 检查项 | 命令 | 正常判据 |
|---|---|---|
| 容器状态 | `docker compose ps` | `Up (healthy)` |
| 健康检查 | `curl -fsS http://127.0.0.1:8000/healthz` | 返回 `{"status":"ok"}` |
| 资源占用 | `docker stats --no-stream rag-fin-qa` | 内存 < 3.5G（限制 4g） |
| 日志异常 | `docker compose logs --since 1h rag-api \| grep -iE "error\|traceback"` | 无输出 |
| 磁盘 | `docker system df`、`df -h` | 卷占用稳定，无暴涨 |

## 二、监控与告警（建议）

工单17 会在 RAGFlow 里补 Prometheus `/metrics`；本轻量服务可用下列脚本做最小监控：

```bash
# 每 10 秒打一次内存/CPU
while true; do docker stats --no-stream --format \
  "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" rag-fin-qa; sleep 10; done
```

关键告警阈值建议：
- 内存 > 90% 限制值持续 5 分钟
- `/healthz` 连续 3 次失败
- 平均响应时间 > 3s（工单13 的验收口径）

## 三、扩缩容

```bash
# 水平扩容（注意每个实例都会加载一份 embedding 模型，内存 ×N）
docker compose up -d --scale rag-api=3
```
> 建议前置 Nginx 做负载均衡；单实例内存按 4G 规划。

## 四、数据备份

```bash
# 备份知识库卷
docker run --rm -v rag-kb:/from -v %CD%:/to alpine \
  tar czf /to/kb_backup_$(date +%Y%m%d).tgz -C /from .
# 恢复
docker run --rm -v rag-kb:/to -v %CD%:/from alpine \
  tar xzf /from/kb_backup_20260928.tgz -C /to
```

## 五、更新发布

```bash
docker compose build            # 重建镜像
docker compose up -d            # 滚动替换（compose 会重建变化的容器）
docker image prune -f           # 清理悬空镜像
```

## 六、故障处置速查

| 故障 | 处置 |
|---|---|
| 服务无响应 | `docker compose restart rag-api` → 看日志 → 必要时回滚镜像 |
| 内存持续上涨 | 见工单17 的资源泄漏排查（模型单例 / 连接池 / 显式释放） |
| 检索结果变差 | 重建知识库：`docker compose exec rag-api python scripts/build_all.py ccf` |
| LLM 调用失败 | 检查 Key 余额与网络；`curl -s https://api.deepseek.com/v1/models -H "Authorization: Bearer $KEY"` |
| 磁盘将满 | `docker system prune -a -f`；清理 `logs/` 旧文件 |
