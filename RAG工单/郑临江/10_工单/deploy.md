# 金融问答系统部署文档

**工单编号**：人工智能NLP-RAG-金融问答系统部署

## 一、部署步骤

```bash
# 1. 构建镜像
docker build -t finance-qa .

# 2. 启动容器（docker run 方式，挂载数据卷）
docker run -d --name finance-qa \
  -p 8000:8000 \
  -v "D:/软件/QQ/data/RAG 工单/附件:/data:ro" \
  -e DATA_DIR=/data \
  finance-qa

# 3. 或使用 docker compose
docker compose up -d

# 4. 验证服务
curl http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"武汉兴图新科电子股份有限公司法定代表人是谁？"}'
```

## 二、容器管理

| 项目 | 说明 |
|------|------|
| 启动与运行 | `docker run -d` 启动，`docker logs finance-qa` 查看日志（无异常输出） |
| 数据管理 | 通过 Docker 卷 `-v ...:/data` 持久化，容器销毁数据不丢失 |
| 数据共享 | 卷可挂载到多个容器，实现容器间数据共享 |
| 网络配置 | `docker-compose.yml` 定义 `qa-net` 桥接网络，可与其它服务（如 RTMP）通信 |

## 三、过程问题记录

1. **端口占用**：8000 被占用时修改 `-p 宿主机端口:8000`；
2. **数据卷路径**：Windows 下卷路径需用绝对路径且注意中文编码；
3. **依赖下载慢**：构建时可换国内 pip 源 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple`；
4. **LLM 未配置**：未配置 API Key 时服务降级为“检索式回答”，仍可正常提供服务。

## 四、验收对照

- `docker run` 成功启动，指定端口提供服务，运行无异常日志；
- Docker 卷持久化关键数据；
- 容器网络配置正确，支持与其它服务通信。
