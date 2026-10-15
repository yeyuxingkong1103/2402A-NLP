# 工单10 · 金融问答系统部署

> 📌 **端口提醒**：本机 **8000 端口已被另一个项目占用**
> （`curl http://127.0.0.1:8000/` 会返回 `{"service":"知愈医疗RAG",...}`），
> 所以工单10 部署时请把宿主机端口换成 **8010**：
> `docker run -p 8010:8000 ...`，或 compose 里写成 `"8010:8000"`，
> 对应的访问地址也换成 `http://127.0.0.1:8010/`。


工单编号：人工智能NLP-RAG-金融问答系统部署

## 一、工单目标

以 Docker 容器形式部署金融问答服务并运行成功，含卷持久化、网络配置与运维文档。

## 二、怎么跑

```bash
docker build -f 工单10/Dockerfile -t rag-fin-qa:1.0 .
cd 工单10 && docker compose up -d
curl http://127.0.0.1:8000/healthz
```

## 三、产出物
- `工单10/Dockerfile`
- `工单10/docker-compose.yml`
- `工单10/DEPLOYMENT.md`
- `工单10/OPERATION.md`
- `工单10/部署问题记录.md`

## 五、需要截图的地方

见同目录 `截图清单.md`。
