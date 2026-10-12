# 工单 10 — 金融问答系统部署

> 工单编号：人工智能NLP-RAG-金融问答系统部署

将金融问答系统以 Docker 容器形式部署到服务器，支持数据卷持久化、网络配置与容器间数据共享。

## 架构

```
客户端 ──HTTP──> 容器(rag-finance-qa:8000) ──> RAG 索引(向量+BM25)
                     │
                     └── 数据卷 /data（挂载宿主 ./data，持久化金融年报）
```

## 文件

```
10/
├── app.py              # Flask HTTP 服务（/health /ask /index/build）
├── Dockerfile          # 镜像构建定义
├── docker-compose.yml  # 一键编排
├── requirements.txt
├── .dockerignore
└── README.md
```

## 部署步骤

### 1. 准备数据

把金融年报 txt 放到本目录的 `data/` 下（本项目中即 ccf_competition 的 txt）：

```bash
mkdir -p data
cp -r "../附件/ccf_competition/txt/"* data/
```

### 2. 构建并启动容器

方式 A（docker run）：

```bash
docker build -t rag-finance-qa .
docker run -d --name rag-finance-qa \
  -p 8000:8000 \
  -v "$(pwd)/data:/data" \
  -e LLM_API_BASE= -e LLM_API_KEY= -e LLM_MODEL=deepseek-chat \
  rag-finance-qa
```

方式 B（docker-compose，推荐）：

```bash
docker compose up -d --build
```

### 3. 验证服务

```bash
curl http://localhost:8000/health
# {"status":"ok","chunks":24917}

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"平安银行在2019年的董事长致辞中，提到其盈利增长的关键因素有哪些？"}'
```

### 4. 容器管理

```bash
docker logs -f rag-finance-qa      # 查看日志
docker stop rag-finance-qa         # 停止
docker rm rag-finance-qa           # 删除容器（数据卷不丢失）
docker compose down                # 停止并移除 compose 服务
```

## 验收要点

- `docker run` 成功启动容器并在 8000 端口提供服务，无异常日志
- 数据通过 Docker 卷 `/data` 持久化，重启不丢失；多容器挂载同卷可实现数据共享
- 容器网络配置正确，可与其它服务（如 LLM 服务、RTMP 服务器）正常通信
