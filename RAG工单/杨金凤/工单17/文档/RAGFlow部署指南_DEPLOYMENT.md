# RAGFlow Docker Compose 部署指南

---

## 一、系统要求

- Windows 10/11 + Docker Desktop 4.30+
- 内存 16GB+，磁盘 50GB+
- 端口 80、9380、9000、9001、5455、6379、1200 未被占用

---

## 二、部署步骤

### 1. 准备镜像

```bash
docker pull infiniflow/ragflow:v0.19.1
docker pull elasticsearch:8.11.3
docker pull mysql:8.0.39
docker pull valkey/valkey:8
docker pull pgsty/minio:latest
docker tag elasticsearch:8.11.3 docker.elastic.co/elasticsearch/elasticsearch:8.11.3
```

### 2. 获取源码

从 Gitee 下载 `ragflow-v0.19.1.zip`，解压到 `C:\Users\<用户名>\ragflow\`

### 3. 修改配置

编辑 `docker\.env`：

```
RAGFLOW_IMAGE=infiniflow/ragflow:v0.19.1
MINIO_USER=minioadmin
MINIO_PASSWORD=minioadmin
HF_ENDPOINT=https://hf-mirror.com
```

编辑 `docker\docker-compose-base.yml`：

```
minio.image: pgsty/minio:latest
```

### 4. 启动

```bash
cd C:\Users\<用户名>\ragflow\docker
docker compose -f docker-compose.yml up -d
```

### 5. 验证

```bash
docker compose ps
```

浏览器打开 `http://localhost`

---

## 三、模型配置

### Ollama 本地模型（推荐）

```bash
ollama pull bge-m3
ollama pull qwen2.5:7b
```

RAGFlow 界面：

- Model providers → Ollama
- Embedding: `bge-m3:latest`
- Chat: `qwen2.5:7b`
- Base URL: `http://host.docker.internal:11434`
