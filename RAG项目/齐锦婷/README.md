# 法律领域 RAG 智能问答系统

这是一个前后端完整的法律 RAG 问答系统，包含文档解析、清洗、结构化分块、向量化、Milvus 入库、BM25 + 向量混合检索、RRF 融合、BGE Reranker 重排、DeepSeek 生成回答、多用户隔离、Redis 短期记忆和 Milvus 长期记忆。

## 技术栈

- 前端：原生 HTML、CSS、JavaScript
- 后端：FastAPI
- 结构化数据：MySQL
- 非结构化向量：Milvus
- 短期记忆：Redis
- 文档解析：远程 MinerU，可降级到本地 PyMuPDF 文本解析
- 图片解析：DashScope `qwen-vl-max`
- 向量模型：本地 `BGE-M3`
- 重排模型：本地 `bge-reranker-large`
- 最终回答：DeepSeek 官方 API

## 快速启动

1. 复制环境变量文件：

```bash
cp .env.example .env
```

2. 编辑 `.env`，填写：

- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`
- `MINERU_API_URL`
- `MINERU_API_KEY`
- `JWT_SECRET_KEY`

3. 启动基础设施：

```bash
docker compose up -d
```

4. 安装 Python 依赖：

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

5. 启动后端：

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

6. 可选：导入示例法律 PDF：

```bash
python scripts/import_sample_laws.py
```

脚本会创建默认账号：

```text
用户名：admin
密码：123456
```

7. 打开前端：

直接用浏览器打开：

```text
frontend/index.html
```

## 目录说明

```text
backend/app/api/          后端接口层：登录、知识库、文档、问答
backend/app/core/         核心配置：环境变量、数据库、JWT 安全
backend/app/models/       数据模型：MySQL 表结构和接口入参出参
parsing/                  文档解析：远程 MinerU 和本地 PDF 兜底解析
cleaning/                 文档清洗：页眉、页脚、页码、噪声删除
chunking/                 结构分块：按法律章节、条款、长度切分
embedding/                向量模型：BGE-M3 文本转向量
vectorstores/             向量库：Milvus 文档向量和长期记忆存储
retrieval/                混合检索：BM25、向量检索、RRF 融合
rerank/                   重排模型：bge-reranker-large
generation/               答案生成：DeepSeek 生成带引用回答
memory/                   记忆模块：Redis 短期记忆和 Milvus 长期记忆
ingestion/                入库流水线：解析、清洗、分块、向量化、入库
frontend/                 前端页面代码
data/uploads/             用户上传的原始文件
data/parsed/              MinerU 或本地解析结果
data/cleaned/             清洗后的文档结果
data/chunks/              结构化分块结果
data/embeddings/          向量化记录
model/bge-m3/             本地向量模型
model/bge-reranker-large/ 本地重排模型
scripts/                  辅助脚本
```

## 法律免责声明

系统生成内容仅供学习参考，不构成正式法律意见。真实案件请咨询专业律师。
