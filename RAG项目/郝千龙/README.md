# 基于 RAG 的角色扮演系统

> 项目目标：基于 RAG 的多角色聊天机器人，支持医生 / 心理医生 / 律师 / 科学家 / 英语教师 等多个角色，提示词模板 + 多源知识库（Milvus + MySQL + Redis + BM25）+ 大模型生成。
> 默认主数据集为中英翻译句对 TSV，默认角色为「英语教师」。

## 1. 功能概览

- 多用户 / 多角色聊天（Web + HTTP API + CLI）
- 多轮对话：Redis 短期记忆 + Milvus 长期知识库
- 检索增强生成：BM25 + BGE-m3 向量 + BGE-rerank 多路召回与重排
- 知识库动态更新：上传 txt / pdf / tsv 自动入库
- PDF 处理：PyMuPDF 去水印 + PDFPlumber 表格兜底
- LangChain 编排（Prompt + Model + Retriever + Memory + Chain）
- Query 改写、余弦相似度过滤、流式输出、RAGAS 评测

## 2. 技术栈

| 类别 | 组件 |
| --- | --- |
| 大模型 | OpenAI 协议：DeepSeek / 豆包 / 硅基流动 / 千问 / Claude / ChatGPT / Gemini；本地 vLLM / SGLang / xInference |
| 编排框架 | LangChain（可选 LlamaIndex / Dify / LightRAG / RagFlow） |
| 向量库 | Milvus（混合检索 BM25 + 向量） |
| 关系库 | MySQL / SQLite |
| 缓存 / 短期记忆 | Redis（List 数据类型） |
| 向量化模型 | BGE-m3 |
| 重排序模型 | BGE-rerank |
| PDF 处理 | PyMuPDF (fitz) + PDFPlumber |
| 评测 | RAGAS |
| Web/API | Streamlit + FastAPI + Uvicorn |
| 日志 | Python logging（RotatingFileHandler） |
| 测试 | pytest / unittest / Postman / JMeter |

## 3. 目录结构

```
作业/
├── main.py              # CLI 入口
├── app.py               # Streamlit Web 入口
├── api.py               # FastAPI HTTP 入口
├── config.py            # 配置（环境变量）
├── database.py          # SQLAlchemy：用户 / 角色 / 会话 / 消息 / 知识文档
├── auth.py              # 注册 / 登录 / token
├── roles.py             # 多角色提示词模板
├── rag.py               # BM25 检索 + 混合召回
├── ingest.py            # 知识库动态更新（分块 + PDF 解析）
├── vector_store.py      # Milvus 向量库
├── embeddings.py        # BGE-m3 向量化
├── rerank.py            # BGE-rerank 重排
├── query_rewrite.py     # Query 改写
├── memory_store.py      # Redis 短期记忆
├── generation.py        # 提示词拼装 + RAG 生成（含流式）
├── langchain_chain.py   # LangChain 编排
├── logger.py            # 日志
├── ragas_eval.py        # RAGAS 评测
├── requirements.txt
├── .env.example
├── install.sh           # 部署/安装脚本
├── run.sh               # 启动脚本
├── shutdown.sh          # 停止脚本
├── docs/
│   ├── REQUIREMENTS.md  # 需求规格说明书
│   ├── DESIGN.md        # 设计文档（技术架构 / 功能设计）
│   └── API.md           # 接口文档
└── tests/
    ├── test_rag.py
    ├── api.postman_collection.json
    └── jmeter_test_plan.jmx
```

## 4. 环境准备（Ubuntu / CentOS）

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv unzip git
# 可选：redis-server / mysql-server / docker（拉取 milvus）
sudo apt install -y redis-server mysql-server
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
```

## 5. 安装

```bash
git clone <your-repo-url> roleplay
cd roleplay
cp .env.example .env   # 按需修改
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 如需 MySQL：pip install PyMySQL cryptography
```

## 6. 运行

### 6.1 CLI 对话
```bash
python main.py
```

### 6.2 Web UI（Streamlit）
```bash
streamlit run app.py
# 或
python -m streamlit run app.py
```

### 6.3 HTTP API（FastAPI + Uvicorn）
```bash
python api.py            # 默认 0.0.0.0:8000
# 或
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 6.4 一键启停
```bash
bash install.sh         # 安装并初始化
bash run.sh             # 启动 API + Streamlit（后台）
bash shutdown.sh        # 停止
```

## 7. 配置说明（.env）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| APP_ENV | development | development / testing / production |
| TSV_PATH | ./中英翻译.tsv | 主数据集路径 |
| LLM_API_KEY | 空 | 不填则走纯检索讲解模式 |
| LLM_BASE_URL | https://api.deepseek.com | 兼容 OpenAI 协议 |
| LLM_MODEL | deepseek-chat | 模型名 |
| DATABASE_URL | sqlite:///data/app.db | MySQL 形如 `mysql+pymysql://user:pwd@host:3306/roleplay?charset=utf8mb4` |
| REDIS_ENABLED | 0 | 1 启用 Redis 短期记忆 |
| MILVUS_ENABLED | 0 | 1 启用 Milvus 向量召回 |
| EMBEDDING_ENABLED | 0 | 1 启用 BGE-m3 |
| RERANK_ENABLED | 0 | 1 启用 BGE-rerank |
| TOP_K / RERANK_TOP_K | 6 / 4 | 召回 / 重排条数 |
| SHORT_MEMORY_TURNS | 8 | 短期记忆轮数 |
| SCORE_THRESHOLD | 0.0 | BM25 得分过滤阈值 |
| SECRET_KEY | dev-secret-change-me | 生产环境必须修改 |

## 8. 测试

```bash
# 单元测试
python -m pytest tests/ -v
python -m unittest tests.test_rag

# API 接口测试：在 Postman / APIpost 中导入 tests/api.postman_collection.json

# 压力测试：用 JMeter 打开 tests/jmeter_test_plan.jmx
F:\apache-jmeter-5.6.3\bin\jmeter.bat -n -t tests/jmeter_test_plan.jmx -l tests/result.jtl
# RAG 评测
python ragas_eval.py --dataset tests/eval_samples.jsonl --out tests/ragas_report.json
```

## 9. 部署

- 开发：本地直接 `streamlit run app.py` / `python api.py`
- 测试 / 生产：`bash install.sh && bash run.sh`
- 推荐 Nginx 反向代理 + 负载均衡（http-proxy / nginx）
- 进程托管：systemd / supervisor / nohup

## 10. 默认账号

- 用户名：`guest`  密码：`guest123`（init_db 时自动创建）

## 11. 角色列表

| code | 名称 | 领域 |
| --- | --- | --- |
| teacher | 英语教师 | 英语学习 |
| npc_friend | 虚拟朋友 | 社交 NPC |
| customer_service | 学习客服 | 客服 |
| doctor | 医生 | 医疗 |
| psychologist | 心理医生 | 心理咨询 |
| lawyer | 律师 | 法律 |
| stock_advisor | 证券投资顾问 | 证券投资 |
| financial_planner | 金融理财师 | 个人理财 |
| scientist | 科学家 | 科学普及 |
