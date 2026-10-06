# 基于 RAG 的角色扮演系统

这是一个可运行的 Python/FastAPI 后端示例，面向“虚拟朋友、NPC、医生、心理医生、律师、金融研究员、教师、英语教练”等多角色聊天场景。项目把离线知识库处理和在线问答链路拆成可替换模块，默认不依赖外部服务即可启动。

## 已实现能力

- PDF 文本解析：PyMuPDF；表格解析：pdfplumber；扫描件预留 PaddleOCR-VL 接口
- TXT/MD/CSV/JSON 文档入库
- 结构感知 + 固定长度重叠分块
- BGE-m3 / Sentence Transformers 可选，默认使用确定性的本地 hash embedding 降级模式
- 本地持久化向量索引，可切换 Milvus
- 向量召回 + BM25 风格词法召回 + RRF 混合融合
- BGE-Reranker 可选，未安装时使用可解释的词法重排
- 多用户、多角色、多会话
- Redis 短期记忆；内存降级；MySQL/SQLite 持久化聊天记录
- OpenAI 兼容大模型 API：DeepSeek、千问、豆包兼容网关、硅基流动等可通过 URL/Key 切换
- Mock LLM、本地流式 SSE、查询改写、提示词模板、回答后处理
- RAGAS 评测入口和本地代理指标
- 详细 JSON 日志、请求 ID、阶段耗时、引用信息
- Docker Compose、Ubuntu/CentOS 安装脚本、启动/停止脚本

## 快速启动（Conda zhuangao6）

当前工作目录：`D:\rag-roleplay-system`。项目使用 Conda 环境：`D:\develop_tool1\anaconda3\envs\zhuangao6`，后续命令请优先使用该环境。

当前 Conda 环境目录由管理员组保护，项目依赖安装在 D 盘的 `conda_packages` 覆盖目录；启动时由 Conda 的 `zhuangao6\python.exe` 加载，并关闭 C 盘用户级 Python 包。

PowerShell：

```powershell
Set-Location D:\rag-roleplay-system
conda activate zhuangao6
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = "D:\rag-roleplay-system\conda_packages"
python run.py
```

也可以直接运行，不依赖当前 PATH：

```powershell
Set-Location D:\rag-roleplay-system
.\run_zhuangao6.ps1
```

## 快速启动（从零创建 Conda 环境）

```powershell
Copy-Item .env.example .env
conda create -n zhuangao6 python=3.10 pip -y
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = "D:\rag-roleplay-system\conda_packages"
conda run -n zhuangao6 python -s -m pip install --ignore-installed --target D:\rag-roleplay-system\conda_packages -r requirements.txt
conda run -n zhuangao6 python run.py
```

打开 `http://localhost:8000/docs` 查看 Swagger 接口文档。首次启动会自动创建 SQLite 数据库和 5 个示例角色。

PowerShell 也可以直接执行：

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 最小演示流程

1. `GET /api/v1/roles` 获取角色 ID。
2. `POST /api/v1/documents/upload` 上传 `data/demo_knowledge.txt`。
3. `POST /api/v1/chat`，将返回的角色 ID 放入 `role_id`。
4. 将 `stream` 设为 `true` 可使用 SSE 流式返回。

示例请求：

```json
{
  "user_id": "demo-user",
  "role_id": "从 GET /api/v1/roles 获取",
  "conversation_id": "demo-conversation",
  "message": "高血压管理有哪些注意事项？",
  "stream": false
}
```

## 接入真实模型

编辑 `.env`：

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=你的密钥
LLM_MODEL=deepseek-chat
RAGAS_ENABLED=true
```

`RAGAS_ENABLED` 默认是 `false`。开启后评测接口会使用配置的 OpenAI 兼容模型执行真实 RAGAS；开发环境保持关闭即可离线运行。

使用 BGE-m3：

```dotenv
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
```

如果使用 FlagEmbedding 的 BGE-M3 原生接口，可以把 `EMBEDDING_PROVIDER` 设为 `bge_m3`，并按模型实际向量维度修改 `EMBEDDING_DIMENSION`。首次推理会下载模型权重，生产环境建议预下载并挂载模型目录。

启用 BGE-Reranker：

```dotenv
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

## Docker

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps
```

Compose 会启动应用、MySQL、Redis、Milvus Standalone 及其依赖。示例 Compose 默认使用 hash embedding 和 mock LLM，便于先验证服务；接入 GPU、本地 vLLM/SGLang 或在线 API 时，通过环境变量替换。

## 目录说明

```text
app/
  api/       FastAPI 路由
  core/      配置、日志、数据库
  rag/       解析、分块、Embedding、检索、重排、提示词、LLM、评测
  storage/   SQLAlchemy 模型和仓储
tests/       单元测试
docs/        需求、设计、接口、评测、部署文档
scripts/     Ubuntu/CentOS/Compose 脚本
data/        演示资料、本地索引、上传文件
logs/        JSON 日志
```

## 生产注意事项

- 医疗、法律、金融角色必须配置可靠、时效明确的知识库，并在上线前进行人工审核。
- 不要将真实 API Key 写入代码或提交到 Git；用环境变量或密钥管理系统。
- 生产环境建议使用 MySQL、Redis、Milvus 集群和 Nginx/负载均衡，并配置鉴权、限流、审计和 HTTPS。
- 本项目默认的 Mock LLM、hash embedding、内存记忆只用于开发和功能联调。
- `data/demo_knowledge.txt` 是演示数据，不是医疗指南。

更多内容见：

- [需求规格说明书](docs/requirements.md)
- [技术设计文档](docs/design.md)
- [接口文档](docs/api.md)
- [部署文档](docs/deployment.md)
- [RAG 评测与优化](docs/evaluation.md)
- [项目使用文档](docs/user-guide.md)
- [代码解释文档](docs/code-explanation.md)
