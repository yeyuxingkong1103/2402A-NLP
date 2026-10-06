# 基于 RAG 的多角色扮演系统

一个模块化的多角色 RAG（检索增强生成）平台。支持律师、医生、中医、心理咨询师、金融理财师、
证券顾问、英语老师、虚拟朋友、客服、科普科学家等角色，具备多引擎文档解析、混合检索、
多路召回、重排序、短/长期记忆，以及 Redis 与 Milvus 可视化监控。

## 功能特性

**离线部分**
- 多引擎文档解析：PyMuPDF、pdfplumber（表格）、PaddleOCR（扫描件）、MinerU（复杂版面）、多模态大模型
- 文档清洗：去水印、去多余空白、低质量过滤、去重
- 五种分块策略：固定长度 / 句子 / 段落 / 标题 / 语义，支持父子块
- 知识库数据增强：LLM 摘要、假设问题生成（可开关）
- BGE-m3 向量化后写入 Milvus

**在线部分**
- 查询改写 / 扩写（多查询）
- 混合检索：Milvus 向量 + BM25（原生 / 本地兜底）
- 多路召回：Milvus、Redis、MongoDB、Neo4j、MySQL，RRF 融合
- BGE-reranker 精排 + 相似度阈值过滤
- 角色提示词模板 + 大模型生成（支持流式）
- 后处理：正则清洗 + 法条引用校验

**记忆与多角色**
- 短期记忆：Redis（最近 N 轮，防过期）
- 长期记忆：MongoDB 摘要
- 多用户 + 多角色（MySQL 存储角色，domain 绑定知识库）
- 知识库动态更新（上传 / 批量入库 / 删除）

**工程化**
- 多环境配置（dev / test / prod）+ `.env` 集中管理，代码零硬编码
- 统一 logging（控制台 + 滚动文件）
- 运维可视化：Redis（键类型 / 内存 / 命中率 / 热问题）+ Milvus（实体 / 域分布 / 检索预览）
- 单元测试（pytest）、集成测试、RAGAS 评测、JMeter 压测、Postman 集合

## 技术栈

| 组件 | 技术 |
|---|---|
| 后端 | FastAPI + Uvicorn |
| 前端 | Streamlit + Plotly |
| 大模型 | OpenAI 兼容（DeepSeek / 千问 / 豆包 / 硅基流动 / OpenAI / 本地 vLLM/SGLang） |
| 向量化 | BGE-m3 |
| 重排序 | BGE-reranker |
| 向量库 | Milvus（原生 BM25 混合检索） |
| 缓存/短记忆 | Redis |
| 角色/元数据 | MySQL |
| 长记忆/归档 | MongoDB |
| 图谱（可选） | Neo4j |
| 框架 | LangChain / LlamaIndex |
| 评测 | RAGAS |

## 目录结构

```
app/
  main.py            # FastAPI 装配 + 生命周期
  config.py          # 多环境配置
  logging_conf.py    # 统一日志
  schemas.py         # Pydantic 模型
  api/               # chat/roles/kb/memory/health/ops
  core/              # llm/embedder/reranker/registry
  db/                # milvus/redis/mysql/mongo/neo4j store
  ingest/            # parsers/chunker/clean/pipeline
  rag/               # query_rewrite/retriever/rerank/prompt/generator/postprocess/pipeline
  roles/presets.py   # 角色注册表
  frameworks/        # langchain/llamaindex
frontend/app_ui.py   # Streamlit（对话 + 知识库 + 运维监控）
scripts/             # install.sh / run.sh / shutdown.sh / ingest_cli.py
tests/               # test_unit.py / test_integration.py
```

## 快速开始

```bash
# 1. 安装（检查系统、装依赖、生成 .env）
bash scripts/install.sh
# 编辑 .env，填写 LLM_API_KEY 与数据库连接信息

# 2. 启动（基础服务 + 后端 + 前端）
bash scripts/run.sh
# 后端接口文档: http://localhost:8000/docs
# 前端界面:     http://localhost:8501

# 3. 停止
bash scripts/shutdown.sh
```

手动入库：`python scripts/ingest_cli.py --dir laws --domain law`

## 环境变量（.env）

| 变量 | 说明 |
|---|---|
| `APP_ENV` | 运行环境 dev / test / prod |
| `LLM_PROVIDER` | deepseek / qwen / doubao / siliconflow / openai / local-vllm / local-sglang |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | 大模型连接（本地部署时指向 vLLM/SGLang） |
| `MILVUS_*` / `REDIS_*` / `MYSQL_*` / `MONGO_*` | 各存储连接 |
| `RAG_ENGINE` | native / langchain / llamaindex |
| `CHUNK_STRATEGY` | fixed / sentence / paragraph / heading / semantic |

## 主要接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/chat` | POST | 对话 |
| `/chat_stream` | POST | 流式对话 |
| `/roles` `/roles/{id}` | GET | 角色列表 / 详情 |
| `/roles` | POST | 创建角色 |
| `/upload_pdf` | POST | 上传 PDF 入库 |
| `/kb/ingest_dir` | POST | 批量入库 |
| `/kb/docs` | GET | 文档列表 |
| `/kb/docs/{doc_id}` | DELETE | 删除文档 |
| `/memory/{uid}/{rid}` | GET | 查看短期记忆 |
| `/clear` | POST | 清空会话 |
| `/health` | GET | 健康检查（含组件状态） |
| `/ops/redis/stats` `/ops/redis/keys` `/ops/redis/key/{key}` | GET | Redis 可视化 |
| `/ops/milvus/stats` `/ops/milvus/entities` | GET | Milvus 可视化 |
| `/ops/milvus/search` | POST | 检索预览 |

完整文档见 `接口文档.md` 或访问 `/docs`。

## 测试与评测

```bash
pytest tests/                 # 单元 + 集成测试
python test_rag.py            # 接口冒烟测试（需后端运行）
python evaluate_ragas.py      # RAGAS 评测，输出 ragas_result.csv
# JMeter: 导入 test_plan.jmx 进行压力测试
```

## 部署

- 环境：Win11 + WSL2 + Ubuntu（亦支持算力云 / 腾讯云 / 阿里云）
- 基础服务：Milvus（Docker Compose）、Redis、MySQL、MongoDB
- 本地大模型：vLLM / SGLang 部署 Qwen 系列，`LLM_PROVIDER=local-vllm`
