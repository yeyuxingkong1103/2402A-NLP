# RAG 角色扮演系统（多角色对话 + RAG 知识库）

📄 文档：[接口文档](docs/接口文档.md) · [设计文档](docs/设计文档.md) · [部署文档](docs/部署文档.md) · [压测报告](stress_report/压测报告.md) · [RAGAS 基线](eval_report/baseline.md)

基于大模型的角色扮演聊天系统：
- 每个角色有独立人设和提示词模板（存 MySQL）
- 多轮对话短期记忆（Redis List，最近 10 轮）+ 全量消息持久化（MySQL）
- 大模型统一走 OpenAI 兼容协议：本地 vLLM/SGLang 与在线 API（DeepSeek/豆包/千问等）只改 `.env` 切换
- 流式（SSE）与非流式两种对话接口
- **RAG 知识库**：PDF 上传 → PyMuPDF 解析 → 分块 → BGE-m3 向量化 → Milvus 混合检索（稠密 + BM25 稀疏，RRF 融合）→ BGE-rerank 重排序 → 注入角色对话；每个角色独立 Collection，支持文档动态删除

后续阶段规划：知识库数据增强（摘要/父子块）、Query 改写、多路召回、RAGAS 评测、Jmeter 压测、部署脚本。

## 技术栈

| 组件 | 用途 |
|---|---|
| FastAPI + Uvicorn | Web 框架（自带 Swagger 接口文档） |
| SQLAlchemy + PyMySQL | MySQL 持久化 |
| redis-py | 短期记忆 / 登录态 |
| openai SDK | 大模型兼容层（OpenAI 兼容协议） |
| Milvus 3.0 (Docker) | 向量库：每角色一个 Collection，稠密 + BM25 稀疏混合检索（RRF） |
| BGE-m3 (FlagEmbedding) | 向量化（1024 维，本地 CPU） |
| BGE-reranker-v2-m3 | 重排序（本地 CPU） |
| PyMuPDF + jieba | PDF 文本提取 / BM25 中文分词 |
| pytest | 单元 / 集成测试 |

## RAG 环境准备

```bash
# 1. 启动 Milvus（需 Docker Desktop 运行中）
docker compose up -d

# 2. 下载向量化/重排序模型（约 4.5GB；脚本已内置 hf-mirror 镜像与 Xet 禁用）
python scripts/download_models.py

# 3. 安装 RAG 依赖（含 torch CPU 版）
pip install FlagEmbedding
```

## 快速开始

```bash
# 1. 环境
python -m venv venv
venv/Scripts/activate          # Linux: source venv/bin/activate
pip install -r requirements.txt

# 2. 配置
cp .env.example .env           # 填 MySQL 密码、LLM base_url/api_key/model

# 2.5 建库（Navicat 或命令行执行一次）
# CREATE DATABASE rag_roleplay CHARACTER SET utf8mb4;

# 3. 启动（首次自动建表 + 写入 5 个内置角色）
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. 打开接口文档
# http://127.0.0.1:8000/docs
```

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /api/users/register | 注册，返回 token |
| POST | /api/users/login | 登录，返回 token |
| GET | /api/roles | 角色列表（需 `X-Token` 头） |
| POST | /api/roles | 创建角色 |
| PUT | /api/roles/{id} | 更新角色 |
| DELETE | /api/roles/{id} | 删除角色 |
| POST | /api/chat | 非流式对话 `{"role_id":1,"content":"你好"}` |
| POST | /api/chat/stream | SSE 流式对话 |
| GET | /api/chat/history?role_id=1 | 历史消息 |
| POST | /api/knowledge/upload?role_id=1 | 上传 PDF 入库（multipart，字段 file） |
| GET | /api/knowledge/docs?role_id=1 | 知识库文档列表 |
| DELETE | /api/knowledge/doc?role_id=1&source=a.pdf | 删除文档（动态更新） |

对话请求体 `{"role_id":1,"content":"...","use_rag":true}`，`use_rag=false` 可关闭本次检索；回复含 `sources` 引用来源（流式在最后一条 `{"sources":[...]}` 事件中）。

## RAGAS 评测

```bash
# 一键基线评测：知识入库 → 15 题跑真实 RAG 管线 → LLM-as-judge 打分 → eval_report/baseline.md
python scripts/run_eval.py

# 数据准备（知识文档 txt → 中文 PDF；题库在 data/eval/eval_dataset.json）
python scripts/prepare_eval_data.py
```

- 指标：faithfulness / answer_relevancy / context_precision / context_recall / answer_correctness
- Judge 大模型与生成大模型同源（默认 DeepSeek）；Embeddings 用本地 BGE-m3
- 每题明细中 `—` 表示该项 judge 调用失败（nan），总分按有效样本聚合
- **基线**：recall 1.000 / precision 0.968 / relevancy 0.901 / faithfulness 0.718 / correctness 0.597
- **提示词优化（5 条准确性指令）**：faithfulness **0.931** / correctness 0.700（faithfulness +0.213）
- **few-shot（枚举列全/归类照原文示例）**：faithfulness **0.951** / correctness **0.726** / relevancy 0.955，全部指标 ≥0.70
- 对比报告：`eval_report/comparison.md`（基线→二轮）、`comparison_round3.md`（二轮→三轮）；`python scripts/compare_eval.py 基线.json 本轮.json`

> **已知依赖补丁**：ragas 0.4.3 会 `from langchain_community.chat_models.vertexai import ChatVertexAI`，但 langchain-community ≥0.4 已移除该模块，需在 venv 里给 `site-packages/ragas/llms/base.py` 的该导入加 try/except（重建 venv 后要重打）。

## Jmeter 压测

```bash
# 1. 生成测试 token
python scripts/gen_tokens.py 50

# 2. 起被测实例（压 API 骨架用 MOCK 模式）
LLM_MOCK=true uvicorn app.main:app --port 8001

# 3. 跑压测（JAVA_HOME 指向 tools/jdk-17.0.20.1+1）
tools/apache-jmeter-5.6.3/bin/jmeter.sh -n -t jmeter/plan_chat.jmx \
  -Jthreads=50 -Jloops=10 -JPORT=8001 -l results/S2.jtl

# 4. 出报告（解析 jtl → stress_report/压测报告.md）
python scripts/make_stress_report.py
```

- 场景：S1 注册（纯后端）/ S2 对话 MOCK（API 骨架）/ S3 真实 DeepSeek（端到端）/ S4 nginx 双实例负载均衡（`docker run -d --name nginx-lb -p 8090:8090 -v jmeter/nginx_8090.conf:/etc/nginx/nginx.conf:ro nginx:alpine`）
- 关键结论：单实例 100 并发 50 req/s（anyio 线程池 40 是上限）；nginx 双实例 106 req/s（+111%）；真实 LLM 延迟主导端到端（约 1.3s）
- 压测发现并修复：MySQL 连接池默认 5+10 耗尽（已调 20+40）；BGE 模型懒加载竞态（已改 LazyModelProxy）

### 快速体验（curl）

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/users/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"123456"}' | python -c "import sys,json;print(json.load(sys.stdin)['token'])")

curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" -H "X-Token: $TOKEN" \
  -d '{"role_id":1,"content":"你好，今天心情不太好"}'
```

## 内置角色

| 角色 | 分类 |
|---|---|
| 小阳 | 虚拟朋友 |
| 林医生 | 医生 |
| 王律师 | 律师 |
| 张老师 | 教师 |
| Emily | 英语学习 |

## 核心设计

- **提示词模板**：`app/core/prompts.py`，占位符 `{role_name} {persona} {history} {user_input}`；角色可自定义模板存库
- **短期记忆**：Redis key `chat:history:{user_id}:{role_id}`（List，LRU 裁剪到 10 轮，TTL 7 天）——多用户 × 多角色会话互相隔离
- **登录态**：Redis key `session:token:{token}` → user_id，TTL 7 天
- **密码**：sha256 + 随机盐，不存明文

## 测试

```bash
pytest -q          # 104 个用例：提示词/记忆/LLM 层/API 集成/RAG/评测/压测解析
```

API 集成测试用 SQLite 内存库 + 本机 Redis 15 号库（跑完自动清空）+ 假 LLM，不需要真实大模型 key。
