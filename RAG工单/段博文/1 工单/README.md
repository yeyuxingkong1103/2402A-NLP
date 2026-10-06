# 基于 PDF 文档的问答系统

> 工单编号：**人工智能NLP-RAG-基于PDF文档的问答系统**

## 1. 项目简介

本项目是一个基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 思想构建的 PDF 文档问答系统。用户把 PDF 招股说明书 / 财报 / 技术文档等上传到知识库后，即可用自然语言提问，系统会：

1. 用 **bge-m3** 把问题向量化；
2. 在 **Milvus** 中做 **向量召回 + BM25 关键词召回 + RRF 融合**；
3. 用 **bge-reranker-large** 做二次精排；
4. 把 Top-K 片段拼进上下文，调 **DeepSeek** 大模型生成回答；
5. 在前端同时展示 **回答** 与 **引用的原文片段**，便于核对。

技术栈：FastAPI（端口 8000）+ 原生 HTML/JS/CSS 前端 + Milvus 向量库 + LangChain 1.x + DeepSeek。

## 2. 工单编号

- **工单编号**：人工智能NLP-RAG-基于PDF文档的问答系统
- **测试语料**：武汉兴图新科电子股份有限公司《招股说明书》PDF（位于 `data/招股说明书1.pdf`）
- **测试问题**：工单规定的 10 道问题（详见 `tests/test_questions.py`）

## 3. 目录结构

```
成品/
├── README.md                       # 本说明文档
├── .env                            # 配置文件（install.sh 自动生成，需填入 API Key）
├── data/                           # 知识库 PDF 原文件（上传接口也落盘到此）
│   └── 招股说明书1.pdf
├── src/                            # 后端代码 + 前端静态资源
│   ├── main.py                     # FastAPI 入口（uvicorn main:app，需在 src/ 内启动）
│   ├── config.py                   # 全局配置（Milvus / 模型路径 / DeepSeek / 服务端口）
│   ├── db_milvus.py                # Milvus 数据层（向量召回 + 写入）
│   ├── retriever.py                # 混合检索（向量 + BM25 + RRF + 重排）
│   ├── llm_client.py               # DeepSeek 调用（同步 + 流式）
│   ├── pdf_parser.py               # PDF 解析
│   ├── text_splitter.py            # 文本切分
│   ├── ingest_pdf.py               # 入库工具脚本
│   ├── logger.py                   # 日志
│   └── web/                        # 前端静态资源（main.py 把它挂在 /web 路径）
│       ├── index.html              # 问答界面（两个标签页：问答对话 / 知识库管理）
│       ├── app.js                  # 前端逻辑（API 封装 / 流式问答 / Markdown / Toast）
│       └── style.css               # 简洁现代风格，左右分栏布局
├── deploy/                         # 部署脚本
│   ├── requirements.txt            # Python 依赖清单
│   ├── install.sh                  # Linux 一键安装脚本
│   ├── start.sh                     # 启动脚本（后台运行，日志输出到 logs/）
│   └── stop.sh                     # 停止脚本（可选停止 Milvus：--milvus）
├── tests/                          # 测试
│   ├── test_questions.py           # 10 道工单问题对比测试（RAG vs 纯 LLM）
│   ├── screenshots/                # 测试截图存放目录
│   │   └── placeholder.txt
│   ├── results.json                # 测试结果（机器可读，运行后生成）
│   └── results.md                  # 测试结果（人工可读对照表，运行后生成）
└── logs/                           # 运行日志（start.sh 自动创建）
    └── app_YYYYMMDD_HHMMSS.log
```

> 说明：后端代码（main.py / config.py 等）平铺在 `src/` 下，main.py 用 `from config import ...` 做平级导入，所以 `uvicorn main:app` 必须在 `src/` 目录内启动——`start.sh` 已自动 `cd src/` 处理此事。

## 4. 环境要求

| 组件 | 版本 / 说明 |
| --- | --- |
| 操作系统 | Linux（Ubuntu 20.04+ / CentOS 7+）；前端开发可在 Windows 调试 |
| Python | 3.10（由 conda 环境 `rag_pdf_qa` 提供） |
| conda | Miniconda 或 Anaconda |
| Docker | 用于运行 Milvus standalone |
| GPU（可选） | NVIDIA GPU + CUDA，加速 bge-m3 / bge-reranker 推理；CPU 也能跑，只是慢 |
| Milvus | 2.4.x standalone，端口 19530 |
| 本地模型 | BAAI/bge-m3（约 2.3GB）+ BAAI/bge-reranker-large（约 2.1GB） |
| LLM | DeepSeek API（`deepseek-chat` 模型，走 OpenAI 兼容协议） |

## 5. 安装步骤

> 以下命令在 Linux 终端执行，工作目录为项目根目录 `成品/`。

### 5.1 一键安装（推荐）

```bash
chmod +x deploy/install.sh
./deploy/install.sh
```

`install.sh` 会自动完成 8 个步骤：

1. 检查 conda 是否安装（缺失则报错退出）；
2. 创建 conda 环境 `rag_pdf_qa`（Python 3.10）；
3. `pip install -r deploy/requirements.txt`；
4. 检查 CUDA（`nvidia-smi`），输出 GPU 信息；
5. 检查 Docker 是否安装并运行；
6. 下载并启动 Milvus standalone 容器（端口 19530）；
7. 提示下载 `bge-m3` 与 `bge-reranker-large`（首次启动时 langchain-huggingface 会自动拉取）；
8. 生成 `.env` 模板，提示填入 `DEEPSEEK_API_KEY`。

### 5.2 手动安装（分步）

```bash
# 1. 创建 conda 环境
conda create -n rag_pdf_qa python=3.10 -y
conda activate rag_pdf_qa

# 2. 安装依赖
pip install -r deploy/requirements.txt

# 3. 启动 Milvus（Docker）
wget -O deploy/milvus-standalone-docker-compose.yml \
  https://github.com/milvus-io/milvus/releases/download/v2.4.10/milvus-standalone-docker-compose.yml
cd deploy && docker compose -f milvus-standalone-docker-compose.yml up -d && cd ..

# 4. 配置 .env（参考 install.sh 生成的模板）
# 主要字段：DEEPSEEK_API_KEY / MILVUS_URI / API_PORT
```

### 5.3 配置 DeepSeek API Key

编辑项目根目录下的 `.env` 文件，填入真实的 `deepseek_api_key1`（后端 `config.py` 读取的就是这个变量名；测试脚本同时兼容大写 `DEEPSEEK_API_KEY`）：

```env
# DeepSeek API（后端 config.py 读这两个变量名）
deepseek_api_key1=sk-xxxxxxxxxxxxxxxx
deepseek_base_url=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 测试脚本 tests/test_questions.py 兼容用的大写名（值同上）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# Milvus
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=rag_pdf_qa

# 服务
API_HOST=0.0.0.0
API_PORT=8000

# 本地模型路径（Linux 必填，config.py 默认是 Windows 路径）
EMBED_MODEL_PATH=/abs/path/to/models/bge-m3
RERANK_MODEL=/abs/path/to/models/bge-reranker-large
```

> DeepSeek API Key 申请：https://platform.deepseek.com/
> 模型可用 `huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3` 预下载到项目根 `models/` 下。

## 6. 使用方法

### 6.1 启动服务

```bash
chmod +x deploy/start.sh
./deploy/start.sh
```

`start.sh` 会：

1. 激活 conda 环境 `rag_pdf_qa`；
2. 检查 Milvus 容器是否运行（未运行会自动 `docker start`）；
3. 用 `uvicorn main:app` 后台启动 FastAPI（nohup）；
4. 日志输出到 `logs/app_YYYYMMDD_HHMMSS.log`，进程 PID 写入 `logs/app.pid`。

启动成功后：

- 前端页面：<http://localhost:8000/>
- API 文档：<http://localhost:8000/docs>
- 查看日志：`tail -f logs/app_*.log`

### 6.2 停止服务

```bash
./deploy/stop.sh              # 仅停 FastAPI
./deploy/stop.sh --milvus     # 连 Milvus 一起停（容器保留，下次 start 直接复用）
```

### 6.3 前端操作

打开 <http://localhost:8000/> 后，浏览器加载 `src/web/index.html`。

**标签页一：问答对话**

- 左侧输入框输入问题，按 `Enter` 发送（`Shift + Enter` 换行）；
- 回答以 **Markdown 流式** 方式逐字渲染；
- 右侧「检索来源」区显示引用的原文片段、来源文件名、相似度分数；
- 「停止」按钮可中断正在生成的回答。

**标签页二：知识库管理**

- 点击或拖拽上传 PDF（仅支持 `.pdf`）；
- 点「上传 PDF」按钮把文件传到后端落盘；
- 点「入库」按钮触发解析 → 切分 → 向量化 → 写入 Milvus；
- 「知识库状态」面板显示已入库文档数、总切片数；
- 「刷新」按钮重新拉取统计。

## 7. 接口说明

后端 FastAPI 提供以下 HTTP 接口（详见 <http://localhost:8000/docs>）。前端 `src/web/` 由后端挂在 `/web` 路径托管，访问根路径 `/` 会自动跳转到 `/web/index.html`。

| 方法 | 路径 | 说明 | 请求参数 |
| --- | --- | --- | --- |
| `GET`  | `/` | 根路径，自动 302 跳转到 `/web/index.html` | 无 |
| `GET`  | `/api/health` | 健康检查，返回集合切片数 | 无 |
| `POST` | `/api/upload` | 上传 PDF（仅落盘到 data/，不入库） | `multipart/form-data`：字段 `file`（**单文件**，前端逐个上传） |
| `POST` | `/api/ingest` | 入库：解析 → 切分 → 向量化 → 写入 Milvus + 重建 BM25 | `application/json`：`{ "files": ["xx.pdf"] }`；`files` 为空/不传则处理 data 下所有 PDF |
| `GET`  | `/api/search` | 纯检索（不调 LLM），用于展示引用 | query string：`?query=关键词&top_k=5` |
| `POST` | `/api/chat` | RAG 问答（支持流式 SSE） | `application/json`：`{ "query": "...", "top_k": 5, "stream": true }` |

### 7.1 `/api/chat` 响应

- **流式（`stream: true`）** 返回 `text/event-stream`，逐行 `data: {"chunk": "..."}`，结束发 `data: {"done": true}`，出错发 `data: {"error": "..."}`。流式不返回引用，前端在调用 chat 前先调 `/api/search` 拿来源。
- **非流式（`stream: false`）** 返回 `application/json`：
  ```json
  { "query": "...", "answer": "...",
    "references": [ { "page_content": "...", "source": "xx.pdf", "score": 0.87, "page_number": 12 } ] }
  ```

### 7.2 cURL 示例

```bash
# 1. 上传 PDF（单文件，字段名 file）
curl -X POST http://localhost:8000/api/upload \
  -F "file=@data/招股说明书1.pdf"

# 2. 入库（不传 files 表示处理 data 下全部 PDF）
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{}'

# 3. 问答（非流式）
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"武汉兴图新科电子股份有限公司注册资本是多少？","top_k":5,"stream":false}'

# 4. 仅检索（GET，query 走 query string）
curl -G http://localhost:8000/api/search \
  --data-urlencode "query=注册资本" --data-urlencode "top_k=3"

# 5. 健康检查
curl http://localhost:8000/api/health
```

## 8. 测试方法

### 8.1 10 道工单问题对比测试

`tests/test_questions.py` 对工单规定的 10 个问题分别跑两路回答：

- **A. RAG 问答**：调用后端 `/api/chat`（带知识库检索 + 重排 + 上下文拼装）；
- **B. 纯 LLM 问答**：直接调 DeepSeek（不经过任何检索），用于对照展示 RAG 的价值。

每题记录：回答正文、引用来源、耗时、关键词命中数。

#### 工单 10 个测试问题

1. 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？
2. 武汉兴图新科电子股份有限公司参与制定了哪个技术标准？
3. 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？
4. 根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？
5. 武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？
6. 根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？
7. 武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？
8. 武汉兴图新科电子股份有限公司注册资本是多少？
9. 武汉兴图新科电子股份有限公司法定代表人是谁？
10. 武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？

#### 运行方式

```bash
# 前提：deploy/start.sh 已把服务跑起来，且 data/ 下已入库 PDF
conda activate rag_pdf_qa
python tests/test_questions.py
```

#### 输出

- `tests/results.json`：机器可读的完整结果（含回答、来源、耗时、关键词命中）；
- `tests/results.md`：人工可读的对照表（汇总表 + 每题详细对比 + 检索来源片段）；
- 控制台会打印每题的 RAG / LLM 耗时与关键词命中统计。

#### 环境变量

`test_questions.py` 会自动从项目根 `.env` 读取以下变量（也支持 shell `export` 覆盖）：

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `API_BASE` | `http://localhost:8000` | 后端 FastAPI 地址 |
| `DEEPSEEK_API_KEY`（或 `deepseek_api_key1`） | （必填） | 纯 LLM 对照组调用 DeepSeek；两个名都查，任一填了即可，都没填则自动跳过 LLM 对照 |
| `DEEPSEEK_BASE_URL`（或 `deepseek_base_url`） | `https://api.deepseek.com/v1` | DeepSeek 接口地址 |
| `DEEPSEEK_MODEL`（或 `LLM_MODEL`） | `deepseek-chat` | 模型名 |

> 说明：后端 `config.py` 用的是小写名 `deepseek_api_key1` / `deepseek_base_url` / `LLM_MODEL`；测试脚本为复用同一份 `.env`，会同时兼容大写名和小写名。

### 8.2 测试截图

`tests/screenshots/` 目录用于存放测试截图（前端问答页、知识库管理页、控制台输出、`results.md` 渲染效果等），便于交付归档。详见该目录下的 `placeholder.txt` 说明。
