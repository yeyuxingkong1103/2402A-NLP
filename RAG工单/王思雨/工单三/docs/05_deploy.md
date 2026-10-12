# 05 部署

**工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化**

---

## 1. 环境要求

| 项 | 要求 | 本机实测 |
|---|---|---|
| 操作系统 | Windows 10/11（Linux/macOS 亦可，路径写法自行调整） | Windows 11 |
| Python | 工单指定 3.11 | **3.12.7**（代码未用 3.12 专有语法，3.10+ 均可） |
| GPU | 可选。有则 `DEVICE=cuda`，无则改 `cpu`（索引构建与检索会慢 5～10 倍） | RTX 4060 Laptop 8G，torch 2.13.0+cu126 |
| Node.js | 18+（仅前端需要） | 已安装 |
| Docker | 启动 Milvus 与 Redis 需要 | 已安装 |

**依赖服务端口**：Milvus `19530`、Redis `6379`、FastAPI `8000`、Vue 开发服务器 `5173`。

> ⚠️ **端口冲突提醒**：本机 19530 / 6379 已被既有容器占用。**不要再执行 `docker compose up`**，直接复用现成服务即可（见 §3）。

---

## 2. 目录结构

```
工单三/
├── app/
│   ├── config.py          # 配置：从 .env 与环境变量读取，密钥不落代码
│   ├── pdf_parser.py      # MinerU 主解析（-t True 开表格识别）+ 三级兜底
│   ├── table_struct.py    # ★ 表格结构化与定位：HTML→矩阵→行块→定位串
│   ├── chunking.py        # 语义分块：表格行 / 标题 / 段落分别处理
│   ├── vector_store.py    # bge-m3 编码 + Milvus 读写
│   ├── retriever.py       # BM25 + 向量并行召回 + 融合 + 重排
│   ├── cache.py           # Redis 缓存，不可用时静默降级
│   ├── rag_core.py        # 索引构建 + DeepSeek 生成 + 问答主流程与容错
│   ├── main.py            # FastAPI：/upload /parse /ask /evaluate /health
│   └── evaluate_ragas.py  # 工单 14 题评测 + Ragas 四项
├── frontend/              # Vue3 + Vite 问答界面
├── data/
│   ├── raw/               # 两篇招股说明书 PDF（工单指定知识库）
│   ├── parsed/            # MinerU 解析产物（自动生成，可删）
│   ├── index.json         # 分块结果（自动生成，可删）
│   └── bm25.pkl           # BM25 索引（自动生成，可删）
├── tests/test_questions.json  # 工单 14 题 + 标准答案 + 关键字段 + 出处页
├── results/               # 评测结果留档
├── docs/                  # 01~05 与演示脚本
├── requirements.txt
├── docker-compose.yml     # 仅 Milvus + Redis（本机已有，勿重复启动）
└── .env.example           # 配置模板，复制为 .env 后填密钥
```

---

## 3. 本地启动（推荐，本机已实测）

```bash
cd "d:/Users/雨子/Desktop/专高六/liu-专高六/RAG/工单三"

# ① 配置：复制模板并填入 DeepSeek 密钥
cp .env.example .env
#   注意：路径一律用正斜杠。写成 D:\模型\bge-m3 时 "\b" 会被 .env 当成退格符。

# ② 依赖
pip install -r requirements.txt

# ②' 仅当知识库含扫描件时才需要：把 PaddleOCR 模型复制到纯英文路径
#     PaddleOCR 2.7.3 把缓存目录写死为 ~/.paddleocr/，用户名含中文时 Paddle 的 C++ 推理层打不开，
#     会报 "Cannot open file .../inference.pdmodel"（但文件其实存在）。
cp -r ~/.paddleocr/whl/{det,rec,cls} C:/paddleocr_models/
#     两篇招股说明书都不是扫描件，正常情况下这一步用不到。

# ③ 确认依赖服务在跑（本机已由既有容器提供，无需再起）
docker ps --format "{{.Names}}\t{{.Ports}}" | grep -E "milvus|redis"

# ④ 构建索引：解析 → 表格结构化 → 语义分块 → 向量入库 → 建 BM25
python -m app.rag_core
#   首次约 25~50 分钟（MinerU 解析 + 约 1.2 万个语义单元编码）；
#   解析产物与分块结果会落盘，之后重启服务是秒级加载。

# ⑤ 启动后端（启动时会自动预热：载入索引与两个本地模型，约 20~40 秒）
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
#   日志出现 [启动预热] {'ok': True, 'chunks': 6776} 即表示就绪；
#   不预热的话，第一个提问要多等 20 秒加载模型，会被误认为系统卡死。

# ⑥ 启动前端（另开一个终端）
cd frontend && npm install && npm run dev
#   浏览器打开 http://localhost:5173
```

**验证服务健康**：`curl http://127.0.0.1:8000/health`

---

## 4. Docker Compose 启动方式（可选）

`docker-compose.yml` 只编排本系统依赖的 Milvus（etcd + minio + milvus）与 Redis：

```bash
docker compose up -d          # 仅在 19530/6379 未被占用时执行
docker compose ps
```

> **本机不要执行**：19530 与 6379 已被既有容器（`rag-milvus` / 本机 Redis 服务）占用，重复启动会端口冲突。
> 应用本身按本地进程启动即可，无需容器化。

---

## 5. 环境变量示例

见 `.env.example`，关键项：

```ini
DEEPSEEK_API_KEY=sk-在这里填写你的DeepSeek密钥   # 必填
LLM_MODEL=deepseek-chat
LLM_REASONING_EFFORT=                            # 推理型模型（如 deepseek-v4-flash）须填 none，否则思考过程吃光 max_tokens 返回空内容

EMBED_MODEL_PATH=D:/模型/bge-m3                  # 正斜杠！反斜杠会被当转义符
RERANK_MODEL_PATH=D:/模型/bge-reranker-v2-m3
DEVICE=cuda                                      # 无 GPU 改 cpu

MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=gd3_chunks
REDIS_URL=redis://localhost:6379/0

MINERU_PYTHON=                                   # 装了 MinerU 的解释器；与主程序同环境则留空
MINERU_LANGFIX_DIR=C:/mineru_models/langfix      # 用户名含中文时必需，缺了 MinerU 必失败

# 检索与分块调参入口（改这里不用改代码）
SEM_CHUNK_MAX=700
SEM_THRESHOLD=0.62
SECTION_SPAN=3
VEC_TOP_K=20
BM25_TOP_K=20
FUSION_TOP_K=25
RERANK_TOP_N=6
RERANK_MAX_CHARS=512
VEC_WEIGHT=0.7
BM25_WEIGHT=0.3
```

---

## 6. 接口速查

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/upload` | 上传 PDF 到 `data/raw/` |
| POST | `/parse` | 解析并重建索引（`{"doc": "", "use_mineru": true}`） |
| POST | `/ask` | 提问（`{"question": "...", "lang": "auto"}`） |
| POST | `/evaluate` | 跑工单 14 题（`{"with_ragas": true}` 附 Ragas 四项） |
| GET | `/health` | 依赖健康检查 |

```bash
# 中文提问
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？","lang":"zh"}'

# 英文提问
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?","lang":"en"}'
```

> ⚠️ **Git Bash 下 curl 传中文会乱码**，表现为"未找到相关信息"，会被误当成检索 bug。
> 用 PowerShell、Postman 或直接在前端页面提问；先换 UTF-8 客户端对照，别急着查服务端。

---

## 7. 验收清单

| # | 验收项（工单原文） | 验证方式 | 结果 |
|---|---|---|---|
| 1 | 支持上传/解析两篇招股说明书 | `POST /upload` + `POST /parse`，`/health` 的 `pdfs` 列出两篇 | ✅ 实测：上传返回文件名/字节数；`/parse` 报告两篇均 `parser=mineru` |
| 2 | 用表格解析技术定位答案在 PDF 中的位置 | 检索结果含 `locator`（如 `招股说明书2 P157 表1 第1行`），前端逐条展示 | ✅ 实测：解析出 590 张表 / 5,131 个表格行块，前端来源卡片显示定位串 |
| 3 | 14 题问答准确率 ≥ 90% | `POST /evaluate` 或 `python -m app.evaluate_ragas` | ✅ **100%（14/14）** |
| 4 | 14 题检索结果准确无误 | **逐题人工复核答案与出处页**（不只依赖关键词判分） | ✅ 加严口径后 14/14、负样本通过率 100%；首轮假满分已修复 |
| 5 | 响应时间 ≤ 3 秒 | 评测汇总的 `avg_elapsed` / `max_elapsed` / `within_3s` | ⚠️ 评测平均 2.110s、最大 2.332s、14/14 ≤3s；但**在线大模型长尾会超出评测窗口**（实测到过端到端 5.196s）。本项目可控的检索段稳定在 579～863ms。详见 `03_test.md` §7.7 |
| 6 | 交互友好 | 打开 5173，问题输入 + 答案 + 来源 + 定位 + 耗时一屏可见 | ✅ **无头浏览器 9 项 DOM 断言全通过**，确认表格定位确实渲染到了页面上 |
| 7 | 支持中英文问答 | 中英文各提问若干，答案语言跟随提问 | ✅ 英文 4/4 |
| 8 | 高可用、容错 | 停 Milvus / 停 Redis / 坏重排路径 / 坏向量模型路径 / 坏索引文件 / DeepSeek 失效 / 扫描件 / 空问题 / 非 PDF 上传 等 | ✅ **15 项全部停机实测**，其中 **4 项**实测暴露出真实缺陷并已修复 |
| 9 | Ragas 评估并出测试报告 | `python -m app.evaluate_ragas --ragas --rows results/<file>.json` | ✅ faithfulness 1.0 / answer_relevancy 0.8509 / context_recall 1.0 / context_precision 0.9186 |
| 10 | 只实现工单要求，无多余功能 | 见 §9 自查 | ✅ |

> 验收时请特别注意第 4 项：**关键词判分通过不等于答案正确**。本项目首轮就出过一次 14/14 的假满分
> （id 4 多答了 2 家范围相近但不同的企业），靠逐题读原文才发现。复核时请连同"答案里有没有多出不该有的东西"一起看。

---

## 8. 演示视频脚本

工单 P5 要求交付"完整的演示视频"。**分镜脚本与 7 张逐镜头实拍截图见 `docs/演示视频脚本.md`**，
本文不重复，以免两处副本各自漂移。

**视频已交付**：`docs/video/工单三演示视频.webm`（84 秒，8 个镜头，由无头浏览器自动录制）。
本文件的分镜表是**人工配音版**脚本（约 7 分钟），覆盖终端演示等自动录制采不到的画面，可按需重录。

## 9. 最后自查：只做工单要求的事

**已实现（工单明确要求）**
- 两篇招股说明书的解析与检索（P3 任务描述）
- 表格解析技术定位答案（P8 备注 1）
- 14 题检索回复（P3~P7）
- 准确率、响应时间、稳定性、中英文（P4、P7）
- Ragas 评估与测试报告（P5 产出物）
- 代码注释含工单编号（P8 备注 2）

**明确未做（工单未提及）**

| 项 | 原因 |
|---|---|
| 登录 / 注册 / 权限 / 多租户 | 工单未要求 |
| 管理后台 / 支付 | 工单未要求 |
| 其他公司文档、其他问题 | 工单只指定两篇 PDF 与 14 题 |
| 模型微调 | 工单未要求（那是工单十一的范围） |
| 图像/图表内容解析 | 属工单四范围；本工单聚焦表格 |
| Query 改写 / 多轮对话 | 属工单五范围 |
| GraphRAG | 属工单八/九范围 |
