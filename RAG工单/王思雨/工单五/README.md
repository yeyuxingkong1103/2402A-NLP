# 招股说明书多轮问答系统（工单五：Query 理解优化）

> **工单编号：人工智能NLP-RAG-Query理解优化**
> 对应工单：《人工智能NLP-RAG项目-05-Query理解优化任务工单V1.1-20250206.pdf》

基于两份招股说明书 PDF（武汉兴图新科电子股份有限公司、武汉力源信息技术股份有限公司）的检索问答系统，
在单轮问答之上实现**多轮对话**：指代消解（"他""这个公司"）与**实体切换**（"那武汉力源信息技术股份有限公司呢？"）。

---

## ⚠️ 先读这一条：Q5 的答案与工单期望答案不一致

| | 工单"期望答案" | 招股说明书2 原件 |
| --- | --- | --- |
| 哪个销售部的销售处最多 | 渠道销售部 | **大客户销售部** |
| 有哪些销售处 | 珠海/深圳/北京/武汉/广州（5 个） | 珠海/深圳/北京/武汉/广州/**成都**（6 个） |

**本系统默认按 PDF 原件作答，不按工单答案硬编码。** 两条独立证据：

1. **图像几何证据**（可复现）：把 `data/parsed/招股说明书2/招股说明书2/auto/images/ec4973106d….jpg` 二值化后做形态学线段检测，
   「销售处」分发线（y=1350, x=503..1104）向上的**唯一**竖线位于 **x=777**，落在**大客户销售部**（x 752..828）范围内，
   该竖线上端 y=1264 恰为大客户销售部的下边。程序化还原的层级为：
   `销售部 → 大客户销售部 → 珠海/深圳/北京/武汉/广州/成都销售处`
2. **正文证据**：招股说明书2 正文载明"……在北京、广州、成都、深圳、武汉、珠海各设有 1 家销售处"。

**验收兼容开关（默认关闭）**：若验收方强制要求工单口径，在 `.env` 中设置

```ini
Q5_ACCEPTANCE_COMPAT=true
```

此时系统**只会在事实答案之后追加一行**工单口径对照说明，**不会篡改主答案**。

---

## 验收结果一览（本机实测）

| 工单验收项 | 要求 | 实测 |
| --- | --- | --- |
| 问答准确率 | ≥90% | **100%（10/10，判分口径=文档事实）**；若按工单口径判 Q5 为错则为 **90%（9/10）** —— 两种口径均达标 |
| 响应时间 | ≤3 秒 | **最大 2755 ms，10/10 全部 ≤3s，平均 2132 ms** |
| 缓存命中响应时间 | — | **0~1 ms** |
| 多轮对话 | 支持 | 5 轮连续，指代消解与实体切换全部正确 |
| 中英文问答 | 支持 | 英文 3 题全对，英文问英文答 |
| 容错 | 有 | Milvus / Redis / 重排 / 大模型四类故障**停机实测**均可降级 |
| ragas 四项指标 | 需评估 | faithfulness 0.9708 / answer_relevancy 0.7499 / context_precision 0.7752 / context_recall 0.8000 |

完整数据、口径与复现命令见 **`docs/测试报告.md`**；界面截图见 `docs/screenshots/`。

---

## 1. 环境要求

| 项 | 要求 |
| --- | --- |
| 解释器 | Python 3.11+（本项目实测环境为 `D:\myconda1`，Python 3.12.7，CUDA torch 2.13.0+cu126） |
| GPU | 有则用（`DEVICE=cuda`），无则设 `DEVICE=cpu` |
| Milvus | `http://localhost:19530` |
| Redis | `redis://localhost:6379`（本项目针对 Redis 5.0.14 实测，客户端固定 `protocol=2`） |
| Node.js | 18+（前端构建） |
| 本地模型 | `D:\模型\bge-m3`、`D:\模型\bge-reranker-v2-m3` |
| 大模型 | DeepSeek API Key（环境变量 `DEEPSEEK_API_KEY`） |

### 一次性准备：启动 Milvus 与 Redis

```bash
# Milvus（本机已有容器栈）：若未运行则启动
docker start rag-etcd rag-minio rag-milvus
docker ps --filter name=rag-milvus        # 等待状态变为 (healthy)

# Redis（本机为 Windows 原生服务，端口 6379）
redis-cli ping                            # 期望 PONG
```

> **坑**：本机 Redis 是 5.0.14，**不支持 RESP3**。`redis-py 8.x` 默认会发 `HELLO 3`，直接报
> `unknown command 'HELLO'`。本项目已在 `backend/config.py` 固定 `REDIS_PROTOCOL=2`，无需手工处理。

---

## 2. 安装

```bash
cd "D:/Users/雨子/Desktop/专高六/liu-专高六/RAG/工单五"

# 1) Python 依赖（用你的解释器，下文以 D:/myconda1 为例）
D:/myconda1/python.exe -m pip install -r requirements.txt

# 2) 配置密钥与路径
cp .env.example .env      # 然后编辑 .env，填入 DEEPSEEK_API_KEY

# 3) 前端依赖
cd frontend && npm install && cd ..
```

---

## 3. 入库（构建知识库）

```bash
python scripts/ingest.py
```

流程：解析两份 PDF → 图像内容提取（组织结构图会做层级还原）→ 语义分块 → 写入 Milvus + 建 BM25 索引。

- 首次运行约 **5～10 分钟**（图像 OCR 约 2.5 分钟 + 向量入库数分钟）。
- 图像 OCR 结果缓存到 `data/image_texts.json`，**重建索引时自动复用**。
- 只想重建纯文本索引（跳过图像）：`python scripts/ingest.py --no-images`

---

## 4. 启动

```bash
# 后端（终端 1）
python -m backend.main
#  → http://127.0.0.1:8000/docs  可直接调试接口

# 前端（终端 2）
cd frontend && npm run dev
#  → http://127.0.0.1:5173
```

前端通过 Vite 代理把 `/api/*` 转发到后端 `127.0.0.1:8000`（代理已去掉 `/api` 前缀）。

---

## 5. 验证：工单五的 5 个多轮问题

**必须按顺序在同一个会话内提问**（Q2/Q3 依赖 Q1 的实体，Q4 依赖实体切换）。
前端左侧「工单验收多轮问题」栏可依次点击；也可用命令行。

#### 5.1 最小可复制示例（单问，先确认服务通）

> ⚠️ **这只是"单问通不通"的冒烟示例，不是全部验收内容。**
> **完整 5 问多轮对话（含指代消解与实体切换）见下方 §5.2** —— 多轮能力只有按顺序跑完 §5.2 才能验到。

```bash
python -c "import json,urllib.request as u; R=u.Request('http://127.0.0.1:8000/chat', data=json.dumps({'session_id':'demo','question':'武汉力源信息技术股份有限公司的法定代表人是谁？'}).encode(), headers={'Content-Type':'application/json'}); print(json.loads(u.urlopen(R).read())['answer'])"
```

预期输出：`武汉力源信息技术股份有限公司的法定代表人是赵马克。【招股说明书2 P23】…`

> **坑**：Git Bash 下 `curl` 传中文会乱码，而且会**伪装成检索失败**（后端收到乱码，返回"未找到相关内容"），
> 容易误判成服务端 bug。用上面的 Python 写法（或 `curl --data-binary @file` 走 UTF-8 文件）即可。

#### 5.2 完整 5 问脚本（验收用，工单原文 5 问）

```bash
python - <<'PY'
import json, urllib.request
SID = "acceptance-demo"
QS = [
 "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
 "他参与的哪个工程荣获了国家科技进步一等奖？",
 "这个公司的法定代表人是谁？",
 "那武汉力源信息技术股份有限公司呢？",
 "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
]
for q in QS:
    req = urllib.request.Request("http://127.0.0.1:8000/chat",
        data=json.dumps({"session_id": SID, "question": q}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req).read())
    print("Q:", q)
    print("  改写:", r["rewritten_query"])
    print("  A:", r["answer"][:300])
    print("  引用:", [(x["doc"], x["page"]) for x in r["references"]][:3], 
          "耗时", r["latency_ms"], "ms", "降级", r["degraded"])
PY
```

#### 5.3 期望结果

| 轮次 | 问题 | 期望答案 | 期望改写 |
| --- | --- | --- | --- |
| Q1 | 报告期内，武汉兴图新科…军用领域的收入分别是多少？ | 2016年 6,464.51 万元；2017年 14,414.16 万元；2018年 18,780.67 万元；2019年 1-6月 4,627.14 万元 | 原句（首轮） |
| Q2 | 他参与的哪个工程荣获了国家科技进步一等奖？ | 武汉兴图新科参与的"某情报、指挥、控制与通信网络一体化工程"荣获国家科技进步一等奖 | 把"他"补成公司全称 |
| Q3 | 这个公司的法定代表人是谁？ | 程家明 | 把"这个公司"补成兴图新科 |
| Q4 | 那武汉力源信息技术股份有限公司呢？ | 赵马克（Mark Zhao） | 切换实体 + 继承"法定代表人"主题 |
| Q5 | 武汉力源…组织结构图中，哪个销售部的销售处最多？有哪些销售处？ | 大客户销售部，下辖珠海、深圳、北京、武汉、广州、成都销售处（共 6 个） | 原句 |

> Q5 的差异见本文开头"⚠️ 先读这一条"。

#### 5.4 英文提问（工单要求支持中英文）

```
What are the military-related revenues of Wuhan Xingtu Xinke Electronics Co., Ltd. during the reporting period?
Who is the legal representative of this company?
What about Wuhan P&S Information Technology Co., Ltd.?
```

---

## 6. 评估

```bash
python scripts/evaluate.py            # 跑三组题（多轮/英文/单轮），结果先落盘到 results/
python scripts/evaluate.py --ragas    # 在最新结果上补算 ragas 四项指标
python scripts/evaluate.py --selftest # 判分器自检（用已知错误答案回验）
```

验收指标：准确率 ≥90%，响应时间 ≤3 秒。结果见 `docs/测试报告.md`。

#### 5.5 演示截图（可选）

```bash
# 需先启动后端(8000)与前端开发服务器(5173)
node docs/screenshot.mjs        # 截图输出到 docs/screenshots/
```

> 依赖根目录的 `package.json`（仅含 `puppeteer-core`，开发用，不影响后端与前端运行）：
> `npm install` 后即可使用。脚本会自动寻找本机可用的 Chromium/Edge。

---

## 7. 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/chat` | `{session_id, question}` → `{answer, references, rewritten_query, cached, degraded, latency_ms}` |
| POST | `/feedback` | `{session_id, question, answer, rating, comment}` → `{ok, message}` |
| GET | `/health` | 各依赖可用性 + 知识库规模 |
| GET | `/documents` | 知识库内 PDF 列表 |
| POST | `/upload` | 上传 PDF（保存到 `data/raw`，随后需重跑 `scripts/ingest.py`） |

---

## 8. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `unknown command 'HELLO'` | Redis 版本 <6 不支持 RESP3。确认 `.env` 中 `REDIS_PROTOCOL=2` |
| 刚入库却检索不到 | Milvus 默认 `Bounded` 一致性。本项目检索固定用 `Strong`，若自行改造需保留该参数 |
| `OSError: [WinError 127] ... torch\lib\shm.dll` | **Windows 上先导入 paddleocr、后导入 torch 会崩**（MKL/OpenMP DLL 冲突，顺序敏感）。本项目已强制"先 torch 后 paddle"，详见 `backend/ocr_image.py` 注释 |
| 大模型返回空内容 | 推理型模型（如 `deepseek-v4-flash`）必须传 `reasoning_effort="none"`。本项目对 `flash`/`reason` 模型名自动处理 |
| 表格里出现"八维教育""刘敏"等字样 | 招股说明书带斜向水印，MinerU 读页面图像时必然读进单元格。`backend/text_clean.py` 用实测比对出的词表清洗，并对污染块降权 |
| 响应超过 3 秒 | 见 `docs/优化报告.md`：首次请求含模型加载，需先预热；缓存命中约 200ms |

---

## 9. 目录结构

```
工单五/
├── backend/        # FastAPI 后端
├── frontend/       # Vue 3 + Vite 前端
├── scripts/        # ingest.py 入库 / evaluate.py 评测
├── eval/           # 评估题集
├── data/           # raw 原始 PDF、parsed 解析产物、index.json、bm25.pkl、entities.json
├── results/        # 评测结果（带时间戳留存）
├── docs/           # 设计文档 / 测试报告 / 优化报告 / 部署说明
├── requirements.txt
└── README.md
```
