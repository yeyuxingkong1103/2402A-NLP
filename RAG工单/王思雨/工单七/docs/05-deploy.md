# 第五步 · 部署

> 工单编号：**人工智能NLP-RAG-功能测试及评估**

---

## 1. 本机现状（先看这里，别重复拉起依赖）

| 组件 | 状态 | 说明 |
|---|---|---|
| Milvus | **已在运行** `localhost:19530`，容器 `rag-milvus` | 健康端口 9091。**不要再执行 `docker compose up`** —— 19530 / 6379 均已被占用，重复拉起会端口冲突 |
| Redis | **已在运行** `localhost:6379` | 本机 Redis 5.0.14，客户端必须 `protocol=2` |
| 端口 8000 | **已被其它服务占用** | 本工单 API 用 **8010** |
| 解释器 | `D:/myconda1/python.exe`（Python 3.12.7） | 全项目硬编码此解释器；无 venv |

---

## 2. 启动命令

### 2.1 后端（FastAPI）

```bash
cd D:/Users/雨子/Desktop/专高六/liu-专高六/RAG/工单七

# 方式一：一键脚本（会先探活 Redis，再启动 API）
scripts/start.bat

# 方式二：直接启动
D:/myconda1/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

启动后：

| 地址 | 说明 |
|---|---|
| `http://127.0.0.1:8010/` | **Vue 演示页**（前端由 FastAPI `StaticFiles` 挂在根路径，同源，无需跨域配置） |
| `http://127.0.0.1:8010/docs` | FastAPI 自动生成的接口文档 |
| `http://127.0.0.1:8010/health` | 健康检查（Milvus / Redis / LLM / 集合块数） |

### 2.2 前端（Vue）

前端是 **Vue 3 单文件**（`frontend/index.html`）+ 本地 `vue.global.prod.js`：
- **无构建步骤、无 npm install**（不走 CDN，离线可用）；
- 由后端 `StaticFiles` 直接托管在 `/`，**随后端一起启动，不需要单独起前端服务**。

若确实要单独起一个静态服务器（例如调试前端）：

```bash
cd D:/Users/雨子/Desktop/专高六/liu-专高六/RAG/工单七/frontend
D:/myconda1/python.exe -m http.server 8020
# 注意：此时前端与 API 不同源，需相应调整 API 地址或加 CORS
```

### 2.3 建索引与评估

```bash
D:/myconda1/python.exe scripts/build_index.py                # 建索引（9 份 PDF → Milvus + BM25，约 12 分钟）
D:/myconda1/python.exe scripts/run_eval.py                   # 10 题 + RAGAS 四指标
D:/myconda1/python.exe scripts/analyze_retrieval.py          # 确定性检索质量分析
D:/myconda1/python.exe scripts/sweep_retrieval.py            # 参数扫描（零 API 成本，含尺子自检）
D:/myconda1/python.exe scripts/measure_noise.py              # 量 RAGAS 噪声底（比较指标前必须先跑）
D:/myconda1/python.exe tests/test_pipeline.py                # 自检（必须 0 退出）
D:/myconda1/python.exe scripts/mineru_smoke.py               # MinerU 主路径冒烟验证
```

---

## 3. 接口清单

| 接口 | 方法 | 请求体 | 用途 |
|---|---|---|---|
| `/health` | GET | — | Milvus / Redis / LLM 状态 + 集合块数（用 `count(*)` 实时统计） |
| `/questions` | GET | — | 10 题题面 + ground_truth |
| `/retrieve` | POST | `{"question": "...", "mode": "hybrid"}` | 只跑检索链路，返回重排后的片段（含公司/年份/页码/来源通道/重排分） |
| `/ask` | POST | `{"question": "...", "mode": "hybrid", "use_cache": true}` | 检索 + 生成完整链路 |
| `/results` | GET | — | 历次评估结果文件名 |
| `/results/latest` | GET | — | 最近一次评估完整结果（前端主页数据源） |

`mode` 可选 `hybrid`（默认）/`vector`/`bm25`，用于对比三种检索方式。

---

## 4. 演示视频操作指引

> 视频需自行录制，以下是**分镜脚本**，按顺序操作即可覆盖「测试及评估流程」的全部环节。

| 镜号 | 时长 | 画面与操作 | 讲解要点 |
|---|---|---|---|
| 1 | 15s | 终端执行 `docker start rag-etcd rag-minio rag-milvus`，再 `scripts/start.bat`，浏览器打开 `http://127.0.0.1:8010/` | 说明 Milvus 三容器会随 Docker 重启集体退出、需先复活；API 起在 8010 |
| 2 | 20s | 页面顶部健康灯：Milvus up / Redis up / LLM configured / 集合 `gd7_chunks` 5660 块 | 语料是 `ccf_competition` 的 9 份年报；块数来自 `count(*)` 实时统计（5660 = 已做页眉页脚过滤后的索引） |
| 3 | 20s | 终端 `wc -l app/config.py app/ingest.py app/rag_core.py app/evaluate.py` → **300** | **硬约束展示**：核心链路 4 模块共 300 行且逐行中文注释；`main.py`(69) 与 Vue 按工单不计入 |
| 4 | 30s | 左栏展示 10 题列表，点开 Q01，展开「标准答案」 | 4 题来自 `sample_questions.pdf`、6 题按语料仿写；ground_truth 逐字取自 PDF 原文并标页码 |
| 5 | 40s | 点「① 运行检索」，展示 8 个片段：公司/年份/页码/来源通道/重排分 | **演示重点**：检索结果可溯源到具体页；`vec+bm25` 表示两路都命中 |
| 6 | 30s | 点「② 生成答案」→ **主动指出 Q01 的"多答"现象** | 答案并列了平安/招商/邮储。**强调这不是幻觉**：题面是 `xx银行` 没有公司名 + 平安银行数据出现在母公司中国平安年报里；看片段来源公司即可核验 |
| 7 | 30s | 换 Q06（邮储银行绿色贷款），重复检索+生成 | 对照标准答案 **2,433.01 亿元 / 增长 27.78%**，单公司、干净、正确 |
| 8 | 40s | 页面下方 RAGAS 指标卡：四指标 + 各指标有效样本数 | 四指标含义；**逐指标串行**与 `EVAL_MAX_WORKERS=2` 是为避免限流把样本打成 nan |
| 9 | 30s | 逐题明细表，指出低分题 | 结合 `scripts/analyze_retrieval.py` 定位问题（目标页未命中 / 模板噪声） |
| 10 | 40s | 终端跑 `sweep_retrieval.py` 与 `entity_filter_probe.py` | 两个实测结论：**融合权重无效**（四组权重首条重排分恒 0.868）；**实体过滤 0.708→1.000 但只 6/10 题触发** |
| 11 | 30s | 终端跑 `tests/test_pipeline.py`，25 条自检全 PASS | 自造产物进评估前先过自检（正例全过 + 反例全拦） |
| 12 | 40s | **诚实收尾（建议保留、别剪）**：打开 `docs/04-optimize.md` §4 念边界 | ①模板过滤**降噪但 RAGAS 测不出改善**；②方案⑥ 查询澄清**超出本工单边界**（属 01–06）；③ oracle 实验**仅上限、非可上线配置**。**这段最能体现工程可信度** |
| 13 | 20s | 目录树 + `docs/06-delivery.md` 验收对照表 | 收束到"工单要求 → 交付物 → 实测状态" |

**录制建议**：分辨率 1920×1080；终端字号 ≥16；第 5、6、8、12 镜是核心，不要快进；全程约 6 分钟。
第 6 镜（主动展示系统短板）与第 12 镜（主动声明边界）**不要为了"好看"而删掉**——它们证明的是"测过、知道边界"，比全绿的数字更可信。

> ⚠ **冷启动别当卡死（实测）**：首次 `/retrieve` 要加载 bge-m3 与重排模型，实测 **16.4 s**；之后同题回到 **3 s** 级（`/ask` 实测 3.2 s）。
> 两种录法任选：**A**（省事）录制前先跑一次 `/retrieve` 预热，镜 5 直接展示 3 s 级热态；
> **B**（严谨）保留这 16.4 s 并加字幕说明"首次请求冷启动，后续恢复 3 s 级"。
> **投产优化方向**：加**预热探针**（服务启动后自跑一次真实形状的检索+生成）或让模型常驻，消除首个用户的冷启动惩罚。
> ⚠ 注意：**只加载模型是不够的**——预热必须端到端跑一次真实形状（含检索+重排+生成），否则首帧仍会补齐未初始化的路径。
> Q01 的"多答"现象在**当前索引上可复现**（实测：片段来源与答案提及公司都是 平安银行/招商银行/邮储银行 三家），可放心实拍，无需剪辑。

**录屏工具**（本机可用组合，已实测）：`ffmpeg` 管道录屏需显式 `-vcodec mjpeg`；用浏览器自动化截图时 `puppeteer-core` 必须锁 `21.x` 且加 `--no-proxy-server`（系统代理会把本地页面也代理掉）。

---

## 5. 换机部署要点

1. 装依赖：`pip install -r requirements.txt`（**torch 必须先于 paddleocr 导入**，反了会 WinError 127）。
2. 起 Milvus + Redis（本机已有可跳过）：`docker compose up -d`（`docker-compose.yml` 已给，仅供新机器用）。
3. 下载模型到本地：`bge-m3` → `D:/模型/bge-m3`，`bge-reranker-v2-m3` → `D:/模型/bge-reranker-v2-m3`。
4. 复制 `.env.example` 为 `.env`，填 `DEEPSEEK_API_KEY`，并按新机器调整 `CORPUS_DIR` 与各类路径（**正斜杠**）。
5. 若语料换成**扫描件**（无文本层），必须把 `MINERU_ENABLED` 改回 `1`，否则 pymupdf 会得到空文本而静默降级。
6. 按 §2.3 顺序建索引 → 评估 → 自检。

---

## 6. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 检索一律返回空 | Milvus 查询用了默认 Bounded 一致性 | 已固定 `consistency_level="Strong"`；自定义调用时务必带上 |
| `/health` 块数与实际不符 | `get_collection_stats()['row_count']` 在数据未落盘时是陈旧值 | 已改用 `count(*)`；`MilvusClient` 没有 `flush` 方法 |
| Redis 连接直接失败 | 本机 Redis 5 不支持 RESP3 | 必须 `protocol=2` |
| RAGAS 指标大面积 `nan` | 并发过高触发限流，或判官输出被 `max_tokens` 截断 | `EVAL_MAX_WORKERS=2`；判官已显式 `max_tokens=4096` |
| 路径报“找不到” | `.env` 里写了反斜杠路径 | 一律改正斜杠（`\b` 会被解析成退格符） |
| 中文出现乱码 | Git Bash 的 curl 传中文会乱码 | 换 UTF-8 客户端对照，别急着查服务端 |
