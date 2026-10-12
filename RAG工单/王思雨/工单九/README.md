# 工单九 · 人工智能NLP-RAG-Graph RAG优化任务

> 在 **08 工单已搭建的 GraphRAG 服务与检索结果**之上做优化，从 **prompt / 检索 / 生成 / 评估** 四层下手，
> 目标：`context_precision ≥ 0.8`、`context_recall ≥ 0.9`、端到端 **≤ 3 秒**、容错与中英文问答。
>
> **本单不修改、不重写 08 工单的任何一行代码**，只以 `import 工单八.app.*` 复用其服务与索引
> （Milvus 集合 `gd8_chunks`、bge-m3、bge-reranker-v2-m3、DeepSeek、Redis）。
> 这样 before/after 的差异**只可能来自优化本身**，不来自语料或分块器的漂移。

---

## ⚠ 待需求方裁定的口径问题（**验收前请先读这一条**）

附件 `sample_questions.pdf` 的 **Q03、Q04 参考答案为综述/推断性文本**（如"拨备覆盖率的持续优化确保了银行能够吸收潜在冲击"），
其论断**不存在可逐字归属的原文片段**。按 RAGAS `context_recall` 的 claim 归属口径，
这两题**在原理上取不到高分**，故**综合 `context_recall` 无法达到工单要求的 0.90**。

该结论已排除四类原因（检索失败 / 参数不足 / 问题拆解无效 / 检索能力不足），证据见 `docs/03_测试.md` §5.7 与 `docs/04_优化.md` §1.4：

| 证据 | 实测 |
|---|---|
| 金标主题是否真被检索到 | Q03 的 8 个片段里「风险管理」出现 **5/8**、「不良贷款」**3/8** ⇒ 在 |
| 加大召回条数 | `RERANK_N` 6→12，recall **一点没动** |
| 问题拆解 | 事实题 Q02 可达 **1.0000**，但 Q03/Q04 仍 **0.0000**（该路线 precision 崩至 0.407，未采用） |
| **★社区摘要消融（力度最强）** | 同一开关把事实题 **0.60/0.75 → 1.00/1.00**，**对 Q03/Q04 纹丝不动**，precision 逐题未变 |

**请裁定其一**（详见 `docs/04_优化.md` §1.5）：
**(A)** 接受"事实题口径"，综述题单列；**(B)** 授权把参考答案改写为可归属的事实清单（=改题）；**(C)** 维持现状并如实记录未达标与根因。

**本单不擅自选择，也不为凑指标而修改题集或评估口径。**
**本单提交的建议：同时提交 (A) + (C)，并推荐 (A)**——完整提交件见 `docs/08_待裁定口径提交.md`。

---

## 附件题集（已就位）

`data/sample_questions.pdf`（211,320 字节）已放入。**注意：附件实为 4 组问答，工单文字写的是 3 题**
——本单**以附件为准**，按全部 4 题评估，并输出逐题得分（`per_*` 字段）以便按任意 3 题子集重新聚合。

```bash
python -m app.eval_ragas --sample-pdf data/sample_questions.pdf     # 完整评估（before + after + RAGAS）
```

也可用 `--sample-pdf <绝对路径>` 或环境变量 `SAMPLE_Q_PDF` 指定别处的题集。

---

## 快速开始

```bash
# 0) 依赖（解释器：D:/myconda1/python.exe，Python 3.12.7）
python -m pip install -r requirements.txt
cp .env.example .env            # 填 DEEPSEEK_API_KEY；路径一律用正斜杠

# 1) 确认四件套在线（Milvus / Redis / Key / 图谱）
docker ps | grep rag-           # 若容器 Exited：docker start rag-etcd rag-minio rag-milvus
curl -s http://127.0.0.1:8012/health

# 2) 建社区摘要（**建议跑**：实测该通道把事实题 recall 从 0.60/0.75 提到 1.00/1.00，precision 不变）
python -m app.graph_rag build

# 3) 起服务（端口 8012；8000/8010/8011 已被其它工单占用）
python -m uvicorn app.main:app --host 127.0.0.1 --port 8012

# 4) 预热模型（**必做**：冷启动加载 15~18 秒，不预热会吃掉首个请求的 3 秒预算）
curl -s http://127.0.0.1:8012/health        # 期望 "models":"warm"

# 5) 打开前端
#    浏览器访问 http://127.0.0.1:8012/    （问答 / 知识图谱 / 优化前后指标 / 上传 PDF）
```

Vue 前端是**单文件 + 本地运行时**，由 FastAPI 直接挂在根路径，**无需 npm、无需单独起服务**。

---

## 环境变量

`.env.example` 含工单要求的 5 个键（`DEEPSEEK_API_KEY` / `MILVUS_HOST` / `REDIS_URL` /
`BGE_M3_PATH` / `BGE_RERANKER_PATH`），外加本单的优化旋钮。

| 键 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 空 | **必填**，只从环境变量读，不进代码 |
| `MILVUS_HOST` | `localhost:19530` | 工单要求的键名；**实际生效的是 `MILVUS_URI=http://localhost:19530`**（沿用工单八口径） |
| `REDIS_URL` | `redis://localhost:6379/0` | 本机 Redis 5.0.14 必须 `REDIS_PROTOCOL=2`（不支持 RESP3） |
| `BGE_M3_PATH` | — | 向量模型，**必须写正斜杠**：`D:/模型/bge-m3` |
| `BGE_RERANKER_PATH` | — | 重排模型：`D:/模型/bge-reranker-v2-m3` |
| `HOPS` / `EXPAND_TERMS` | 2 / 8 | 【检索层】图扩展跳数与注入 BM25 的邻居词数（08 为 1 / 6） |
| `VEC_W` / `BM25_W` | 0.6 / 0.4 | 【检索层】融合权重（08 为 0.7 / 0.3），先通道内归一化再加权 |
| `FUSION_K` / `RERANK_N` | 40 / 6 | 【检索层】融合候选数（08 为 30）/ 送进大模型的片段数（08 为 8） |
| `USE_COMM` | 1 | 【检索层】社区摘要 global 通道开关（消融用） |
| `API_PORT` | 8012 | 服务端口 |

> ⚠ **路径陷阱**：写成 `D:\模型\bge-m3`（工单原文的写法）时 `\b` 会被 dotenv 解析成退格符，加载即报路径不存在。**一律用正斜杠。**

---

## 测试命令

```bash
# 模块导入自检
python -c "import app.config, app.retrieval, app.eval_ragas, app.graph_rag, app.main; print('ok')"

# 图谱自检（节点/边/类型分布；载入 08 的 graph.json）
python -c "import app.graph_rag as G; print(G.stats())"
# 预期：{'nodes': 320, 'edges': 780, 'communities': 0, 'by_type': {'公司': 202, '实体': 55, '人物': 33, '指标': 30}}

# 实体链接自检（单公司应锚定唯一 code；多公司应为空）
python -c "import app.graph_rag as G; print(G.context('平安银行2019年的不良贷款率是多少？')['codes'], \
                                             G.context('中国平安和中国人寿的净利润分别是多少？')['codes'])"
# 预期：['000001'] []

# 端到端问答（走真实链路，别用 Git Bash 的 curl 传中文，会乱码并伪装成检索失败）
python -c "import app.retrieval as R, json; print(json.dumps(R.answer_after('平安银行2019年的不良贷款率是多少？', use_cache=False), ensure_ascii=False)[:400])"

# 评估：先只看检索与时延，不调 RAGAS
python -m app.eval_ragas --sample-pdf data/sample_questions.pdf --skip-ragas

# 正式评估：before（08 原链路）+ after（工单九优化）+ RAGAS 两项指标
python -m app.eval_ragas --sample-pdf data/sample_questions.pdf
```

RAGAS 会消耗 DeepSeek 额度；只跑工单点名的两项指标，逐指标串行（避免混跑触发限流把样本打成 nan）。

---

## 目录结构

```
工单九/
  app/                 # 297 行 Python（含注释与空行，≤ 300 行）
    config.py          #   本单 .env 优先，缺省回落 08 工单 config
    schemas.py         #   HTTP 请求模型（限长校验）
    graph_rag.py       #   图谱层：NetworkX + JSON / DeepSeek 抽三元组 / 社区摘要 / 实体链接与邻域扩展
    retrieval.py       #   检索层 + 生成层：before 与 after 两条链路 / Redis 缓存 / 降级
    eval_ragas.py      #   评估层：题集 PDF 三级降级解析 + RAGAS 两指标 + before/after 对比表
    main.py            #   FastAPI 路由 + 前端挂载
  frontend/            # Vue 单页：问答 / 图谱 / 指标对比 / 上传 PDF
  data/                # 题集 PDF 与图谱落盘（sample_questions.pdf 放这里）
  eval/                # before.json / after.json
  docs/                # 01 设计 ~ 08 待裁定口径提交
  .env.example  requirements.txt  README.md
```

---

## 文档索引

| 文档 | 内容 |
|---|---|
| `docs/01_设计.md` | 问题分析、GraphRAG 优化技术方案、架构图、数据流、四层设计、抽取 prompt 全文 |
| `docs/02_研发.md` | 目录结构、模块职责与复用边界、接口清单、**开发期自检抓出的 5 个缺陷与修复** |
| `docs/03_测试.md` | 环境连通性、中英文问答、**时延实测**、**E1~E4 容错停机实测**、**真实 RAGAS 结果**、判官噪声校准 |
| `docs/04_优化.md` | **优化前后指标对比表**、四层优化点、调参记录、达标判定口径 |
| `docs/05_部署.md` | 启动命令、环境变量、自检清单、排障表 |
| `docs/06_优化方案.md` | **完整优化方案**：诊断 → 目标 → 四层方案 → 实施 → 验证 → 回退与风险 |
| `docs/07_演示视频脚本.md` | **两段视频脚本**：系统功能演示、优化思路（含优化前后指标变化） |
| `docs/08_待裁定口径提交.md` | **提交需求方的口径裁定件**：正文 + 实测结果 + 四条排除证据 + (A)(B)(C) 后果对照与建议 |

---

## 已实测的关键结果（2026-10-11，本机 · 附件 4 题口径）

### RAGAS（判官 `temperature=0`，同一环境跑两条链路）

| 指标 | 目标 | before（08 原链路） | **after（工单九优化）** | Δ | 达标 |
|---|---|---|---|---|---|
| `context_precision` | ≥ 0.80 | 0.7024 | **0.8000** | **+0.0976** | ✅ **达标（卡线）** |
| `context_recall`（综合） | ≥ 0.90 | 0.1875 | **0.5000** | **+0.3125** | ❌ 综合未达标 |
| ↳ 事实题 Q01 / Q02 | ≥ 0.90 | 0.00 / 0.75 | **1.00 / 1.00** | +1.00 / +0.25 | ✅ **均达标** |
| ↳ 综述题 Q03 / Q04 | ≥ 0.90 | 0.00 / 0.00 | **0.00 / 0.00** | 0 | ❌ 结构性，见下 |
| 端到端时延 p50 | ≤ 3000 ms | 2999 | **2678** | −321 | ✅ |
| 端到端时延 max | ≤ 3000 ms | 4189 | 3246 | −943 | ⚠ Q04 单题超 246 ms |

**为什么两道综述题的 recall 是 0**（已排除"检索没找到"）：
① Q03 检索到的片段里「风险管理」出现 5/8、「不良贷款」3/8 ⇒ 金标主题确在上下文中；
② `RERANK_N` 6→12 后 recall **纹丝不动**；
③ **社区摘要开关能把事实题拉到 1.00，却对综述题毫无作用**。
根因是 **Q03/Q04 的参考答案属综述/推断性长文**（如"……确保了银行能够吸收潜在冲击"），
其论断**无法逐句归属到任何原文片段**。**这是题目性质决定的**，调 `HOPS`/`RERANK_N`/权重都无效；
唯一解法是改写参考答案（=改题，工单禁止）。完整证据见 `docs/03_测试.md` §5.7。

**社区摘要（GraphRAG global 通道）的实测贡献**：`USE_COMM` 开/关对比 ——
**recall 0.3375 → 0.5000（+0.1625）**，**precision 逐题完全不变（仍 0.80）**，p50 还从 3169 降到 2678 ms。

### 其他

| 项 | 结果 |
|---|---|
| Milvus `gd8_chunks` | 4879 块（复用 08 索引，未重建） |
| 图谱 | 320 节点 / 780 边 |
| 纯检索时延 p50（不调 LLM） | before 727 ms / after 1005 ms |
| 缓存命中 | 2 ms |
| 中英文问答 | 均通过（英文的图通道边界见 `docs/03_测试.md` §2.2） |
| 容错 E1~E4 | **四条全部停机实测通过** |
| 判官噪声（校准） | 默认温度下 recall 三次极差 **0.1875**；设 `temperature=0` 后降到 **0.0385** |
| 高并发压测 | **未做**（8 G 显存只容一个实例；已在 `docs/03_测试.md` §3.2 如实标注） |

### 最终推荐配置

```ini
HOPS=2            EXPAND_TERMS=8      # 图扩展（公司类邻居名在锚定唯一公司时会被剔除）
VEC_W=0.6         BM25_W=0.4          # 通道内归一化后加权
FUSION_K=40       RERANK_N=6          # 融合候选 40 → 精排后只喂 6 条（精度分母收窄）
USE_COMM=1                            # 社区摘要通道（需先跑 python -m app.graph_rag build 才有内容）
LLM_MAX_TOKENS=300                    # 时延—精度实测拐点：800 超 3 秒；250 掉精度；300 两者兼顾
```

---

## 不做清单（工单九未要求，一律不实现）

用户登录 / 权限 / 后台管理 / 可视化大屏 / 监控告警 / 模型训练微调 / 多租户 / K8s / 支付 /
工单管理 / 复杂运维平台；不替换指定技术选型；不引入 Neo4j 等未指定图数据库（图存储用 **NetworkX + JSON**）。
