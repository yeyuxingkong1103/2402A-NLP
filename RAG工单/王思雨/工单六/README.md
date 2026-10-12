# 人工智能NLP-RAG-混合检索任务（工单六）

> 工单编号：**人工智能NLP-RAG-混合检索任务**
> 对应工单：《人工智能NLP-RAG项目-06-混合检索任务工单V1.1-20250206.pdf》
> 项目根目录：`d:\Users\雨子\Desktop\专高六\liu-专高六\RAG\工单六`

基于两份招股说明书的**向量检索（召回+重排）+ 全文检索（BM25）+ 混合检索**问答系统，
带 Vue 交互页面、Redis 缓存与多轮对话、用户反馈驱动的自适应重排，以及 RAGAS 评估。

---

## 一、快速开始（Windows）

```bash
cd "d:/Users/雨子/Desktop/专高六/liu-专高六/RAG/工单六"

# 1) 装依赖（本机唯一齐备的解释器已带全部依赖，通常可跳过）
D:/myconda1/python.exe -m pip install -r requirements.txt

# 2) 配置密钥
cp .env.example .env      # 然后编辑 .env，填入 DEEPSEEK_API_KEY

# 3) 启动（Milvus 与 Redis 本机已在运行，无需再起 compose）
scripts\start.bat                                    # 或 bash scripts/start.sh
# 等价于： D:/myconda1/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010

# 4) 浏览器打开
#    http://127.0.0.1:8010/
```

首次入库（若 `data/` 下没有索引产物）：

```bash
D:/myconda1/python.exe -m app.ingest      # 解析 PDF → 分块 → 建 BM25 → 写 Milvus
```

## 二、验收自测

```bash
D:/myconda1/python.exe tests/test_api.py            # 接口冒烟，23 个用例
D:/myconda1/python.exe tests/test_mineru_smoke.py   # MinerU 解析路径停机实测
D:/myconda1/python.exe tests/compare_modes.py       # 检索层精度对比（向量/全文/混合）
D:/myconda1/python.exe tests/validate_questions.py  # 【强制】自造题集准入闸门：题面成立 + 答案可核验
D:/myconda1/python.exe -m app.eval_ragas            # RAGAS 端到端评估（约 20~25 分钟）
```

## 三、代码行数统计

工单要求「所有 .py 总行数 ≤ 300 行（不含空行）」。统计命令与结果：

```bash
$ find . -name "*.py" -not -path "*__pycache__*" -exec grep -vh '^[[:space:]]*$' {} \; | wc -l
1621
```

| 文件 | 非空行数 |
|---|---|
| `app/__init__.py` | 2 |
| `app/config.py` | 81 |
| `app/eval_ragas.py` | 156 |
| `app/ingest.py` | 330 |
| `app/main.py` | 123 |
| `app/rag_core.py` | 520 |
| `tests/compare_modes.py` | 62 |
| `tests/eval_recall_expanded.py` | 76 |
| `tests/test_api.py` | 106 |
| `tests/test_mineru_smoke.py` | 57 |
| `tests/validate_questions.py` | 108 |
| **合计** | **1621** |

| **合计** | **0** |

**如实说明：实际 1621 行，超出 300 行上限。** 原因与取舍见 `docs/02-dev.md` 第 3 节，这里给结论：

- 本项目采用**行尾注释**（工单要求"每行代码末尾写注释"），实测**纯注释行只有 82 行**，其余 ~1574 行全是"语句 + 行尾注释"。把注释全删掉也只减少约 4%——行数**不是**被注释撑大的。
- 工单明确要求的功能集合（3 种重排算法、加权平均+投票两种以上融合、布尔/短语/模糊查询、多字段检索、语义分块、RAGAS 评估、7 个接口）**即使完全不写注释也需要约 1080 行**。
- **"300 行不可行"有三条互相独立的证据**：① 无注释的裸代码下限 ≈1080 行（工单点名的功能本身就装不下）；
  ② 删光注释只减约 4%（行数不是被注释撑大的）；③ 工单没要求但质量必需的**准入闸门 +108 行**。
  三者都不是"写得啰嗦"，而是同一结论的三条独立证据。
- 要压到 300 行，只能砍掉约 3/4 功能，那会直接违反验收标准。
- 已在压缩上做的实事：所有可调参数收进 `.env`（40 余项）、题集放 JSON、重排器/融合用注册表分派、能用库就不手写、前端不占 Python 行数。

## 四、五个阶段的产物

| 阶段 | 产物 | 文档 |
|---|---|---|
| 一、设计 | 架构、数据流、API、Milvus Schema、分块策略、融合公式、评估方案 | [`docs/01-design.md`](docs/01-design.md) |
| 二、研发 | 全部代码、前端、配置、docker-compose、requirements | [`docs/02-dev.md`](docs/02-dev.md) |
| 三、测试 | 测试脚本与测试报告 | [`docs/03-test.md`](docs/03-test.md) |
| 四、优化 | 缓存/调参/延迟优化的前后对比 | [`docs/04-optimize.md`](docs/04-optimize.md) |
| 五、部署 | 部署文档、启动脚本、演示视频脚本 | [`docs/05-deploy.md`](docs/05-deploy.md) |
| 演示 | 视频脚本 + 实际问答集 + 自动截图 | [`docs/demo-video-script.md`](docs/demo-video-script.md)、[`docs/demo-qa.md`](docs/demo-qa.md)、`docs/screenshots/` |
| 专题 | 陷阱专文：便宜预筛级会系统性杀掉金标 | [`docs/pitfall-tfidf-prescreen.md`](docs/pitfall-tfidf-prescreen.md) |
| 专题 | 验收口径对齐（召回率分层上报 vs 整体 ≥95%） | [`docs/acceptance-recall-alignment.md`](docs/acceptance-recall-alignment.md) |

## 五、API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 逐组件健康检查（Milvus / Redis / 大模型密钥 / 索引块数），降级原因如实列出 |
| GET | `/config` | 返回可用检索模式、融合算法、重排器、嵌入模型与默认参数（驱动前端控件） |
| POST | `/search` | 执行向量 / 全文 / 混合检索，返回片段、出处、命中通道与两路名次 |
| POST | `/chat` | 多轮问答，支持中英文，返回答案、引用来源、检索问句与降级信息 |
| POST | `/feedback` | 用户👍/👎写入 Redis，供自适应重排器调整片段/文档权重 |
| POST | `/ingest` | 解析 PDF → 分块 → 建 BM25 → 写 Milvus |
| POST | `/eval` | 运行 RAGAS 评估并落盘报告 |

## 六、技术选型（严格按要求）

| 环节 | 选型 |
|---|---|
| 语言 / 界面 | Python 3.12（本机唯一齐备解释器，工单要求 3.11，说明见 `docs/05-deploy.md` §1） / FastAPI + Vue 3 |
| PDF 解析 | **MinerU 主**，pymupdf → pdfplumber 兜底 |
| 分块 | 语义分块（句向量余弦 + 阈值切块 + 长度硬约束 + 小块合并） |
| 向量模型 | **bge-m3**（本地 `D:/模型/bge-m3`），1024 维 |
| 向量库 | **Milvus**（`gd6_chunks`，IVF_FLAT / IP） |
| 关键词检索 | **BM25**（`rank_bm25`，标题/正文/摘要三字段） |
| 重排 | **bge-reranker-v2-m3**（本地 `D:/模型/bge-reranker-v2-m3`）+ TF-IDF + 用户反馈自适应 |
| 大模型 | **DeepSeek API**（`DEEPSEEK_API_KEY` 从环境变量读取） |
| 缓存 | **Redis**（答案/检索两级缓存 + 会话历史 + 反馈票箱） |
| 评估 | **RAGAS**（faithfulness / answer_relevancy / context_precision / context_recall） |

## 七、关键实测结论（详见 `docs/03-test.md`、`docs/04-optimize.md`）

- 接口冒烟 **23/23 PASS**；MinerU 与兜底解析链路均已停机实测通过。
- **重排消融（全篇最硬的一条）**：混合检索 k=8 召回 **1.0000**（开 bge 重排）→ **0.8696**（关重排）；
  top-4 时 1.0000 → 0.6957，且低于同时关重排的纯向量（0.8261）。**重排不是装饰，是精度达标的前提。**
- 检索层精度：混合 **1.0000** > 纯向量 0.9565 ≈ 纯 BM25 0.9565。
- 测试中**被测试抓出并修掉的 4 个 bug**（`docs/03-test.md` §2.1）：用户 topK 被误当融合池大小而挤出金标；
  级联预筛腰斩金标（专题：[`docs/pitfall-tfidf-prescreen.md`](docs/pitfall-tfidf-prescreen.md)）；
  RAGAS 四指标混跑把 17/23 样本静默打成 nan；DeepSeek 客户端每请求新建多付约 1 秒 TLS。
- 端到端准确率与响应时间分布：见优化报告的「优化前后对比」。
- **召回率不达标且分类型差异极大**（扩样 114 题实测，见 `docs/03-test.md` §5.5）：
  单事实类问题 **0.91~1.00**；**多期数据聚合类只有 0.5909**（仅计有效题 0.6579；确定性召回与 RAGAS 召回
  **完全相等**，说明是真实检索短板而非判官噪声）。**本报告不声称召回率达标**，也不给出"整体召回率"这种单点数字。
- 多期聚合类**已做补召回最小验证并否证**：相邻块续接 +1/19、表格定向补召 +0/44，净增为零，
  机制是**表头与数值行被切进不同的块** ⇒ 缺的不是补召回，是分块策略（属独立工作量，本工单未做）。

## 八、本机环境备忘

- 解释器：`D:/myconda1/python.exe`（Python 3.12.7，CUDA 12.6）
- Milvus：容器 `rag-milvus`，`localhost:19530` 已在运行；Redis：`localhost:6379`（服务端 5.0.14）
- 服务端口用 **8010**（8000 已被本机其它服务占用）
- 两份招股说明书：`data/raw/招股说明书1.pdf`（武汉兴图新科，科创板，548 页）、`招股说明书2.pdf`（武汉力源信息，创业板）
