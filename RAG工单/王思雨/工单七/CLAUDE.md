# 工单七 · 本单专属说明

> 工单编号：**人工智能NLP-RAG-功能测试及评估**
> 跨工单通用纪律见上级 `../CLAUDE.md`，本文件只写**本单特有的环境、命令、已定结论与交付边界**。

---

## 1. 环境（本机实测值）

| 项 | 值 | 备注 |
|---|---|---|
| 解释器 | `D:/myconda1/python.exe` | **Python 3.12.7**，CUDA 12.6；唯一同时有 CUDA torch + MinerU + ragas 的解释器。工单写 Python 3.11，本机实际 3.12 |
| Milvus | `http://localhost:19530` | 容器 `rag-milvus`（配套 `rag-etcd`/`rag-minio`），**勿再跑 docker compose**（19530/6379 已被占用）。⚠ **前提会失效**：Docker Desktop 重启后这三个容器会全部 `Exited`、19530 消失、pymupdf/pymilvus 报 `Fail connecting to server`。此时用 `docker start rag-etcd rag-minio rag-milvus` 复活即可（数据不丢），**不要 up**。跑评估/演示前先 `docker ps --filter name=rag-` 确认（2026-10-10 实测踩到） |
| 本单集合 | `gd7_chunks` | **独立集合，绝不写工单六的 `gd6_chunks`**；`.env` 从工单六拷贝后必须改这一项 |
| Redis | `redis://localhost:6379/0`，`protocol=2` | 本机 Redis 5.0.14 不支持 RESP3 |
| 向量模型 | `D:/模型/bge-m3`（1024 维） | 归一化 + COSINE |
| 重排模型 | `D:/模型/bge-reranker-v2-m3` | FlagReranker + fp16，输入截断 512 字符 |
| 大模型 | `deepseek-chat` @ `https://api.deepseek.com` | Key 从 `DEEPSEEK_API_KEY` 读，不进代码 |
| API 端口 | **8010** | 8000 被其它服务占用 |
| GPU | RTX 4060 Laptop，8 GB | 重任务必须串行（建索引 / MinerU / RAGAS 不要同时跑） |
| MinerU 补丁 | `C:/mineru_models/langfix` | 子进程调用必须同时设 `MINERU_LANGFIX_DIR` **和** `PYTHONPATH` |

## 2. 常用命令

```bash
D:/myconda1/python.exe scripts/build_index.py        # 建索引（解析→语义分块→Milvus+BM25）
D:/myconda1/python.exe scripts/run_eval.py [N] [--no-ragas]   # 评估
D:/myconda1/python.exe tests/test_pipeline.py         # 自检（失败非 0 退出）
scripts/start.bat                                     # 启动演示 :8010
```

## 3. 语料特有的事实（不要重新踩）

1. **`ccf_competition` 没有 zip**。工单原文写的是《ccf_competition.zip》，实际附件目录 `../RAG 工单/附件/ccf_competition/` 下已是解压好的 `pdf/`（9 份）与 `txt/`（9 份）。按已解压目录处理。
2. **文件名中文已被 `U+FFFD` 破坏且不可逆**（例：`2020-02-14__������__000001__...pdf`）。唯一稳定标识是 6 位股票代码；报告年度由文件名前缀的披露日期减 1 推得（均为 3 月前披露的年报）。
   - 代码→公司名映射写在 `app/config.py::CODE2NAME`。
   - 判断解析是否成功**不能靠文件名匹配**，要靠内容（rglob + 前缀模糊认领）。
3. **PDF 抽取出来的中文词间会插空格**（如「逾期 60 天」）。做关键词核验时必须**先去空格再匹配**，否则得到假阴性「未检出」。
4. 语料 9 份年报共 **2812 页 / 约 298 万字符**。

## 4. 已定结论与交付边界

| 事项 | 结论 | 成立前提 / 触发条件 |
|---|---|---|
| 10 题的构成 | sample 原题 4 + 仿写 6 | 前提：实测 `sample_questions.pdf` 只有 4 个问题（不足 10 个）。**前提若变化（拿到更多示例问题），应改为优先使用原题** |
| 原题 `xx银行` 的还原 | Q01=平安银行、Q02=招商银行 | 依据：「逾越者联盟/咖啡零售/出行预订/影票销售」在 9 份年报中**只**命中 600036；「巩固收入贡献/支柱地位」**只**命中 000001。可核查触发条件：任一命中数变为非唯一即需重判 |
| ground_truth 口径 | 全部逐字取自 PDF 原文并用页码核对 | 前提：不接受任何语料外知识或模型推测 |
| 批量建索引用 pymupdf | `MINERU_ENABLED=0` | 前提：MinerU 对 2812 页耗时过长。**触发条件：若语料换成扫描件（无文本层），必须改回 MinerU=1**，否则 pymupdf 会得到空文本 |
| 300 行口径 | 核心链路 4 模块 = 300 行；`app/main.py`(68 行，FastAPI 路由层) 与 Vue 不计入 | 见 `docs/01-design.md` §4，验收时按此口径核对 |
| 评估判官 | 与生成同用 DeepSeek，存在同源偏差 | 故问题分析以逐题人工复核 + `ctx_meta` 通道溯源为主，RAGAS 分数仅作辅助 |
| 页眉页脚过滤（建库期） | **已实施**：模板噪声 28.7%→6.2%（确定性指标），**但四项 RAGAS 全部落在噪声底内、测不出改善** | 前提：模板片段不与问题语义竞争、大模型本就忽略它们。⚠ 噪声率与过滤器共用"行首/页边"判据，**该指标下降部分是"按定义"必然的**，不得单独当优化成效。见 `docs/04-optimize.md` §1.4 |
| 噪声底 | **0.027**（`EVAL_MAX_WORKERS=2` + 逐指标串行，同输入重打分；四指标 10/10 有效） | 前提：本机 + deepseek-chat + 并发 2。**并发越高噪声底越大；每次要比较指标前必须重新量**（`scripts/measure_noise.py`） |
| 融合权重 | **不是有效杠杆**：0.7/0.3~0.3/0.7 四组，首条重排分恒 0.868、文档命中恒 9/9 | 前提：链路末端有 bge-reranker-v2-m3，重排把融合次序差异抹平。见 `docs/04-optimize.md` §2.3 |
| 主要剩余短板 | **同名校实体混淆**：问平安银行(000001) 时被中国平安(601318) 年报片段挤占，目标公司占比仅 0.12~0.25 | 前提：语料同时含母公司与子公司、且母公司大段披露子公司数据。可核查触发条件：任一单文档题占比 < 0.5 需复检 |

## 5. 本单不做（属 01–06 或其它工单）

Query 改写、表格结构化重建、图像/图表解析、Graph+RAG、多轮会话、反馈票箱。
