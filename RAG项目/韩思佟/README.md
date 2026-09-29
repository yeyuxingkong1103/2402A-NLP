# 基于RAG的医生健康问答系统

先检索医学指南，再由DeepSeek结合资料生成回答。项目在本地Ubuntu/WSL运行；MySQL保存账号和医生角色，Redis保存近期对话，Milvus保存医学知识向量。无需开启AutoDL GPU。

## 先看四个答辩文件

1. `app/offline_pipeline.py`：一个294行离线文件，完成PDF解析、清洗、分块、BGE向量化和Milvus同步。
2. `app/single_app.py`：一个299行在线核心文件，完成记忆、混合检索、重排序、提示词、DeepSeek生成、后处理和API；账号等工程路由由内部辅助文件安装。
3. `app/evaluate.py`：一个255行评测文件，完成检索指标和正式RAGAS评测。
4. `app/pressure_test.py`：一个201行压力测试文件，只读取JMeter的完整RAG结果，计算成功率、QPS及延迟分位数。

四个文件分别对应“准备知识、回答问题、验证效果、测试并发”。每个文件都不超过300行并带中文注释；`app/internal/*`和旧`scripts/*`保存兼容与工程细节，程序会调用，但不属于答辩逐行讲解范围。

零基础学习顺序：先看`docs/20-四文件答辩笔记.md`记住流程，再打开`http://127.0.0.1:8000/learn`按27个业务代码块学习，最后按`docs/21-四文件代码答辩范围.md`练习45分钟答辩。逐行查询只在老师随机点行时使用。

现场要在 PyCharm 按问题找代码、展示四份 PDF 解析 TXT 或打开 Redis/Milvus 可视化时，直接看 `docs/25-答辩现场打开什么.md`。四份解析 TXT 使用原 PDF 前 3 页节选；可视化服务用 `bash deploy/local/start.sh --visual` 启动。

答辩现场不要朗读全部代码。按`docs/21-四文件代码答辩范围.md`中的“必须展开/一句话带过/不主动展开”准备；老师随机点到其他行时，再用源码中文注释解释。

## 本次采用的方案

| 环节 | 方案 |
|---|---|
| 角色 | 一个医生，健康科普，不替代面诊 |
| 资料 | 现有《国家基层高血压防治管理指南2020版》，既有约89块，以实时统计为准 |
| 解析 | PyMuPDF、PDFPlumber、PaddleOCR、MinerU均已真实运行；四份前三页节选结果在`data/parser_samples/` |
| 向量 | bge-small-zh-v1.5，512维；当前不是BGE-M3 |
| 召回 | Milvus语义检索 + Python BM25关键词检索 |
| 融合/精排 | RRF融合 + 可配置的BGE CrossEncoder精排；本轮已补齐6个模型文件并核对清单哈希，实际执行状态看rerank_state和评测报告 |
| 对话 | Redis近20条消息；有历史时改写独立检索问题 |
| 生成 | DeepSeek在线API，模型名和Key由后端配置提供 |
| 网页/API | FastAPI提供网页、注册登录、问答、来源、历史 |
| 评测 | `app/evaluate.py`统一提供检索指标、正式RAGAS和重排前后对照 |

## 使用 Git 提交

项目已配置 `.gitignore`，不会上传 API Key、1.27 GB 本地模型、本地数据库或 JMeter 安装包。第一次提交和老师要求的 `clone/add/commit/push/pull/diff/branch` 命令说明见 `docs/23-Git命令提交指南.md`。

当前项目目录第一次提交的核心顺序是：

```powershell
cd D:\rag-roleplay
git init
git add .
git diff --cached --stat
git commit -m "提交基于RAG的医生问答项目"
git branch -M main
git remote add origin 你的远程仓库地址
git push -u origin main
```

## 启动

Windows目录是`D:\rag-roleplay`，Ubuntu目录是`/mnt/d/rag-roleplay`。在Ubuntu终端执行：

```bash
cd /mnt/d/rag-roleplay
bash deploy/local/start.sh
```

首次准备环境才需要执行`bash deploy/local/setup.sh`。DeepSeek Key由本人填写到后端`.env.local`，不放在网页或源码中。

启动后访问`http://127.0.0.1:8000/chat`。不要直接双击HTML文件。停止：

```bash
bash deploy/local/stop.sh
```

分块学习页：`http://127.0.0.1:8000/learn`。默认按“目的→输入→处理→输出→失败处理→答辩话术”学习，也保留逐行查询备用。项目尚未启动时可直接查看文档中的逐行解释。

## 离线准备、测试和评测

新增或替换PDF时，在Ubuntu中按顺序运行；平时聊天不重复执行：

```bash
cd /mnt/d/rag-roleplay
bash deploy/local/setup_parsers.sh       # 首次安装解析器时执行
bash deploy/local/parse_documents.sh     # 解析、清洗、分块
/home/lenovo/.venvs/rag-roleplay/bin/python -m app.offline_pipeline embed
/home/lenovo/.venvs/rag-roleplay/bin/python -m app.offline_pipeline sync --dry-run
/home/lenovo/.venvs/rag-roleplay/bin/python -m app.offline_pipeline sync
```

解析器安装在独立环境中，`setup_parsers.sh`只需首次执行；`parse_documents.sh`实际解析、清洗和分块。`embed`用核心环境中的本地BGE生成向量；`sync`首次建库或按“来源＋块号”稳定更新Milvus。同步后重启在线服务，让内存BM25读取新知识。

`embed`会原子写入本轮JSONL并生成`data/embeddings/snapshot-manifest.json`，不会自动删除目录中的旧向量文件；普通`sync`会保留并报告未列入清单的旧文件。需要删除Milvus中过期知识时，必须先完成全量重建，并同时执行`sync --prune --snapshot-manifest data/embeddings/snapshot-manifest.json`；文件名、目录清单或SHA256有任何不一致都会拒绝删除。

先运行不产生DeepSeek裁判费用的检查：

```bash
cd /mnt/d/rag-roleplay
/home/lenovo/.venvs/rag-roleplay/bin/python -m unittest discover -s tests -v
/home/lenovo/.venvs/rag-roleplay/bin/python -m app.evaluate retrieval --compare-rerank
bash deploy/local/evaluate.sh --setup     # 首次创建固定版本评测环境
bash deploy/local/evaluate.sh --limit 8 --compare-rerank
```

`retrieval`只测检索，不生成回答、不调用裁判。`ragas`不带`--run`时只校验数据和环境，不产生RAGAS分数，默认写入`outputs/ragas/input_validation.*`，不会覆盖真实评测报告。完整RAGAS需要可访问的DeepSeek API、评测依赖和完整reranker；先小批量运行：

```bash
bash deploy/local/evaluate.sh --run --limit 2 --compare-rerank
```

只有带`evaluate.sh --run`的命令才真实生成回答并调用RAGAS裁判，会消耗API额度。报告在`outputs/ragas/generation_report.json`和`.md`；检索报告在`outputs/retrieval_report.json`和`.md`。检索初始化或逐题失败写入`outputs/retrieval_report_failed.*`，不会覆盖最近一次成功报告。开启精排对照时必须真实加载reranker，否则明确失败；越界题单列，失败不会伪装成0分。`bash deploy/local/finish_rag.sh`默认跳过付费RAGAS，只有显式增加`--run-paid`才执行在线评测。

课程要求的完整RAG问答使用`docs/verification/rag-chat-load.jmx`由JMeter施压。计划会为每个线程注册独立用户，动态取得医生角色，每轮清空历史后请求`POST /api/chat`，并断言HTTP 200、回答和来源非空、医生角色正确、`rerank_state=ready`且来源含精排分。已完成并发1/2/4阶梯实测；重复生成汇总报告只读取JTL，不会再次调用DeepSeek：

```bash
python -m app.pressure_test --jtl \
  outputs/jmeter/c1/results.jtl \
  outputs/jmeter/c2/results.jtl \
  outputs/jmeter/c4/results.jtl \
  --output outputs/jmeter/rag_chat_report
```

| 并发 | 正式样本 | 成功/失败 | 错误率 | 成功问答QPS | 平均延迟 | P95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 / 0 | 0% | 0.187 | 5.274 s | 5.557 s |
| 2 | 6 | 6 / 0 | 0% | 0.318 | 6.082 s | 6.837 s |
| 4 | 12 | 12 / 0 | 0% | 0.520 | 7.489 s | 8.385 s |

21次正式问答全部通过断言。该测试覆盖Redis读写、BGE问题向量化、Milvus与BM25召回、RRF融合、BGE精排、DeepSeek生成和后处理。它是低成本小样本基线；每轮清空历史，因此不含多轮追问的Query改写。原始JTL和报告保存在`outputs/jmeter/`。

## 交付材料

- `docs/12-需求覆盖与交付边界.md`：完整老师清单，明确已采用、备选、待实测。
- `docs/13-STAR答辩讲稿.md`：45分钟顺序和可练习话术。
- `docs/14-设计流程与答辩细节.md`：流程图、思维导图、痛点到代码到证据。
- `docs/15-部署与验收清单.md`：真实服务、模型、网页、RAGAS验收方法。
- `docs/16-核心代码逐行讲解.md`：299行在线核心的逐行解释，附语法词典。
- `docs/17-RAGAS评测与重排序对照.md`：四项指标、成本范围、失败处理和对照方法。
- `docs/18-知识库更新操作.md`：稳定主键更新、预览变更、可选清理过期块。
- `docs/20-四文件答辩笔记.md`：四个答辩文件、优化前后数据、压力测试与STAR话术。
- `docs/verification`：Postman集合、JMeter角色接口与完整RAG问答计划。
- `outputs/screenshots/chat-running.png`、`outputs/screenshots/learn-running.png`：本轮真实运行页面截图。
- `http://127.0.0.1:8000/docs`：与运行版本一致的接口文档。

## 必须如实介绍的边界

- 当前主要知识资料是高血压指南，其他医学问题可能仅得到一般健康信息，不表示有对应指南支持。
- 本轮已补齐BGE-reranker-base，6个文件大小和SHA-256与下载清单一致；文件完整性与实际推理效果是两项验收。BGE-M3仍未启用，当前embedding使用BGE-small。
- 普通PaddleOCR不等于PaddleOCR-VL；解析代码存在不等于所有文档类型都已实测。
- 旧`outputs/ragas_retrieval_report.json`是自算检索报告，记录过RAGAS导入失败，不是正式RAGAS生成质量结果。
- 当前Milvus存知识库，没有实现用户长期记忆；新增资料使用`app/offline_pipeline.py`的`prepare → embed → sync`更新，重启后刷新BM25；不是网页自动上传更新。
- 当前以本地课堂演示为目标；JWT/HTTPS、公网生产授权、Nginx负载均衡和多副本部署不在本次主路径。
- 本地服务仍会把问题、近期对话和检索片段发送给DeepSeek，因此生成阶段需要联网和API Key。

已有真实RAGAS报告由旧评测入口生成，记录的在线核心SHA-256为`762e9ec...`；当前299行教学核心SHA-256为`CF483E181EF3354F5C7414708F5A0A149E7AEB1426A4A75CB28D8FCF833A2D0B`。旧报告之后修复了多轮Query改写，并把非RAG工程路由下沉以简化讲解。在用当前代码重新执行`bash deploy/local/evaluate.sh --run`前，必须把旧结果标明为历史结果，不能冒充当前哈希重跑结果。最终测试数量、精排状态和真机阻碍以验收记录为准。
