# 项目进度看板（PROGRESS）

> 本文件保留开发过程，后半部分包含已经被替换的旧架构，只用于回顾。当前答辩主线只看`README.md`、`app/offline_pipeline.py`、`app/single_app.py`、`app/evaluate.py`和`docs/13-STAR答辩讲稿.md`。

## 2026-09-22 当前方案：本地 RAG + DeepSeek 在线生成

- 当前主架构：Ubuntu/WSL2 本地运行 FastAPI、BGE、MySQL、Redis、Milvus、BM25/RRF；DeepSeek 在线 API 负责 Query 改写和最终回答。
- 当前模型：`deepseek-v4-pro`，关闭 thinking；无需 GPU、AutoDL 或本地 Qwen。
- 网页/API：`http://127.0.0.1:8000/chat`；唯一角色为通用医学健康科普医生。
- 最终300行核心已完成两轮真实问答验收；首问与追问都返回4条指南来源，追问成功补全“血压”主题，精排状态为`ready`。
- 在完整依赖环境中85项自动测试通过；在线核心`app/single_app.py`严格300行，离线与评测分别集中在另外两个主文件。
- 正式服务重启：在 Ubuntu 执行 `cd /mnt/d/rag-roleplay && bash deploy/local/start.sh`。
- 当前操作清单：`docs/09-开卡执行清单.md`；部署和答辩讲解：`docs/10-本地Ubuntu部署与答辩讲解.md`。

## 2026-09-20 历史记录：Ubuntu 本地数据服务落地

- 架构改为：Ubuntu/WSL2 运行 FastAPI、MySQL、Redis、Milvus 和本地 BGE；AutoDL 只运行 Qwen，并通过 SSH 隧道连接。
- 已确认 Ubuntu 虚拟磁盘位于 `D:\AI_Friend\wsl\Ubuntu-24.04`，Docker Desktop 数据也位于 D 盘；数据库使用 Linux 命名卷。
- 真实验收通过：MySQL、Redis、Milvus 均为 `ok`，Milvus 有 89 个唯一向量、512 维。
- 真实检索通过：本地 BGE + Milvus + BM25 + RRF 返回 4 条高血压指南资料。
- 本地网页/API 已运行在 `http://127.0.0.1:8000/chat`，角色接口只返回医生。
- 14 项自动测试通过；`app/`、`scripts/`、`deploy/`、`tests/` 每个源文件不超过 300 行。
- Ubuntu Python 环境位于 `/home/lenovo/.venvs/rag-roleplay`；首次依赖安装已完成。
- AutoDL 纯模型更新包：`outputs/rag-autodl-model-only.zip`。真实 Qwen + SSH 隧道问答仍需下次开卡验证。
- 完整启动和答辩讲解：`docs/10-本地Ubuntu部署与答辩讲解.md`。

## 2026-09-16 网页演示更新

- AutoDL 已验证：base 环境依赖、4 项原有单元测试、API 首页、4 个角色、接口文档和真实高血压知识库检索。
- 本地新增 `/chat` 聊天网页：注册登录、医生问答、连续提问、资料原文展开、清空对话记忆。
- 按最新要求仅开放医生服务，移除其他角色入口及自定义角色创建接口；已有其他角色记录不删除，但不再对外提供服务。
- 新增 `deploy/autodl/start_demo.sh`，复用 base 与 vllm 两个环境，无额外安装；网页端口 6006，模型端口 8001，检索走 CPU。
- 本地 10 项测试通过，覆盖网页与账号、模型错误、单医生限制、检索逻辑和模型文件预检；JavaScript 与 Bash 语法检查通过。
- 浏览器实测临时账号登录、提问、回答显示、参考资料原文；检索使用真实 BGE 和知识库副本，大模型使用明确标记的模拟响应。验收脚本连续两次通过，不代表真实 Qwen 已运行。
- 最新上传包为 `outputs/rag-web-ready.zip`；完整指令见 `docs/09-开卡执行清单.md`。无卡时先 `start_demo.sh --check`，开卡后启动脚本会自动验收真实问答，再显示 READY。
- 尚待云端验证：84GB GPU 克隆实例上的 Qwen 启动及网页真实问答。当前网页不是已经完成云端大模型联调的证明。
- 模型目录名为 `Qwen3.8-27b`，已读取配置架构为 `Qwen3_5ForConditionalGeneration`；实际参数量仍以模型配置为准。

> 你的作战地图。每次开工先看这里，每完成一项就打勾，并在"阿健的收获"栏用一句话写下你学会了什么——答辩前的底气全在这。

## 项目位置

```
D:\rag-roleplay
```

所有命令都在这个目录下执行：

```bash
Set-Location 'D:\rag-roleplay'
```

## 一、协作分工（谁干什么）

| Buddy（我）负责 | 阿健（你）负责 |
|---|---|
| 查资料、写代码框架、逐行讲解 | 拍板决策（选什么、改成什么样） |
| 在演示环境先跑通、踩坑排错 | 在自己电脑上**亲手跑一遍** |
| 文档和代码初稿 | 读、提问、要求修改到你能讲为止 |
| 答"为什么这么写" | 能**复述**每一步在干什么（答辩检测标准） |

一句话：**我搭台，你唱戏。**

---

## 二、总路线（P0 ~ P6）

| 阶段 | 内容 | 状态 | 阿健的收获（自己填） |
|---|---|---|---|
| P0 | 需求分析：需求规格说明书 / 思维导图 / 业务流程图 | ✅ 三份初稿 | |
| P1 | 建库：指南 PDF → 解析 → 分块 → 向量化 → Milvus | ✅ **89 块入库**（你亲手跑通） | |
| P2 | 问答闭环：检索 → 提示词模板 → LLM → 后处理 | ✅ **闭环打通**（你亲手跑通第一问） | |
| P3 | 记忆：Redis 短期 + 多轮对话 + Query 改写 | ✅ **改写生效** | |
| P4 | 服务化：FastAPI 三层次架构 + 8 个接口 | ✅ **接口全通**（/docs 已验收） | |
| P5 | 优化（混合检索/重排）+ RAGAS 评测 | ⏳ 混合检索已接入；重排代码已接入但本地模型未下载完整；RAGAS 检索评测已产出 | |
| P6 | 测试（unittest/Postman/JMeter）+ Ubuntu 部署 | ⏳ unittest 已开始；Postman/JMeter/部署待做 | |

**交付物进度**

| 交付物 | 状态 |
|---|---|
| 需求规格说明书 | ✅ `docs/01-需求规格说明书.md` |
| 功能思维导图 | ✅ `docs/02-功能思维导图.html` |
| 业务流程图 | ✅ `docs/03-业务流程图.svg` |
| 接口文档 | ✅ `docs/04-接口文档.md` |
| **项目交接文档** | ✅ `docs/05-项目交接文档.md` |
| 接手后修复对照表 | ✅ `docs/06-接手后修复对照表.md` |
| AutoDL 本地大模型部署文档 | ✅ `docs/07-AutoDL本地大模型部署.md` |
| 设计文档（架构/功能设计） | ⬜ |
| RAGAS 评测报告 | ⏳ `outputs/ragas_retrieval_report.md`（检索评测已完成，生成类指标待补） |
| 测试：unittest / Postman / JMeter | ⬜ |
| 部署：Ubuntu shell 脚本 + README | ⏳ `README.md` 与 `requirements.txt` 已补；Ubuntu 脚本待做 |

---

## 2026-09-16：Codex 接手后完成的小目标 1

**目标：把在线 RAG 检索从“纯向量 TopK”升级到“混合检索 + 角色路由 + 低相关过滤”。**

已完成：

1. `app/rag.py` 接入 BM25，和向量召回结果用 RRF（倒数排名融合）合并。
2. `/api/chat` 会把角色名传给 RAG 引擎：医生角色使用高血压知识库；律师、朋友、心理医生暂不误用医生知识库。
3. 增加低相关过滤：像“劳动合同怎么解除”这种无 BM25 命中且向量相似度低的问题，不再返回高血压资料。
4. 新增 `tests/test_rag_retrieval.py`，可用 unittest 验证检索融合、低相关过滤、非医生角色不返回医生资料。

验收命令：

```bash
Set-Location 'D:\rag-roleplay'
set PYTHONIOENCODING=utf-8
C:\Users\lenovo\anaconda3\python.exe -m unittest tests.test_rag_retrieval
```

学习点：

- 向量检索：看“语义像不像”。
- BM25：看“关键词有没有命中”，适合数字、药名、法律条文号等精确词。
- RRF：不直接相加不同算法的分数，而是按两路排名融合，适合快速做混合检索。
- 角色路由：不是所有角色都应该查同一个知识库，先保证医生只查医学库，其他角色等知识库建好后再接入。

---

## 2026-09-16：Codex 接手后完成的小目标 2

**目标：接入 BGE-rerank 精排层，并保证模型没下载完整时服务不崩。**

已完成：

1. `app/rag.py` 增加可选精排层：混合检索先召回候选块，再交给 `CrossEncoder` 精排。
2. 支持环境变量：
   - `RAG_RERANK_ENABLED=auto/true/false`
   - `RAG_RERANK_MODEL=D:\rag-roleplay\models\bge-reranker-base`
   - `RAG_RERANK_N=10`
3. 当前 `models\bge-reranker-base` 缺少 `config.json`，说明模型未下载完整；代码会打印“RAG 精排未启用”，然后自动使用混合检索结果。
4. `sources` 返回字段增加 `rerank_score`；未启用精排时为 `null`，启用后为精排模型分数。
5. `tests/test_rag_retrieval.py` 新增测试：验证精排层可以改变候选排序。
6. 新增 `.env.example`，记录运行时配置模板，不包含真实密钥。

验收命令：

```bash
Set-Location 'D:\rag-roleplay'
set PYTHONIOENCODING=utf-8
C:\Users\lenovo\anaconda3\python.exe -m unittest tests.test_rag_retrieval
```

学习点：

- 召回：先多找一些“可能相关”的资料，宁可多一点。
- 精排：再让 rerank 模型逐条判断“问题和资料到底配不配”，把最相关的排前面。
- 降级：工程里不能因为增强组件缺失就让主流程崩掉；精排没启用时，混合检索仍可工作。

---

## 2026-09-16：Codex 接手后完成的小目标 3

**目标：产出 RAGAS 检索评测报告。**

已完成：

1. 新增 `data/eval/ragas_retrieval_cases.json`，包含 7 个评测问题、标准答案和人工标注的相关资料块。
2. 检索评测现统一在 `app/evaluate.py`，可复跑且不调用大模型API。
3. 生成 `outputs/ragas_retrieval_report.json` 和 `outputs/ragas_retrieval_report.md`。
4. 指标结果：Hit@4 = 1.0000，Recall@4 = 0.7071，Precision@4 = 0.5714，MRR = 1.0000。
5. 当前 `ragas==0.4.3` 与本机 LangChain 版本存在导入兼容问题：`No module named 'langchain_community.chat_models.vertexai'`。因此本次先交付检索评测，faithfulness、answer relevancy 等生成类指标待依赖修复后补充。

验收命令：

```bash
Set-Location 'D:\rag-roleplay'
set PYTHONIOENCODING=utf-8
C:\Users\lenovo\anaconda3\python.exe -m app.evaluate retrieval
```

学习点：

- RAG 评测要拆成“检索质量”和“生成质量”两部分看。
- Hit@K 看有没有命中，Recall@K 看标准资料召回多少，Precision@K 看返回资料里有多少是真相关，MRR 看相关资料排得靠不靠前。
- 当前检索的强项是首个相关块很靠前；弱项是常用药、转诊、急症处理这类跨多个块的问题，Top4 不能召回所有相关块。

---

## 2026-09-16：Codex 接手后完成的小目标 4

**目标：补齐项目复现基础文档。**

已完成：

1. 新增 `requirements.txt`，按当前可运行环境锁定核心依赖版本。
2. 新增 `README.md`，说明项目结构、环境准备、启动服务、测试、评测、当前能力和已知限制。
3. README 使用 PowerShell 命令，避免 `cd /d` 这类 cmd 写法在 PowerShell 中报错。
4. 文档未写入真实 `.env` 密钥，只引用 `.env.example`。

验收命令：

```bash
Set-Location 'D:\rag-roleplay'
C:\Users\lenovo\anaconda3\python.exe -m unittest tests.test_rag_retrieval
C:\Users\lenovo\anaconda3\python.exe -m app.evaluate retrieval
```

学习点：

- `requirements.txt` 解决“别人怎么装环境”。
- `README.md` 解决“别人怎么跑项目、怎么验收”。
- 部署前先把本地复现路径写清楚，后面的 Ubuntu 脚本才有依据。

---

## 2026-09-16：Codex 接手后完成的小目标 5

**目标：对照 WorkBuddy 的交接风险，补安全保护和当前状态说明。**

已完成：

1. 新增 `.gitignore`，防止 `.env`、实验库、缓存、临时下载文件误提交。
2. 新增 `docs/06-接手后修复对照表.md`，逐项说明 WorkBuddy 提到的问题哪些已解决、哪些仍待做。
3. README 增加接手后修复对照说明。

学习点：

- `.env` 可以留在本地运行，但不能进入版本管理。
- 交接文档会过时，所以要有一张“当前状态对照表”承接旧问题。
- 技术债不要只写“待完善”，要写清楚影响、状态和下一步。

---

## 2026-09-16：Codex 接手后完成的小目标 6

**目标：按最终 AutoDL 本地 Qwen 27B 部署方向补部署模板。**

已完成：

1. 新增 `docs/07-AutoDL本地大模型部署.md`，说明 vLLM 本地模型服务和 RAG API 的连接方式。
2. 新增 `deploy/autodl/env.example`，配置 Qwen 模型路径、vLLM 端口、RAG API 端口、`LLM_BASE_URL` 和 `LLM_MODEL`。
3. 新增 `deploy/autodl/start_llm.sh`，用于在 AutoDL 上启动 vLLM OpenAI 兼容服务。
4. 新增 `deploy/autodl/start_api.sh`，用于启动本项目 FastAPI 服务。
5. 新增 `deploy/autodl/check.sh` 和 `deploy/autodl/stop.sh`，用于检查和停止服务。

学习点：

- 本项目已经通过 `LLM_BASE_URL` 和 `LLM_MODEL` 把“大模型供应商”抽象成 OpenAI 兼容接口。
- 在线 API 和本地 Qwen 的切换，本质是改环境变量，不需要重写 RAG 业务代码。
- AutoDL 部署时先启动 vLLM，再启动 RAG API；RAG API 访问 `http://127.0.0.1:8001/v1`。

---

## 三、架构长什么样（答辩要能画出来）

```
        ┌──────────────────────────────────────────┐
        │  浏览器 / Postman / 前端                  │
        └───────────────────┬──────────────────────┘
                            │ HTTP (JSON)
        ┌───────────────────▼──────────────────────┐
        │  app/main.py   接口层（FastAPI，8 个接口）│
        └───────────────────┬──────────────────────┘
                            │
        ┌───────────────────▼──────────────────────┐
        │  app/rag.py    RAG 引擎                   │
        │  取记忆 → 改写问题 → 检索 → 组装提示词    │
        │  → 调大模型 → 后处理 → 写回记忆           │
        └────┬──────────┬────────────┬─────────────┘
             │          │            │
      ┌──────▼───┐ ┌────▼─────┐ ┌───▼──────────┐
      │ Milvus   │ │ Redis    │ │ 大模型 API   │
      │ 向量知识库│ │ 短期记忆 │ │ (OpenAI兼容) │
      │ 89 块    │ │ 最近10轮 │ │              │
      └──────────┘ └──────────┘ └──────────────┘
             ▲
      ┌──────┴────────────────────────────────┐
      │ app/db.py  数据层（SQLite→MySQL）      │
      │ users 表 / roles 表（4 个预置角色）    │
      └───────────────────────────────────────┘
```

---

## 四、P1 五步细表（口诀：拆 → 洗 → 切 → 嵌 → 存）

| 步骤 | 做什么 | 代码 | 状态 |
|---|---|---|---|
| 0. 数据 | 《国家基层高血压防治管理指南 2020 版》PDF（35 页） | — | ✅ |
| 1. 拆 | PDF → 纯文本（25369 字符） | `app/offline_pipeline.py: parse_document` | ✅ 已亲手跑通 |
| 2. 洗 | 去页眉页脚 + 去点线/空格/页码（20924 字符） | `app/offline_pipeline.py: clean_text` | ✅ 已亲手跑通 |
| 3. 切 | 长文本切块（**250 字/块 → 89 块**，从 500 字优化而来） | `app/offline_pipeline.py: split_chunks` | ✅ 已跑通 |
| 4. 嵌 | 每块变向量（bge-small-zh-v1.5，512 维） | `app/offline_pipeline.py: embed_chunks` | ✅ 已跑通 |
| 5. 存 | 向量 + 原文入 Milvus | `app/offline_pipeline.py: sync_milvus` | ✅ 89 条 |

> 课程指定的 **BGE-M3（1024 维）** 尚未启用，当前是 `bge-small-zh-v1.5`。切换模型后需要重新生成向量，并新建匹配维度的 Milvus collection，不能把不同维度写入现有库。

---

## 五、阿健当前待办

### A. 验收 P4 服务化（现在就做）

服务已在跑，**浏览器打开**：`http://127.0.0.1:8000/docs`

1. 点 `POST /api/chat` → Try it out → 填
   ```json
   {"user_id":1,"role_id":1,"message":"高血压怎么诊断"}
   ```
   → Execute，看 answer / rewritten_query / sources 三个字段
2. **再点一次，改填** `{"user_id":1,"role_id":1,"message":"它有什么症状"}`
   → 观察 `rewritten_query` 是否变成"高血压有什么症状"（**这是多轮记忆的证据**）
3. 点 `GET /api/history?user_id=1&role_id=1` → 看记忆里存了什么

### B. 顺手清理（可选）
- [ ] 删掉 `db\` 下的测试文件：test_milvus.db、test2.db、verify.db

---

## 六、待讲解清单（项目结束后统一讲）

**教学节奏已确认：先实现，后统一讲解。** 项目做完后按下面清单逐个文件逐行过。

| 序号 | 文件 | 讲什么 |
|---|---|---|
| 1 | `app/offline_pipeline.py: parse_document` | 解析器选择、PDF文字层与OCR回退 |
| 2 | `app/offline_pipeline.py: clean_text` | 正则清洗和文本去噪 |
| 3 | `app/offline_pipeline.py: split_chunks` | 段落聚合分块、JSONL元数据、分块大小为何从500改250 |
| 4 | `app/offline_pipeline.py: embed_chunks / sync_milvus` | 向量化、Milvus集合设计、稳定ID更新 |
| 5 | `scripts/hybrid_search.py` | 向量检索、BM25、RRF 融合——三种算法对比 |
| 6 | `scripts/ask.py` | 提示词模板、LLM 调用、后处理 |
| 7 | `scripts/chat.py` / `app/rag.py` | 短期记忆、Query 改写、引擎封装 |
| 8 | `app/main.py` / `app/db.py` | FastAPI 接口、pydantic 校验、表设计 |
| 9 | RAGAS 评测 | 检索质量怎么量化 |
| 10 | 总复盘 | 全链路串讲 + 答辩问答预演 |

---

## 七、记住的踩坑经验（答辩可以讲）

| 坑 | 现象 | 解法 |
|---|---|---|
| PDF 下载被截断 | 文件能打开但 0 页 | 看文件头 `/L` 真实大小 → `curl -C -` 断点续传 |
| 假 python | 敲 python 毫无反应 | WindowsApps 占位符抢 PATH；用 `where python` 查真身 |
| conda 环境空壳 | activate 成功但没有 python.exe | 建环境中途失败留下的残骸；删掉重建 |
| 数据清洗误删 | 按"重复次数"删会删掉药名 | 改用"位置法"（页眉页脚按坐标判断） |
| Docker 拉不到镜像 | 各种加速器 EOF | 改用 **Milvus Lite**（免 Docker，本地文件即数据库） |
| pymilvus 版本不匹配 | `MilvusException (code=1)` | pymilvus 2.6.17 + milvus-lite 3.2.1 |
| 环境变量被库占用 | 本地路径报 Illegal uri | 改名 `RAG_MILVUS_URI` 避开 `MILVUS_URI` |
| **分块过大语义稀释** | "80岁降到多少"检索不中 | 一个块混两个主题 → 分块 500 字降到 **250 字** |
| Milvus upsert 残留旧版本 | 89 条查询出 178 条 | 查询后按 id 去重 |
| 纯向量抓不住关键词 | "80岁""150/90" 命不中 | 需要**混合检索**（向量管语义 + BM25 管关键词） |

---

## 八、上课规矩

1. 我先讲这一步要干什么、用什么工具
2. 给你命令/代码，**你在自己电脑上敲**，把输出贴给我
3. 跑通了才勾掉任务，进入下一步
4. 代码逐行讲解统一放到项目完成后（那时你对全局有感觉，理解更深）
