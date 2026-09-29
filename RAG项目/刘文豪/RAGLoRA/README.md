# RAGLoRA · 多用户多角色 RAG 角色扮演系统

基于 RAG 的**角色扮演对话系统**：用户注册登录后可选择不同角色（心血管内科医生 / 执业律师）对话，
每个角色拥有**独立的知识库**与**独立的人格提示词模板**，对话具备多轮记忆能力。

---

## 一、功能特性

| 能力 | 说明 |
|---|---|
| **多用户** | 注册登录（JWT + pbkdf2），会话/消息/检索历史全部按用户隔离 |
| **多角色** | 角色 = 人格三层（身份/风格/约束）+ 知识库绑定；加角色零代码改动 |
| **混合检索** | bge-m3 产出 dense(1024维) + learned sparse 双路召回 → RRF 融合 |
| **精排** | bge-reranker-v2-m3 cross-encoder 重排，GPU 分时复用 |
| **多轮记忆** | Redis LIST 保存最近 6 轮，未命中回源 MySQL 懒加载重建 |
| **长期记忆** | Milvus `kb_memory` 按用户隔离存整轮问答，跨会话语义召回注入提示词 |
| **Redis 五类数据** | String 限流 · Hash 角色缓存 · List 短期记忆 · Set 分类索引 · zSet 热度榜（全部 fail-open 降级） |
| **查询改写** | 多轮对话中做指代消解（启发式判断，无需改写时跳过以省延迟） |
| **流式输出** | SSE 逐 token 推送，并实时下发检索链路明细 |
| **链路可视** | 前端展示改写前后对比、召回/精排耗时、每条来源的精排分数 |
| **知识库管理** | 文档入库（后台任务 + 进度轮询）、集合统计、删除连带删向量 |
| **检索调试台** | 纯检索不生成，用于对比混合检索与精排效果 |

---

## 二、技术栈（均复用本机已有环境，无重复安装）

| 组件 | 选型 | 说明 |
|---|---|---|
| 嵌入模型 | **bge-m3**（本地 `D:\桌面\模型\嵌入模型\bge-m3`） | 原生 transformers 加载，同时产出 dense + sparse |
| 精排模型 | **bge-reranker-v2-m3**（本地） | fp16，GPU 分时复用（约 413ms/轮） |
| 向量库 | **Qdrant 嵌入式** | `QdrantClient(path=...)`，无需单独服务 |
| 生成模型 | **Ollama qwen2.5:7b** | OpenAI 兼容接口，流式 |
| 业务库 | **MySQL 8.0** | 用户/角色/会话/消息/知识库文档 |
| 短期记忆 | **Redis**（免安装版，`tools/redis/`） | **五类数据类型**：List 短期记忆 / String 限流 / Hash 角色缓存 / Set 分类索引 / zSet 热度榜 |
| 后端 | FastAPI + SQLAlchemy（同步）| 单 worker（Qdrant 文件锁限制） |
| 前端 | React 19 + Vite 8 + TypeScript | 复用参考项目设计系统，无新增 npm 依赖 |
| 运行时 | `D:\anaconda3\envs\rag_env`（Python 3.10.14） | **唯一后端环境** |

> 📌 **依赖策略**：本项目严格遵守「绝不重复安装」。相对参考项目**唯一新增的 pip 包是
> `qdrant-client`**（dry-run 确认零既有包被升降级）。明确**不装** FlagEmbedding（会破坏
> transformers 版本链）、pdfplumber、ragas。详见 `backend/requirements.txt`。

---

## 三、快速开始

### 3.1 前置条件

| 依赖 | 要求 | 检查方式 |
|---|---|---|
| 内存 | **≥ 16 GB（推荐 24 GB+）** | 见下方「内存约束」 |
| MySQL | 运行于 127.0.0.1:3306（`root/123456`） | `netstat -ano \| grep :3306` |
| Ollama | 已拉取 `qwen2.5:7b` | `curl http://localhost:11434/api/tags` |
| Redis | 免安装版已随项目提供 | `bash tools/redis/start_redis.sh` |
| 本地模型 | `D:\桌面\模型\嵌入模型\bge-m3`、`D:\桌面\模型\精排模型\bge-reranker-v2-m3` | 目录存在即可 |

> 连接参数均可用环境变量覆盖（`MYSQL_PASSWORD`、`OLLAMA_BASE_URL`、`EMBED_MODEL_PATH` 等）。

#### 内存构成

| 占用方 | 内存 |
|---|---|
| bge-m3（fp32，CPU 常驻） | ≈ 2.3 GB |
| bge-reranker-v2-m3（fp16，CPU 常驻） | ≈ 1.1 GB |
| Ollama qwen2.5:7b | ≈ 4.7 GB |
| 桌面程序（浏览器/IDE/模拟器） | 实测可达 6 GB+ |

**16 GB 机器上余量偏紧**。开发期实测触发过一次 Windows 资源耗尽保护
（系统事件 `Microsoft-Windows-Resource-Exhaustion-Detector` ID 2004）。
建议跑评测/压测前关掉浏览器多开与安卓模拟器；内存更紧时用
`RAGLORA_WARMUP=0` 启动，模型改为首次使用时按需加载（代价：首个请求慢 7~10 秒）。

### 3.2 一键启动

```bash
# 后端（自动拉起 Redis、探活依赖、后台预热模型）
bash backend/run.sh --bg

# 前端
cd frontend && npx vite
```

打开 **http://127.0.0.1:5173** → 注册 → 选角色 → 开始对话。

接口文档：**http://127.0.0.1:8000/docs**

### 3.3 停止

```bash
bash backend/shutdown.sh
```

### 3.4 首次部署：导入知识库

知识库数据不入库则检索为空。**需在后端停止时执行**（Qdrant 嵌入式模式持有独占文件锁）：

```bash
bash backend/shutdown.sh
cd backend && D:/anaconda3/envs/rag_env/python.exe scripts/ingest_all.py
bash run.sh --bg
```

> 服务运行时要新增文档，改用 `POST /api/kb/ingest`（后台任务，可在服务内安全访问 Qdrant）。

---

## 四、目录结构

```
RAGLoRA/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI 入口（CORS + 路由 + 启动建表/种角色/预热）
│   │   ├── core/                config / security(JWT) / db / logging
│   │   ├── models.py            6 张表 ORM
│   │   ├── schemas.py           Pydantic 出入参
│   │   ├── seed.py              内置角色 + 人格模板骨架
│   │   ├── deps.py              当前用户依赖注入
│   │   ├── routers/             health / auth / characters / kb / search / conversations / chat
│   │   └── services/
│   │       ├── embed.py         bge-m3 dense+sparse 编码器（设备可切换）
│   │       ├── ingest.py        解析 → 清洗 → 分块 → 编码 → 写 Qdrant
│   │       ├── retrieval.py     混合检索 + RRF 融合
│   │       ├── rerank.py        精排（GPU 分时复用）
│   │       ├── persona.py       人格三层渲染 + Prompt 组装
│   │       ├── rag_chain.py     主链路编排
│   │       ├── memory.py        Redis 短期记忆
│   │       ├── redis_extra.py   Redis 五类数据业务用法（限流/缓存/索引/热度榜）
│   │       ├── long_memory.py   Milvus 长期记忆（跨会话）
│   │       ├── llm.py           Ollama 客户端 + 查询改写
│   │       └── qdrant_store.py  Qdrant 单例
│   ├── scripts/                 ingest_all.py / test_m1.py / test_m3_m4.py
│   ├── eval/                    qa_set.json + run_eval.py
│   ├── run.sh / shutdown.sh
│   └── requirements.txt
├── frontend/                    React 19 + Vite 8（node_modules 已随项目提供）
├── datasets/
│   ├── medical/                 高血压防治指南 2024修订版(98页) + 国家基层指南2025版(15页)
│   └── legal/                   176 部法律 txt（含民法典 1260 条）
├── tools/
│   ├── redis/                   免安装 Redis 5.0.14.1 + 启停脚本
│   └── m0/                      M0 环境验证脚本（可复跑）
├── qdrant_storage/              Qdrant 嵌入式数据
└── docs/                        需求规格说明书(05) / 设计文档 / 接口文档 / 部署文档 / 思维导图(11) / 业务流程图(12) / 专项报告
```

---

## 五、关键设计决策

### 5.1 精排模型的显存策略（最重要的一处取舍）

8G 显存下，**精排模型不能常驻 GPU**——它会抢走 Ollama 的 KV cache：

| 方案 | 精排延迟 | 生成吞吐 |
|---|---|---|
| 精排常驻 GPU | 28 ms | **5.2 tok/s** ❌ |
| 精排常驻 CPU | 1.4–2.8 s | 47.8 tok/s |
| **权重驻 CPU 内存，用时搬 GPU，用完搬回** | **413 ms** | **47.8 tok/s** ✅ |

因为**精排发生在生成之前，两者本就不重叠**，用完立即释放显存，生成时 GPU 是独占的。

### 5.2 不装 FlagEmbedding

`FlagEmbedding 1.3.5` 要求 `transformers>=4.44.2`，而 rag_env 是 `4.39.3`，安装会触发连锁升级。
改用原生 `transformers` + `torch` 手写，数学完全等价：

```python
dense  = mean_pooling(last_hidden_state, attention_mask) → L2 normalize
sparse = relu(sparse_linear(last_hidden_state)) * attention_mask
         → 按 token_id 取 max 聚合（过滤 tokenizer.all_special_ids）
```

`sparse_linear.pt` 实测为 `OrderedDict{'weight':[1,1024],'bias':[1]}`，`nn.Linear(1024,1)` 直接重建。

### 5.3 分块策略按语料类型分流

| 语料 | 策略 | 效果 |
|---|---|---|
| 法条 txt | 按「第X条」切分 | 176 部 → 14,319 chunks，条文号覆盖 **100%** |
| 指南 PDF | 段落聚合 + 页眉页脚清洗 | 113 页 → 688 chunks |

正则同时兼容两种真实语料格式：`第一条 …`（条文在行首）与 `《法名》第一条…`（法名同行）。

### 5.4 分数阈值只能用精排分

RRF 融合分是 `1/(k+rank)` 的**量化值**，会出现大量并列——实测无关查询也给不相关文档打 0.5 分。
**相关性过滤必须用精排分**。

---

## 六、评测

```bash
cd backend
D:/anaconda3/envs/rag_env/python.exe eval/run_eval.py            # 全量
D:/anaconda3/envs/rag_env/python.exe eval/run_eval.py --limit 5  # 快速冒烟
```

评测维度：**来源命中率**（期望来源是否出现在精排结果）、**答案覆盖率**（期望关键词是否出现在回答）、
**引用标注率**（是否带 `[n]` 来源编号）、**拒答正确性**（越界问题是否守住边界）、**延迟**。

### 实测结果（43 题，2026-09-15）

| 角色 | 题数 | 来源命中率 | 答案覆盖率 | 引用标注率 | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| 心血管内科医生（medical） | 20 | **100%** | 95% | **100%** | 7.1 s |
| 执业律师（legal） | 20 | **100%** | 95% | **100%** | 7.6 s |
| 拒答（越界问题） | 3 | — | **100%** | — | 4.5 s |

链路耗时分解：混合检索 **367 ms** · 精排 **751 ms** · 生成 **5997 ms**

**两处 5% 未覆盖的原因（均非系统缺陷）**：

- 医学「家庭自测血压的正常值」——指南中该值为 `135/85 mmHg`，模型给出了回答但未复述数字（检索侧正确命中）
- 法律「试用期最长多久」——已标注为**语料缺口**：`《劳动法》《劳动合同法》不在 176 部法律语料中`，
  模型没有编造条款，边界守住了

> 📌 早期版本的评测脚本有个缺陷值得一提：它给每个角色**共用一个会话**连问 20 题，跑到第 9 题时
> 上下文已堆了 8 轮问答，模型的回答风格被历史带偏、不再标注 `[n]`，导致医学引用率被误判为 30%。
> 修正为**每题独立会话**后恢复到 100%。评测脚本自身的缺陷会伪装成系统缺陷，这点值得警惕。

完整报告见 [`backend/eval/eval_report.md`](backend/eval/eval_report.md)。

---

## 七、已知限制

| 限制 | 原因 | 规避 |
|---|---|---|
| 后端必须单 worker | Qdrant 嵌入式模式持有独占文件锁 | 不要加 `--workers` |
| 入库不能在服务外另起进程 | 同上 | 用 `POST /api/kb/ingest` |
| 服务重启后 Redis 不会自动拉起 | 免安装版无开机自启 | `run.sh` 已包含幂等启动逻辑 |
| 首次请求较慢 | 模型加载 | 启动时后台预热（`main.py` lifespan） |

---

## 八、踩过的坑（备查）

| 现象 | 根因 | 解决 |
|---|---|---|
| `UnicodeDecodeError: byte 0xd7` | Windows 版 Redis 的 `INFO` 响应含 GBK 中文路径 | 客户端加 `encoding_errors="replace"` |
| `1064 … near 'NULLS LAST'` | `nullslast()` 是 PostgreSQL 语法，MySQL 不支持 | 去掉，MySQL 的 `DESC` 本就把 NULL 排最后 |
| HTTP 403 / 模型答非所问 | 稀疏向量被特殊符 `<s>` 污染 | 编码时过滤 `tokenizer.all_special_ids` |
| 生成吞吐突然暴跌到 5 tok/s | 精排模型挤占显存 | 改为 GPU 分时复用 |
| GitHub CDN 连接被重置 | 大陆网络对 release-assets 的干扰 | 用镜像 `ghfast.top` |
| **后端「静默消失」** | `nohup ... &` 在 Git Bash on Windows 下**并未真正脱离父进程**，启动它的 shell 一退出后端就被一并收走。表现为：无 Python 堆栈、无 Windows 崩溃事件，极易误判成程序 bug | `run.sh --bg` 改用 PowerShell `Start-Process` |
| 评测里模型不再标注 `[n]` | **评测脚本自身的缺陷**：每个角色共用一个会话连问 20 题，上下文堆积后模型风格被带偏 | 改为每题独立会话（详见 §6） |
