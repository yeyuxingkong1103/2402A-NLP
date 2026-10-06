# 项目讲解文档（答辩 / 演讲用）

> 这是一份面向"不懂技术的听众"的讲解稿，尽量少用术语。
> 所有描述都基于项目真实代码，不编造。文中的代码片段也都直接摘自本项目。

---

## 一、这个项目是做什么的

一句话：**一个"能陪你聊天、记得住你、还能读你上传的文档"的 AI 角色扮演系统。**

拆开说，它做了三件事：

1. **扮演一个角色跟你聊天**：你创建一个"角色"（比如温柔图书管理员、毒舌侦探），系统就用这个角色的口吻、性格、背景跟你对话。
2. **记得住事情**：
   - 短期：记得最近几十句话，不会"聊了上一句忘下一句"。
   - 长期：会从对话里自动抽取"你喜欢猫""你下周要考试"这类值得记住的事，下次聊到相关内容时能想起来。
3. **能读文档**：你上传一份 PDF / Word / 文本，它能读懂内容，并在你提问时"翻资料回答"，还能告诉你答案出自哪份文件的哪一段（引用溯源）。

它不是一个单纯的"套壳聊天机器人"，而是把 **大语言模型（LLM）+ 向量检索（RAG）+ 记忆系统** 组合起来的完整后端系统。

---

## 二、整体架构（文字版架构图）

```
                         ┌──────────────────────────┐
                         │      浏览器 / 前端          │
                         └────────────┬─────────────┘
                                      │  HTTP（REST + 流式 SSE）
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │              后端 API（FastAPI / Python）             │
        │                                                     │
        │   负责：登录鉴权、角色管理、会话、对话编排              │
        └───┬──────────┬──────────┬──────────┬──────────┬──────┘
            │          │          │          │          │
            ▼          ▼          ▼          ▼          ▼
     ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
     │  MySQL    │ │ Redis  │ │ Milvus │ │ 大模型  │ │ 后台任务   │
     │ 结构化数据 │ │ 短期记忆│ │向量数据库│ │ DeepSeek│ │ ARQ worker│
     │ 用户/角色/ │ │ 对话历史│ │(检索记忆)│ │ 生成回复│ │ 异步抽记忆 │
     │ 会话/消息  │ │ 摘要    │ │        │ │        │ │ /解析文档  │
     └──────────┘ └────────┘ └────────┘ └────────┘ └──────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      ▼                               ▼
             ┌────────────────┐             ┌────────────────┐
             │ 向量化服务 bge-m3│             │ 重排服务 bge-rerank│
             │ 把文字变成向量   │             │ 给候选结果打分排序  │
             └────────────────┘             └────────────────┘
```

**各部分干什么（一句话）**：

| 组件 | 用通俗的话说 |
|------|--------------|
| **FastAPI 后端** | 系统的"大脑中枢"，接收请求、协调其他组件 |
| **MySQL** | 存"账本"：谁注册了、创建了哪些角色、有哪些会话、每条消息是什么 |
| **Redis** | 存"临时草稿"：最近几轮对话、会话摘要，读写快，重启可丢 |
| **Milvus** | 存"长期记忆的检索库"：把角色设定、记忆、文档都切成片段变成向量，供"按意思搜索" |
| **大模型（DeepSeek）** | 真正"动嘴"生成回复的那一位 |
| **bge-m3 服务** | 把一段文字变成一串数字（向量），文字意思越接近，数字越接近 |
| **bge-rerank 服务** | 从一堆候选片段里，精确挑出和问题最相关的前几名 |
| **ARQ worker** | 后台"打杂工"，干耗时活：抽取长期记忆、解析上传的文档 |

> 关键设计思想：**"重活"都拆到后台去干，聊天请求只走最快的路径**，这样聊天才流畅、不卡。

---

## 三、一次对话的完整流程（从发消息到收到回复）

这是全项目最重要的一个场景。以用户发一句"她喜欢什么？"为例，走一遍：

### 1. 用户发消息
前端调用 `POST /api/v1/chat/stream`，带上 `session_id`（哪个会话）、`content`（说的话）。

### 2. 后端先"确认身份和上下文"
`chat.py` 里的代码先做两件事：
- 确认这个会话确实属于当前登录用户（`_resolve` 函数，防止访问别人的会话）；
- 查出这个会话绑定的角色信息。

```python
# backend/app/api/routes/chat.py
async def _resolve(session_id, user, db):
    s = await db.get(Session, session_id)
    if s is None or s.user_id != user.id:          # 会话不存在或不属于你 → 拒绝
        raise HTTPException(status_code=404, detail="会话不存在")
    c = await db.get(Character, s.character_id)
    return s, c
```

> 人话：先确认"你确实有资格在这个会话里说话"，再拿到"你要跟谁聊天"。

### 3. 检索：去"记忆库"里翻相关的资料（RAG）
调用 `RAGPipeline.retrieve()`，从三个地方翻资料（详见第四节）：
- 这个**角色**的设定里，跟问题相关的部分；
- 你与这个**角色**之间的长期记忆里，相关的往事；
- 你上传的**文档**里，相关的内容。

### 4. 组装提示词（prompt）
`ChatService.prepare()` 把检索到的内容、短期记忆、历史摘要拼成一段"给大模型的说明书"：

```python
# backend/app/services/chat_service.py
async def prepare(self, session_id, user_id, character, query) -> PreparedTurn:
    retrieved = await self.rag.retrieve(character["id"], user_id, query)   # ① 检索相关资料
    summary = await self.redis.get_summary(session_id)                      # ② 取历史摘要
    recent = await self.redis.get_recent(session_id, self.rounds)           # ③ 取最近几轮对话
    memories_text = [s.text for s in retrieved.settings] + [s.text for s in retrieved.memories]
    documents_text = [s.text for s in retrieved.documents]
    msgs = build_prompt(character, memories_text, summary,
                        recent + [{"role": "user", "content": query}],
                        documents=documents_text)
    sources = [s.to_dict() for s in ...]    # 同时把"引用来源"也整理好，后面给前端
    return PreparedTurn(messages=msgs, sources=sources)
```

> 人话：把「角色说明书 + 相关记忆 + 相关文档 + 最近聊天记录 + 你刚说的话」打包成一份材料，交给大模型。

`prompt_builder.py` 里这样组织这份材料（节选）：

```python
# backend/app/services/prompt_builder.py
lines = [f"你扮演「{character['name']}」。"]
sections = [
    ("## 角色设定", character.get("persona")),
    ("## 世界观", character.get("worldview")),
    ("## 你与用户的关系", character.get("relationship")),
    ("## 隐藏设定（只影响行为，禁止主动透露）", character.get("hidden_setting")),
    ("## 说话风格示例", character.get("sample_dialogue")),
]
```

> 人话：告诉大模型"你现在是某某角色，这是你的人设、背景、和用户的关系、甚至还有一些只能影响行为、不能明说的隐藏设定"。

### 5. 大模型生成回复（流式返回）
`OpenAICompatLLM.chat_stream()` 调用 DeepSeek，一次吐几个字地返回，前端像打字机一样逐字显示：

```python
# backend/app/services/chat_service.py
async def chat_stream(self, session_id, user_id, character, query):
    turn = await self.prepare(session_id, user_id, character, query)
    async for token in self.llm.chat_stream(turn.messages):   # 逐字产出
        yield token
```

### 6. 落库 + 回传
后端把这条"用户消息"和"AI 回复"都存进 MySQL（`Message` 表），并把回复**引用了哪些来源**一起存下来；同时更新 Redis 里的短期对话记录。

### 7. 前端展示
前端收到回复，还能看到这条回答"参考了哪些片段"（引用溯源，见第六节）。

**整条链路总结**：发消息 → 校验身份 → 检索资料 → 拼说明书 → 大模型生成 → 存消息 → 返回（附带引用来源）。

---

## 四、RAG 检索是怎么做的

RAG 是"检索增强生成"的缩写。核心思路：**大模型自己记不住你的私有资料，所以先帮你"查资料"，再让它"照着资料答"**。

本项目 RAG 分三步：**混合检索 → 重排 → 降级兜底**。

### 4.1 混合检索（Hybrid Search）

传统的"关键词搜索"只能匹配字面，比如你搜"苹果"，找不到写"手机"的段落。**向量检索**能把文字变成向量，按"意思"找。

本项目的亮点是用了 **BGE-M3 模型，同时产出两种向量**：

1. **稠密向量（dense）**：1024 个数字，表达"整体语义"——"苹果"和"iPhone"在向量空间里会很近；
2. **稀疏向量（sparse）**：表达"具体词"的权重——擅长精确匹配专有名词、人名。

`milvus_store.py` 里对每个集合都做了两次搜索（一次稠密、一次稀疏），再用 **RRF（倒数排名融合）** 把两边的结果合并：

```python
# backend/app/store/milvus_store.py（节选）
def _run():
    dense_res = self.client.search(collection, data=[dense_q],
                                   anns_field="dense_vector", ...)[0]   # 按语义搜
    sparse_res = self.client.search(collection, data=[sparse_q],
                                    anns_field="sparse_vector", ...)[0] # 按词搜
    dense_ids = [h["id"] for h in dense_res]
    sparse_ids = [h["id"] for h in sparse_res]
    fused_ids = rrf_fuse(dense_ids, sparse_ids)   # 融合两份排序
    ...
```

RRF 融合的算法很朴素（`core/rrf.py`），就是"两边都排前面的，综合分更高"：

```python
# backend/app/core/rrf.py
def rrf_fuse(*ranked_lists, k: int = 60) -> list[str]:
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)  # 排名越靠前，加分越多
    return sorted(scores, key=scores.get, reverse=True)
```

> 人话：语义匹配 + 关键词匹配双保险，谁在两边都靠前，谁就更可能是答案。

### 4.2 重排（Rerank）

粗检索会捞回一堆候选（top_k=20），但可能鱼龙混杂。**重排模型**会对"问题 vs 每个候选片段"逐个打分，挑出最相关的 top_m=5 条。

重排是独立的服务（`services/rerank/app.py`），用 BGE-reranker-base 模型：

```python
# services/rerank/app.py（节选）
@app.post("/rerank")
async def rerank(body: RerankIn):
    pairs = [[body.query, p] for p in body.passages]   # 问题与每个片段一一配对
    scores = _reranker.compute_score(pairs)            # 模型给每对打相关分
    ranked = sorted(range(len(body.passages)), key=lambda i: -scores[i])[: body.top_m]  # 取前几名
    return {"results": [{"index": i, "score": float(scores[i])} for i in ranked]}
```

> 人话：粗检索负责"别漏"，重排负责"挑准"。就像先大海捞针捞出一把，再用放大镜挑最像的几根。

### 4.3 降级兜底（Graceful Degradation）

这是体现工程严谨性的地方：**任何一步挂了，都不让整个聊天崩掉**。

- **检索失败**：某个集合查不了，就返回空列表，继续往下走；
- **重排失败**：就用"未重排的原始顺序"兜底，只是相关度分数标记为"无"。

```python
# backend/app/services/rag_pipeline.py（节选）
async def _rerank_safe(self, query, passages, label):
    """rerank 失败时不静默返回空，回退到未重排的原顺序（相关度置 None，前端显示「—」）。"""
    try:
        return await self.rerank.rerank(query, passages, self.top_m)
    except Exception:
        logger.exception("rerank(%s) failed, falling back to un-ranked order", label)
        return [(i, None) for i in range(len(passages))]   # 兜底：不排序，分数为 None
```

还有一节代码把所有检索都包在 `try/except` 里：

```python
# backend/app/services/rag_pipeline.py
try:
    settings = await self.milvus.hybrid_search(...)
except Exception:
    logger.exception("hybrid_search(character_settings) failed")
    settings = []          # 查不到就当没有，不让整个请求失败
```

> 人话：重排服务、向量库都是"可选的加分项"——没了它们，系统还能用，只是答案质量略降，这叫"优雅降级"。

---

## 五、记忆是怎么存的

记忆分两层，各司其职：

### 5.1 短期记忆：Redis（快、易失）

存"最近几轮对话"和"会话摘要"。Redis 是内存数据库，读写飞快，适合"每次聊天都要读"的热数据。

```python
# backend/app/store/redis_store.py
async def push_message(self, session_id, role, content):
    await self.redis.rpush(f"session:{session_id}:history", json.dumps({"role": role, "content": content}))

async def get_recent(self, session_id, rounds):
    items = await self.redis.lrange(f"session:{session_id}:history", -2 * rounds, -1)  # 取最后 N 轮
    return [json.loads(i) for i in items]
```

- `session:{id}:history`：一个列表，存这个会话最近的对话；
- `session:{id}:summary`：这个会话的历史摘要（当对话变长时，用摘要代替旧原文，省 token）。

> 人话：Redis 像"手边的便签本"，记着刚才聊了啥，翻起来快，但换台机器可能就丢了。

### 5.2 长期记忆：Milvus（慢一点、持久、按语义检索）

长期记忆是"抽取"出来的、值得长期记住的事实/事件/偏好，存进向量库 Milvus，供以后按意思检索。

**抽取过程**（`MemoryExtractor`）分三步：

1. **抽取**：让大模型从一段对话里挑出"值得记住的事"：

```python
# backend/app/services/memory_extractor.py（节选）
EXTRACT_PROMPT = """从以下对话中抽取值得长期记住的信息，只抽取事实、事件、用户偏好三类。
以 JSON 返回，格式：{"memories": [{"type": "fact|event|preference", "content": "...", "importance": 0-10}]}
没有则返回 {"memories": []}。"""
```

2. **去重**：如果这件事以前已经记过了，就不再重复记（用向量余弦相似度判断"是不是同一件事"）：

```python
async def dedupe(self, memories, character_id, user_id):
    ...
    too_similar = any(_cosine(emb, e.get("dense_vector", [])) > self.threshold for e in existing)
    if not too_similar:
        kept.append(m)      # 不相似才留下
```

3. **写入**：把保留的记忆向量化后写进 Milvus 的 `long_term_memory` 集合。

**谁来触发抽取？** 抽取是耗时的活，不能挡在聊天主流程里。它被交给**后台任务队列 ARQ** 异步执行（`worker/tasks.py` 的 `extract_memory_task`），通过 `/extract` 接口入队触发。这样"抽取记忆"和"用户聊天"互不干扰。

```python
# backend/app/worker/tasks.py（节选）
async def extract_memory_task(ctx, session_id, user_id, character_id):
    extractor = MemoryExtractor(get_llm(), get_embedding(), get_milvus())
    ...
    written = await extractor.run(session_id, user_id, character_id, conversation)  # 后台慢慢干
```

> 人话：短期记忆是"随手记的草稿"，长期记忆是"认真归档的档案"。归档这活放到后台干，不影响前台聊天。

---

## 六、引用溯源是怎么实现的

引用溯源回答的问题是：**"这条回复是参考了哪些资料得出的？"** 这是 RAG 应用最关键的"可解释性"能力——让用户能核验答案不是模型瞎编的。

### 实现分三步

**1. 检索时就给每段资料打上"来源标签"**

`rag_pipeline.py` 里定义了一个 `Source` 结构，记录每段资料的：类型（角色设定 / 记忆 / 文档）、原文、展示名（文档名 / 设定类型 / 记忆类型）、所在位置（第几块）：

```python
# backend/app/services/rag_pipeline.py
@dataclass
class Source:
    type: str            # "setting" | "memory" | "document"
    text: str            # 片段原文
    label: str = ""      # 展示名：文档名 / setting_type / memory_type
    doc_id: int | None = None
    chunk_index: int | None = None
    score: float | None = None   # 相关度（0~1；回退时 None）
```

**2. 生成回复时，把"来源"和"回复"一起打包返回**

```python
# backend/app/services/chat_service.py
sources = [s.to_dict() for s in retrieved.settings + retrieved.memories + retrieved.documents]
return PreparedTurn(messages=msgs, sources=sources)
```

**3. 落库 + 回传给前端**

在 `chat.py` 里，回复存进 MySQL 时，把来源以 JSON 字符串一并存到 `Message.sources` 字段；同时通过 SSE 流把 `sources` 事件先推给前端，前端就能在回复旁边显示"参考来源"。

```python
# backend/app/api/routes/chat.py（流式）
yield f"event: sources\ndata: {json.dumps({'sources': turn.sources}, ensure_ascii=False)}\n\n"
...
db.add(Message(session_id=s.id, role="assistant", content=reply,
               sources=json.dumps(turn.sources, ensure_ascii=False)))
```

> 人话：系统不是"悄悄引用"，而是把"我参考了这几段，它们分别来自哪"明明白白告诉用户，可以追溯、可以核验。

---

## 七、角色隔离是怎么做的

"角色隔离"就是：**用户 A 和角色 B 的记忆、设定，绝不能让用户 C 串到、串台**。

核心手段是 **Milvus 检索时的"过滤表达式"（filter）**——每次搜索都强制圈定范围。

在 `rag_pipeline.py` 里，三类检索各自带不同的过滤条件：

```python
# backend/app/services/rag_pipeline.py（节选）
settings = await self.milvus.hybrid_search(
    "character_settings", query, f"character_id == {character_id}", self.top_k)   # 只搜当前角色

mems = await self.milvus.hybrid_search(
    "long_term_memory", query,
    f"user_id == {user_id} && character_id == {character_id}", self.top_k)   # 只搜「这个用户 × 这个角色」的记忆
```

- **角色设定**：加 `character_id == 当前角色`，只搜当前角色的设定，不碰别的角色；
- **长期记忆**：加 `user_id == 当前用户 && character_id == 当前角色`，**双重隔离**——记忆既不跨用户，也不跨角色。

**数据落库时也做了隔离**：每条记忆在写入时就带着 `user_id` 和 `character_id` 两个字段（`milvus_store.py` 的 `upsert_memories`），这样检索时才有据可依。

此外，**接口层的权限校验**也兜底：会话查询（`_resolve`）、消息列表、记忆列表、删除等操作，都会校验 `s.user_id == user.id`，非本人一律 404。

> 人话：每个用户的每个角色，都像有独立的"记忆抽屉"，抽屉上贴着双重标签（谁 + 谁的角色）。检索时按标签开抽屉，标签对不上就打不开。

---

## 八、关键文件一句话讲解

### 后端核心服务（backend/app/services/）

| 文件 | 负责什么 |
|------|----------|
| **rag_pipeline.py** | RAG 检索编排：定义 `Source`/`RetrievalResult` 结构，把三类检索 + 重排 + 降级兜底串起来，返回带来源标签的结果 |
| **chat_service.py** | 聊天编排：`prepare()` 把检索结果、短期记忆、摘要拼成 prompt，`chat`/`chat_stream` 分别做非流式和流式生成 |
| **prompt_builder.py** | 拼提示词：把角色人设、世界观、关系、隐藏设定、记忆、文档组装成给大模型的 system 消息 |
| **memory_extractor.py** | 长期记忆：用大模型从对话抽取"事实/事件/偏好"，再做向量去重，最后写进 Milvus |
| **character_service.py** | 角色设定索引：把角色的多段设定切成块（每 500 字一块），向量化后写入 Milvus |
| **document_ingestor.py** | 文档入库：解析 → 分块 → 先清旧块（幂等）→ 向量化写入 Milvus |
| **document_parser.py** | 文档解析：支持 txt/docx/pdf，PDF 无文字层时可选 OCR 兜底；纯同步、不调大模型 |

### 存储层（backend/app/store/）

| 文件 | 负责什么 |
|------|----------|
| **milvus_store.py** | Milvus 封装：建集合、混合检索（dense+sparse+RRF）、增删改；`init_collections` 带连接重试（等 Milvus 就绪） |
| **redis_store.py** | Redis 封装：短期对话历史（list）、会话摘要（string）的读写 |

### 后端接口（backend/app/api/routes/）

| 文件 | 负责什么 |
|------|----------|
| **chat.py** | 聊天接口：`/chat`（非流式）、`/chat/stream`（流式 SSE），校验会话归属、调用服务、落库 |
| **characters.py** | 角色 CRUD：创建/列表/详情/更新/删除，并维护角色设定的向量索引 |
| **sessions.py** | 会话管理：创建会话、列消息、删会话 |
| **documents.py** | 文档上传/下载/列表/删除，上传后入队后台解析 |
| **memories.py** | 记忆查询/删除/手动触发抽取 |
| **favorites.py** | 收藏：收藏某条回答、列出收藏、取消收藏 |

### 模型与外部服务

| 文件 | 负责什么 |
|------|----------|
| **core/providers/llm/openai_compat.py** | 封装 DeepSeek（OpenAI 兼容接口），提供 `chat` 和 `chat_stream` |
| **core/providers/embedding/bge_m3_http.py** | 调用 bge-m3 服务，拿到 dense + sparse 两类向量 |
| **core/providers/rerank/bge_rerank_http.py** | 调用 bge-rerank 服务做重排 |
| **core/fakes.py** | 测试用的假组件（FakeLLM/FakeEmbedding/FakeRerank），让测试不依赖真实大模型 |
| **core/rrf.py** | RRF 倒数排名融合算法 |
| **services/embed/app.py** | bge-m3 独立服务（FastAPI），本地加载模型，对外暴露 `/embed` |
| **services/rerank/app.py** | bge-rerank 独立服务（FastAPI），对外暴露 `/rerank` |

### 数据模型与后台任务

| 文件 | 负责什么 |
|------|----------|
| **db/models.py** | 定义 8 张表：用户、角色、会话、消息、收藏、记忆任务、文档、…… |
| **worker/tasks.py** | 后台任务：抽取记忆、解析文档（ARQ 异步执行） |
| **worker/run.py** | ARQ worker 的配置，注册可执行的任务 |
| **deps.py** | 依赖装配：单例缓存地创建 LLM / embedding / rerank / milvus / redis 组件 |
| **config.py** | 配置项（数据库地址、模型地址、检索参数等），支持 `.env` 覆盖 |

---

## 九、答辩可能被问的问题 + 怎么答

> 每个问题先给"一句话答案"，再给"展开怎么讲"。

### Q1：什么是 RAG？你的项目为什么要用 RAG？

**一句话**：RAG = 先检索相关资料，再让大模型照着资料回答，解决"大模型不懂你的私有资料、还会编造"的问题。

**展开**：大模型（DeepSeek）是通用模型，它不知道你的角色设定细节、你上传的私人文档。直接把这些问题甩给它，它会"一本正经地编"。所以我们在问它之前，先从自己的数据库里把相关资料检索出来，塞进提示词，让它"有据可答"，并且能给出引用来源、可追溯。

### Q2：你的"混合检索"到底混合了什么？为什么？

**一句话**：混合了"语义向量（dense）+ 词权重向量（sparse）"两种检索，再用 RRF 融合。

**展开**：dense 向量擅长"意思相近"（苹果 ≈ iPhone），sparse 向量擅长"精确词"（专有名词、编号）。只靠任一种都有盲区：纯语义会漏掉精确词，纯关键词会漏掉同义表达。BGE-M3 模型一次能同时产出两种向量，我们把两者各搜一遍、用 RRF 融合排名，召回更全。

### Q3：为什么还要再"重排"一次？不是已经检索出来了吗？

**一句话**：检索是"粗筛"，重排是"精挑"。

**展开**：粗检索为了不漏，会捞回 top_k=20 条候选，里面可能有大量弱相关。重排模型会对"问题 vs 每条候选"逐一精确打分，再取 top_m=5 条最相关的交给大模型。这样既保证召回，又保证精度，还省了大模型要看的 token。

### Q4：如果向量库或重排服务挂了，你的系统会怎样？

**一句话**：不会崩，会自动降级——检索失败返回空、重排失败退回原始顺序。

**展开**：`rag_pipeline.py` 里所有检索和重排都包在 try/except 里。检索失败就当作"没查到资料"，聊天还能继续（只是没有额外资料）；重排失败就用未重排的顺序兜底，相关度分数标记为"无"。这叫"优雅降级"，保证核心聊天功能不因一个附属组件故障而中断。

### Q5：短期记忆和长期记忆有什么区别？为什么分两层？

**一句话**：短期记忆快、存最近对话（Redis）；长期记忆按语义持久化、供以后检索（Milvus）。

**展开**：两者定位不同。短期记忆是"最近几轮对话原文"，每次聊天都要读，必须快，所以放内存数据库 Redis。长期记忆是"从对话里抽取出来的、值得长期记住的事实/偏好/事件"，需要"按意思"跨会话检索，所以向量化后放 Milvus。分层是为了各用其所长：快的干快活，检索的干检索活。

### Q6：长期记忆是怎么自动产生的？

**一句话**：后台任务让大模型从对话里抽取"事实/事件/偏好"，去重后向量化写入 Milvus。

**展开**：抽取是耗时的，所以放在 ARQ 后台任务里异步做，不挡聊天主流程。抽取分三步：① 大模型按固定格式抽取值得记的事（含重要性 0-10）；② 用向量余弦相似度去重，避免重复记同一件事；③ 向量化后写入 `long_term_memory` 集合。

### Q7：你的角色隔离是怎么实现的？怎么保证用户 A 看不到用户 B 的记忆？

**一句话**：靠"检索时强制加过滤条件 + 接口层权限校验"双重保障。

**展开**：数据写入时就带上 `user_id` 和 `character_id` 标签；检索时，角色设定强制 `character_id == 当前角色`，长期记忆强制 `user_id == 当前用户 && character_id == 当前角色`，双重隔离。另外接口层还会校验会话归属（`s.user_id == user.id`），非本人一律 404。两道防线保证不串台。

### Q8：引用溯源怎么保证回答"有据可查"？

**一句话**：检索出的每段资料都带来源标签（类型/出处/位置/相关度），随回复一起存库、回传、展示。

**展开**：检索时就把每段资料的 `type`、`label`（文档名/设定类型/记忆类型）、`chunk_index`、`score` 记下来，打包进 `sources`；存消息时把 sources 以 JSON 存进 `Message.sources`，前端在回复旁展示"参考来源"。这样用户能核验答案来自哪里，而不是模型凭空编的。

### Q9：为什么要把 embedding 和 rerank 拆成独立的服务（bge-m3、bge-rerank）？

**一句话**：让模型加载和推理与业务逻辑解耦，可独立伸缩、独立重启。

**展开**：这两个模型（BGE-M3、BGE-reranker）体积大、加载慢、吃显存/内存。拆成独立服务后：① 后端只通过 HTTP 调用，职责清晰；② 模型服务可以单独扩容或重启，不影响聊天；③ 本地一次加载、常驻内存，避免每个请求都重新加载模型导致超时。

### Q10：你的系统是怎么保证"不卡"的？有没有并发/异步设计？

**一句话**：全链路异步 + 耗时活拆到后台队列。

**展开**：① 后端用 FastAPI 全异步（async/await），I/O 密集的数据库、HTTP 调用都不阻塞；② 流式返回（SSE），大模型吐一个字传一个字，用户很快能看到第一句；③ 耗时的记忆抽取、文档解析都丢给 ARQ 后台 worker，不在聊天请求里同步等待。

### Q11：项目里有没有"自愈/健壮性"的设计？

**一句话**：有——比如 Milvus 启动较慢时，API 会重试等待而不是直接挂。

**展开**：`milvus_store.py` 的 `init_collections` 带了连接重试（默认重试 10 次、间隔 3 秒），API 启动时如果 Milvus 还没就绪，会等待重试，而不是直接退出。类似的，索引自愈（`_ensure` 会补齐历史失败缺失的索引）、文档入库幂等（先删旧块再写）也都是健壮性体现。

### Q12：如果让你优化，你会先优化哪里？

**建议方向**（任选一二讲即可，讲"你懂的那个"）：
- **文档检索目前是全局共享**（`filter="id > 0"`），未来可给文档加 `user_id` 隔离，做私有知识库；
- **记忆抽取的自动触发**：目前是手动/接口触发，可用定时任务按"每 N 轮或空闲 M 秒"自动触发（配置里已预留了 `memory_extract_every_n_rounds`、`memory_extract_idle_seconds` 参数）；
- **摘要机制**：长期对话用摘要替代原文能进一步省 token，目前摘要字段已预留，可补齐自动摘要逻辑；
- **缓存**：对高频检索结果加缓存，减少向量库压力。

---

## 附：一句话电梯演讲（开场白备用）

> "我做的是一个**AI 角色扮演系统**，它不只是聊天——它会给 AI 一个'人设'去扮演，会**记住**你们聊过的重要事情（短期记最近、长期记关键），还能**读懂你上传的文档**、引用文档回答并告诉你答案出自哪一段。技术上，我用 **RAG 检索增强 + 混合检索 + 重排 + 双层记忆** 这套组合，让 AI 的回复既有角色灵魂，又有据可查、有记忆可循。"
