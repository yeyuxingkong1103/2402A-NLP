# 会话摘要（ShortTermMemoryStore.read_summary / write_summary）接入方案

> 批次 20 · 任务书 2.4 ｜ 交付物：方案（**不写代码**，待审后另开批次实施）
> 结论先行：**不删**。能力本身完整（读写 + TTL + 测试齐备），缺的只是"接线"。
> 接线的两处落点：**写 = 回答完成后（API 落库层）**，**读 = 检索改写 + 提示词装配**。

---

## 一、现状（代码事实，先确认"为什么它现在是空的"）

| 能力 | 位置 | 现状 |
|---|---|---|
| 消息读写 | `app/memory/short_term.py:62 / :78`（`append_message` / `read_messages`） | ✅ **已接线**：写 `app/api/chat_persistence.py:103-112`；读 `app/retrieval/assembly.py:211` |
| 摘要读写 | `app/memory/short_term.py:92 / :103`（`write_summary` / `read_summary`） | ❌ **零接线**：全项目只有 `tests/` 调用（`test_short_term_memory.py`、`test_retrieval_factory_smoke.py`） |

**关键事实（决定方案的形状）**：

1. Redis key 与窗口：`short_memory:{user_id}:{session_id}:messages`（List，`max_messages=20`，即 **10 轮问答**，在 `app/retrieval/assembly.py:169` 硬编码）与 `short_memory:{user_id}:{session_id}:summary`（String，整键覆盖写）。两者 TTL 相同 = `settings.session_ttl_seconds`（默认 7200s），写入即续期。
2. **消息窗口是"滑动窗口 + 硬截断"**：`append_message` 用 `ltrim(key, -20, -1)`（`short_term.py:74`）。也就是说，第 11 轮开始，**第 1 轮的内容被永久丢掉**，没有任何地方保存它 —— 摘要正是补这个洞的唯一手段。
3. **提示词里目前没有任何历史**：`app/chat/prompt_builder.py:40 build_chat_messages(question, context_block, memory_block)` 只吃"法源清单 + 用户问题 + 长期记忆段落"。也就是说，多轮上下文**今天只影响检索**（查询改写消解指代），**不影响生成**。
4. 短期记忆今天唯一的消费方是查询改写：`assembly.py:211` 读 `read_messages` → `QueryRewriter.rewrite(question, messages)`。
5. 已有可照抄的接线范式（批次 14 长期记忆）：读在 `ChatService.chat()` 第零步（`chat/service.py:118`），写在回答之后（`chat/memory_hooks.py:52`），**失败只 warning、可选后台线程**（`memory_write_async`）。

---

## 二、方案

### 2.1 什么时候写摘要（触发条件）

**触发规则（建议值，一处常量集中配置）**：

```
窗口已满（len(messages) == max_messages == 20，即对话满 10 轮）
  且 距上次摘要又累积 ≥ 6 条消息（3 轮）
  → 触发一次增量摘要
```

- **为什么"窗口满"才触发**：窗口没满时，最近 10 轮原文都还在 Redis 里，改写器直接读得到，摘要没有信息增量。窗口一满，最早的对话开始被 `ltrim` 丢弃 —— 那是"必须压缩"的时间点。
- **为什么再加"≥6 条"的节流**：窗口满之后是**每一轮都会淘汰**，若不加节流就是每轮一次 LLM 调用。3 轮一次把成本压到 1/3，同时摘要滞后最多 3 轮，可接受。
- **摘要输入**：旧摘要（若有）+ 本次即将被淘汰区间的消息（`messages[: len(messages) - KEEP_RECENT]`，`KEEP_RECENT = 6`，即保留最近 3 轮原文不动）。
- **摘要输出**：≤ 300 字，固定四要素：① 用户身份/处境 ② 已讨论的问题 ③ 已给出的结论要点 ④ 仍未解决/待补充的事实。
- **阈值集中放一个常量模块**（如 `app/memory/summary_policy.py`）：`SUMMARY_TRIGGER_AT = full_window`、`SUMMARY_MIN_NEW_MESSAGES = 6`、`SUMMARY_KEEP_RECENT = 6`、`SUMMARY_MAX_CHARS = 300`，便于一处调参、便于测试注入。

### 2.2 写进哪一层 + 失败如何处理

**写的位置：API 持久化层（`app/api/chat_persistence.py::persist_turn` 之后）**，理由：

- 这里是"一轮对话已完成"的唯一收敛点：消息已落 MySQL、已回写 Redis，语义完整。
- 放在能力层（`ChatService.chat()` 里）会让摘要与"检索/生成"耦合，且同步 `chat()` 与流式 `chat_stream()` 两条路径都要各写一遍。
- 与长期记忆写入并列，都在"回答已产出之后"，符合项目"绝不因为记忆功能改动影响回答"的既有原则。

**失败处理（硬要求：不许中断回答）**：

| 环节 | 失败策略 |
|---|---|
| 读 Redis 列表长度 | 失败 → 跳过本次摘要（warning），回答不受影响 |
| LLM 摘要调用 | 超时/报错 → 保留旧摘要不动（**绝不写空串覆盖**），warning |
| 写回 Redis | 失败 → warning；下次满窗还会再触发 |
| 整体 | 外层 `try/except Exception` 全包（照抄 `memory_hooks.py:90-95` 的写法） |

**时序**：回答已经发给用户之后才做，且**建议默认后台线程**（照抄 `memory_write_async` 开关）。最坏情况是摘要比回答晚几秒落地，不影响首字延迟。

### 2.3 读出来怎么用（两处注入）

**用法 A —— 检索侧（消解指代，优先做）**：`app/retrieval/assembly.py::rewrite_query_safely`

- 现状：`read_messages` 为空（如 Redis 重启、会话很久没说话）或很短（< 4 条）时，改写器没有上文可用 → 省略式提问（"那这个钱谁出？"）改写失败 → 检索跑偏。
- 改动：在消息不足时把 `read_summary` 的结果作为"更早的上下文"一并交给改写器（`QueryRewriter.rewrite` 增加一个可选 `background_summary: str | None = None` 参数）。
- 上限：只用 300 字内的摘要，不拼接原文，避免把 prompt 撑大。

**用法 B —— 生成侧（让回答知道"聊过什么"）**：`app/chat/prompt_builder.py::build_chat_messages`

- 新增可选段落，注入位置：**长期记忆段落之后、`# 法源清单` 之前**（与 memory_block 同级，都属于"背景"而不是"依据"）：

```
# 本次会话前情（早前轮次的压缩摘要，仅供理解上下文，不得替代法源清单）
<≤300 字的摘要>
```

- 为什么不放在法源清单之后：法源清单与用户问题的相对位置是铁律（模型先看依据再看问题，引用才不漏编号），**不能动**。
- 与最近 N 轮的关系：**最近 3 轮原文不进提示词**（现状就不进，本方案不扩大改动面）；摘要负责"更早的、已被窗口淘汰的"内容。两者不重叠。
- 上限：摘要注入硬上限 300 字（约 450 tokens），超过则截断并在日志里记一次 warning。
- 安全：摘要文本属于"数据不是指令"，已被系统提示词铁律 7 覆盖；摘要段落里追加一句同义声明更稳。
- 开关：加 `SESSION_SUMMARY_ENABLED`（默认 **false**，与 `SYNONYM_EXPANSION_ENABLED` 同风格）。关闭时行为与现状**逐字一致**（不读、不写、不注入），方便一键回退与对拍。

### 2.4 成本

| 项 | 估算 |
|---|---|
| LLM 调用次数 | 每满 10 轮后，每 3 轮一次 → 一个 30 轮的会话约 **7 次** |
| 单次 token | 输入 ≈ 旧摘要 300 字 + 6 条消息（约 1200 字）≈ **2.3k tokens**；输出 ≈ 300 字 ≈ **0.5k tokens** |
| 延迟 | 后台执行，**不增加用户等待** |
| 费用 | 当前 LLM 是自建 Qwen2.5-14B（AutoDL RTX 4090），边际成本≈电费，可忽略；若换按量付费 API，按现价乘上面 token 量即可，一个 30 轮会话约为"7 次短调用"的量级 |
| 额外风险 | 每 3 轮多一次模型调用 → 自建 GPU 上表现为偶发占用，建议放后台线程 + 并发上限 1（同一会话串行） |

### 2.5 怎么验收（实施批次照此执行）

1. **单元（Fake Redis + Fake LLM）**：造 20 条消息 → 断言触发一次 `write_summary`；再造 2 条 → 不触发（节流）；再造 4 条 → 再触发（增量输入含旧摘要）。
2. **失败路径**：LLM 抛异常 → 断言旧摘要未被污染、回答链路 413+ tests 全绿无变化。
3. **注入后的提示词真实内容**：临时脚本打印 `build_chat_messages(...)` 的完整输出，贴出含 `# 本次会话前情` 段的原文（这是最直观的证据）。
4. **摘要命中 / 未命中对比**：同一句省略式追问（如"那这个钱谁出？"）跑两次：
   - (a) 关闭摘要 / 冷启动（消息已被 TTL 清掉）：记录改写后的查询与检索 top5；
   - (b) 有摘要：记录同样两项。
   期望：改写后的查询包含摘要里的实体（如"经济补偿"），top5 命中率提升；两段输出都贴进报告。
5. **Redis 原始内容**：`redis-cli get short_memory:{uid}:{sid}:summary` 贴出真实摘要。
6. **回归**：`PYTHONPATH= python -m pytest tests -q`（当前基线 **468 passed**，不得减少）+ `python -m app.cli.demo_ask "经济补偿怎么算"` 跑通。

---

## 三、改动清单（预估，实施批次按此评估）

| 文件 | 改动性质 |
|---|---|
| `app/memory/summary_policy.py` | **新增**：触发阈值与上限常量 |
| `app/memory/summary_service.py` | **新增**：`maybe_update_session_summary(...)`（读窗口 → 判触发 → 调 LLM → 写回；全包 try/except） |
| `app/api/chat_persistence.py` | 在 `persist_turn` 之后调用上面的函数（后台线程受开关控制） |
| `app/retrieval/assembly.py` | `rewrite_query_safely` 在消息不足时读摘要并传给改写器 |
| `app/retrieval/query_rewrite.py` | `rewrite()` 增加可选 `background_summary` 参数（默认 None，行为不变） |
| `app/chat/prompt_builder.py` | `build_chat_messages` 增加可选 `session_summary` 段落（默认 None，行为不变） |
| `app/chat/service.py` / `streaming.py` | 把摘要透传给 `build_chat_messages` |
| `app/core/config.py` + `.env.example` | `SESSION_SUMMARY_ENABLED` 等开关与阈值（补齐示例文件，见本批 2.5 报告） |
| `tests/test_session_summary.py` | **新增**：触发/节流/失败/注入四类用例 |

**风险与回退**：默认关闭；开启后若发现摘要污染回答，关掉开关即回到现状（不读不写不注入）。摘要与消息共用 TTL，最长 2 小时自然消失，不留脏数据。

---

## 四、需要你裁决的三点

1. **触发口径**：认可"满窗（10 轮）+ 每 3 轮增量"吗？也可以更保守（满窗后每 5 轮一次）。
2. **用法 A/B 的取舍**：先只做 A（检索改写，见效快、风险低），还是 A+B 一起（B 会改变回答的输入，需要重跑评测集确认 Faithfulness 与引用合规没退化）？**建议先 A 后 B**。
3. **默认开关**：建议默认关（false），验收通过后再改默认开。
