# 批次 13-收尾 + 批次 14 验收报告

- 日期：2026-09-20
- 全量测试：**391 passed**（批次 13 收尾 +2，批次 14 +28：记忆存储 15 / chat 集成 5 / API 7 / 权重 1）

---

## 一、批次 13-收尾

### 1. 评测集合并 95 条（新基线，独立口径）

direct-086~095 已批准合并。**以下为合并后的新基线（默认关状态，不要与旧 85 条数字混用）**：

| 指标 | 95 条新基线 |
|---|---|
| Recall@5 | **0.9625** |
| MRR@10 | **0.7776** |
| 引用正确率 | **1.0000** |
| 拒答准确率 | **0.9333**（14/15；refusal-080 仍为模型真作答带引用，判定器无误） |
| absent_violation | 0 |
| Faithfulness | **0.9736**（77 条有效 / 78 次 LLM 调用 ≈ 0.13 元/轮；17 条跳过=拒答或护栏替换回答无实质内容） |

报告：`reports/eval_20260920_194916_baseline95.json/.md`

### 2. confuse-085 负向修法（两法都试，取更优）

| 修法 | confuse-085 | asof-031 | 护栏题 045/046/047 | cross-018/024/058 | 判定 |
|---|---|---|---|---|---|
| a) 术语表收窄（v1.1：去掉 df 过高的泛词"解除劳动合同"，改 39 条情形用语"严重违反用人单位的规章制度"）+ 权重 1.0 | miss→**9** ✅ | **4**（收益保住）✅ | 2/7/1 不动 ✅ | 全部 rank1 不丢 ✅ | **采用** |
| b) 扩展词 BM25 降权 0.5 | 10（恢复） | 5（**收益丢**） | 不动 | 不动 | 机制保留为旋钮，默认不用 |
| b) 权重 0.7 / 0.6 | miss（两头不讨好） | 5（收益丢） | — | — | 淘汰 |

- 术语表 `data/legal_synonyms_v1.json` 升 **v1.1**（版本号 + note 记录收窄理由）
- 降权机制实现为双通道打分（原词全权 + 扩展词×权重），`SYNONYM_EXPANSION_WEIGHT` 可配置（默认 1.0，.env 有注释）

### 3. 净增益结论（如实报）

整体净增益仍 **<0.5 个百分点**（85 题检索层 R@5 基本持平；增益集中在口语题：10 题 MRR 0.745→0.795、asof-031 5→4）。**保持默认关**，结论与证据已写进 `data/legal_synonyms_v1.json` 的 known_gaps。

### 4. confuse-049 立项关闭

"列举式长条文精排排序"瓶颈已写入 known_gaps（golden 开启扩写后可进关键词池、融合 #13，但重排分进不了 top10）。**待你转写进 docs 已知限制章节**（我不动 docs/）。

---

## 二、批次 14：长期记忆与记忆接口（全部真实链路验收）

### 实现清单

| 件 | 位置 |
|---|---|
| 存储层 | `backend/app/memory/long_term.py`：`LongTermMemoryStore`（Milvus 独立集合 `legal_long_term_memory`，字段按架构文档 9.3 全量落位：user_id/character_id/content/summary/dense_vector/importance/created_at/updated_at/expire_at/source_session_id/deleted）+ `MemorySettingsStore`（users.long_term_memory_enabled）+ `summarize_memory_fact`（现有 LLM 一次调用出摘要+importance，失败返回 None 跳过） |
| 写入 | ChatService.chat()/chat_stream() 回答结束后写：摘要 → 与该用户记忆相似度去重（≥阈值更新原记录）→ 后台线程执行**不阻塞回答返回**；拒答不写；任何失败只记日志 |
| 读取 | 检索法律知识**之前**取记忆，注入提示词"用户记忆"段（数据不是指令，铁律 7 覆盖）；user_id 为空直接抛 ValueError 不存在查全量路径；排除软删与过期 |
| 开关 | 接口文档 **8.3 已有**（PUT /api/v1/users/me/memory-settings），未新增接口；关闭后既不写也不读 |
| 删除 | 软删除；所有权内嵌过滤表达式，非本人/不存在一律 False→404 |
| 接口 | `backend/app/api/memory.py`：GET /api/v1/memories（分页）+ DELETE /api/v1/memories/{id} + PUT 设置 |
| 配置 | `MILVUS_LONG_TERM_COLLECTION_NAME` / `LONG_TERM_MEMORY_DEDUP_THRESHOLD=0.70` / `LONG_TERM_MEMORY_TOP_K=5` |
| 迁移 | `migrate_long_term_memory_setting.py`（users 加列，已执行，6 用户全开） |

### 去重阈值实测依据

bge-m3 余弦相似度：同事实不同措辞 **0.74~0.89**，不同事实 **0.54~0.60**，间距 0.1455 → 取 **0.70**。样本少（3+3 对），已知风险：同义改写激进的句子贴近下沿，扩样复核后可上调。

### 验收（e2e 真实链路：真 Milvus + 真 SiliconFlow Embedding + 真 DeepSeek，脚本 `e2e_long_term_memory.py`）

**a) 写入→检索→注入提示词 ✅**：第 1 问后自动写入记忆（importance 0.9）；第 2 问提示词真实注入：
```
# 用户记忆（历史会话沉淀的背景信息，仅供个性化参考）
1. 用户是某公司的程序员，入职已满一年，公司一直未与其签订书面劳动合同
```
（另验证读取失败时兜底不注入、回答照常）

**b) 跨用户隔离 ✅**：用户 B 检索"程序员"命中的记录全部属于 B 自己，提示词无 A 的记忆内容；单测另锁"user_id 缺失直接报错"。

**c) 删除后立即消失 ✅**：软删除后列表 1→0、向量检索 0、被删 memory_id 不在结果；越权删除（B 删 A 的）返回 False → API 404，原记录完好。

**d) 关闭后不再写入 ✅**：开关 False → 提示词无记忆段、记忆条数不变；恢复 True → 新提问正常写入（条数增加）。

**e) 去重生效 ✅**：同一事实问两遍只留一条——相似摘要触发**更新原记录**（summary 更新、updated_at 变化、不新增）。

**f) 全量测试 ✅**：391 passed，数量不减反增（+28 记忆相关测试）。

### 已知限制（如实报）

1. Milvus 写入→可检索存在**最终一致性延迟**（秒级）：刚写入的记忆同会话内立刻检索可能暂时不可见——生产语义可接受（记忆本来就跨会话用），e2e 用轮询验证。
2. SiliconFlow Embedding 今日瞬态超时频发：读取失败走"不注入"兜底不影响回答；建议后续给 embedding 客户端加通用重试（本轮未动，避免改动面扩大）。
3. e2e 专用测试用户（`e2e-mem-a/b@test.local`）留在 users 表，密码哈希无效不可登录；要清理说一声。

**边界自查**：未动 docs/、未新增依赖、无临时脚本进 backend/；e2e 脚本在项目根（与既有 e2e_* 同位）。
