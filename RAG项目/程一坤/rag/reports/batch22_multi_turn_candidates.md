# 批次 22 · multi_turn 评测题候选（**已审批并落地**）

> **落地状态（批次 23）**：4 条候选 + 备用 `multiturn-005` 已全部写入
> `data/evaluation/eval_set_v1.jsonl`（95 → 100 条），题面与锚点词**逐字未改**；
> `evaluation/run_eval.py` 的 5 项改动已实施（含评测内回写短期记忆、`try/finally` 清会话），
> 并新增 `--no-context-writeback` 反向对照开关。
> **实测发现**：候选里的锚点词有 2 条（003「女职工/产假」、004「竞业限制」）在真实改写输出里
> 不出现 —— 改写器注入的是**上一轮问题原文**而非名词短语，详见
> `reports/batch23_multi_turn_acceptance.md` 第七节待裁决第 1 条。
> 本文保留为当时的方案记录，**不要**再据此判断"当前实现"。

> 用途：把"多轮 + 摘要 + 查询改写"纳入评测集。本文只出**题目草案与改动清单**，
> **没有**写入 `data/evaluation/eval_set_v1.jsonl`（批次 23 已按本方案落地，见上方说明）。

---

## 一、为什么现有 `followup_rewrite` 题目验不到改写（本轮实测发现）

现有评测集已有 10 条 `followup_rewrite`（字段 `context` = 上一轮问句，`question` = 追问），
`run_eval.run_item` 也确实会拿 `context` 先走一遍问答（`evaluation/run_eval.py:226`）。
**但它测不到"改写"**，原因在写入侧：

```
short_term_memory.append_message(...)  的唯一调用方是 app/api/chat_persistence.py（persist_turn）
persist_turn 的唯一调用方是 app/api/chat.py:201（SSE 接口）
run_eval 直接调 ChatService.chat(...)  →  从不写短期记忆
```

于是改写器读到的上下文**永远是空的**：`rewrite_query_safely` 里
`short_term_memory.read_messages(user_id, session_id)` 返回 `[]` → `changed=False`，
追问退化成"没有上文的单轮问题"，`followup_rewrite` 实际只验了"追问能不能检索到"。

**验证**：本轮 14 轮真实多轮探针（`scripts/eval/multi_turn_summary_check.py`）
按真实顺序回写短期记忆后，第 11 轮观测到改写生效：
`那这个要赔多少？ → 孕妇被辞退有什么特别保护 这个要赔多少 怎么算`。
这说明"改写能力本身是通的"，缺的是**评测入口把链路接上**。

---

## 二、建议的题面结构（新类型 `multi_turn`）

```jsonc
{
  "id": "multiturn-001",
  "type": "multi_turn",
  "turns": [                                  // ≥2 轮，按顺序在同一会话里跑
    {
      "question": "签三年期的劳动合同，试用期最长不能超过多久？",
      "golden": [{"law": "劳动合同法", "article": "第十九条"}]
    },
    {
      "question": "那这个期间工资最低不能低于多少？",   // 带指代/省略的追问
      "golden": [
        {"law": "劳动合同法", "article": "第二十条"},
        {"law": "劳动合同法实施条例", "article": "第十五条"}
      ],
      "source_quote": "劳动者在试用期的工资不得低于本单位相同岗位最低档工资的80%或者不得低于劳动合同约定工资的80%，并不得低于用人单位所在地的最低工资标准"
    }
  ],
  "rewrite_expect": {                         // 只在指定轮次断言
    "turn": 2,
    "expect_changed": true,
    "must_contain_any": ["试用期"],            // 改写后查询应补出的主题锚点
    "must_not_contain": []                    // 反向断言（可选）
  },
  "notes": "第 2 轮不带主语，只靠'这个'指代；不改写则检索会掉到《工资支付暂行规定》"
}
```

设计要点（每条都为了"能真的判定改写是否生效"）：

1. **第 2 轮必须是省略/指代句**，去掉主题词后单独看无法检索（这是改写的用武之地）；
2. **第 1 轮与第 2 轮的 golden 必须不同**，否则"改写生效"与"碰巧命中"分不开；
3. `rewrite_expect.must_contain_any` 用**主题锚点词**（如「试用期」「竞业限制」）判定改写质量；
4. 断言对象是 `RetrievalResult.query_rewrite`（检索结果里已带 `QueryRewriteResult`，
   含 `changed` / `rewritten_query` / `reasons`），**不需要改检索层代码**。

---

## 三、候选题目（4 条，golden 已逐条核对存在于库内 approved 版本）

| id | 第 1 轮（建立话题） | 第 1 轮 golden | 第 2 轮（指代追问） | 第 2 轮 golden | 改写应补出的锚点 |
|---|---|---|---|---|---|
| `multiturn-001` | 签三年期的劳动合同，试用期最长不能超过多久？ | 劳动合同法 第十九条 | 那**这个期间**工资最低不能低于多少？ | 劳动合同法 第二十条 / 实施条例 第十五条 | 试用期 |
| `multiturn-002` | 工作满十年，年休假有几天？ | 职工带薪年休假条例 第三条 | **那些天**没休成，钱按什么标准算？ | 职工带薪年休假条例 第五条 | 年休假 |
| `multiturn-003` | 公司可以辞退怀孕的女员工吗？ | 女职工劳动保护特别规定 第五条 | 那**她**能休多久？ | 女职工劳动保护特别规定 第七条 | 女职工 / 产假 |
| `multiturn-004` | 公司要我在离职后不去同行，需要给我钱吗？ | 劳动合同法 第二十三条 | 那**这个**最长能约定几年？ | 劳动合同法 第二十四条 | 竞业限制 |

要点与预期：

- **001**：第 2 轮去掉"试用期"后是"工资最低不能低于多少"——不改写会命中《工资支付暂行规定》
  这一类的通用工资条款，命中不了试用期专项条文（第十九/二十条成对出现，是天然的判别点）。
- **002**：第 1 轮问天数（第三条），第 2 轮问钱（第五条），**答案来源条号不同**，
  改写生效与否直接体现在 `golden_rank`。
- **003**：第 2 轮的指代最弱（"她"+无主题名词），**是最能暴露"改写没生效"的一条**；
  不改写时"那她能休多久？"几乎无检索价值。
- **004**：第 1 轮问"要不要给钱"（第二十三条：约定经济补偿），第 2 轮问"最长几年"
  （第二十四条：不得超过二年），同法不同条，判定清晰。

`multiturn-005`（备用，若你认为 4 条不够）：第 1 轮「公司一直没跟我签书面劳动合同，我能要双倍工资吗？」
（劳动合同法 第八十二条）→ 第 2 轮「**这笔钱**从什么时候开始算，算到什么时候为止？」
（实施条例 第六条）。注意：它与现有 `followup-026` 主题重合，属"同题不同测法"（单轮 context
版 vs 真实多轮版），是否保留由你定。

---

## 四、`evaluation/run_eval.py` 需要的改动（最小集，待批准后动手）

| # | 改动 | 说明 |
|---|---|---|
| 1 | 新增 `multi_turn` 分支（按 `turns` 逐轮跑） | 每轮：先 `retrieval_service.retrieve(...)`（拿 `query_rewrite` + golden_rank），再 `chat_service.chat(...)` |
| 2 | **每轮结束后回写短期记忆**（等价 `persist_turn` 的 Redis 写侧） | `append_message(user)` + `append_message(assistant)`，窗口 20 条与生产一致；**不写 MySQL**（避免评测数据污染库） |
| 3 | 记录每轮 `query_rewrite` | `changed` / `rewritten_query` / `reasons` 落进 `detail["turns"][i]` |
| 4 | 新增两项判定 | `rewrite_hit`（该轮 `changed` 是否等于 `expect_changed` 且锚点命中）、`turn_hit_at_5`（每轮 golden 是否 top5） |
| 5 | 会话清理 | 该题跑完删掉 `eval_<id>` 的短期记忆 key（沿用 `delete_session_memory`） |

**不改**的部分：检索层、改写器、摘要服务、`prompt_builder` 一行都不动——
`query_rewrite` 已经挂在 `RetrievalResult` 上，评测侧只是"把链路按真实顺序连起来 + 读出来"。

### 指标（建议随题目一起定口径）

| 指标 | 定义 |
|---|---|
| `multi_turn_rewrite_rate` | 指定轮次里 `rewrite_hit=True` 的比例（分母 = 有 `rewrite_expect` 的题目数） |
| `multi_turn_turn2_hit_at_5` | 第 2 轮 golden 落在 top5 的比例（**这是"改写是否有用"的落点**） |
| `multi_turn_context_carryover` | 第 2 轮检索命中的条号是否**只在有上文时才可能命中**（人工复核项，先只记录不判定） |

### 验收口径（建议）

1. 改造后跑这 4 条：`rewrite_rate = 4/4`、`turn2_hit_at_5 = 4/4`；
2. **反向对照**（证明测试有效）：临时禁掉上下文回写（不 `append_message`）重跑，
   `rewrite_rate` 应掉到 0、`turn2_hit_at_5` 应明显下降 —— 两个数都掉才算"测到了"；
3. 全量评测集（95 + 新增）跑一遍，确认既有 95 条指标不回归
   （新增类型不改动既有路径，预期逐位相同）。

---

## 五、请你裁决

1. 这 4 条（+ 备用的 `multiturn-005`）题目、期望锚点词，**要不要改**？
2. `rewrite_expect` 这种"断言方式"接受吗？还是你希望只记录不断言、事后人工看？
3. 第四节 5 项改动是否批准（尤其第 2 项"评测里回写短期记忆"——它是让改写生效的关键，但也意味着
   评测过程会写 Redis 短期记忆 key，跑完即删）？
4. 指标口径按上表定，还是要加/减？

裁决通过后我按"先交审 → 落地 → 正反两轮对照 → 全量不回归"的顺序做完再汇报。
