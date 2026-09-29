# 批次 24 验收报告：五项裁决落地 + run_eval 拆分

> 边界遵守：不改 `docs/`（发现 1 处过期表述，登记在第七节等你处理）；
> 临时脚本全在系统临时目录（未进 `backend/`）；做完停下汇报。

---

## 结论速览

| # | 你的裁决 | 结果 |
|---|---|---|
| ① | 锚点词改方案 A（003→`["怀孕的女员工"]`、004→`["离职后不去同行"]`） | ✅ 只改 `must_contain_any`，题目一字未改；正向 `rewrite_rate` **2/4 → 4/4** |
| ③a | 删掉"去疑问后缀"死分支 | ⚠️ **按你的指令删了 → 2 条既有单测变红，已回滚**；改按证据只删**真正死项**（`？`/`?`）—— 见第二节，请你确认这个改法 |
| ③b | 「怎么算」尾巴加/不加 A/B，用数字决定 | ✅ 三轮 × 两模式**逐位复现**；加尾巴更差（子集 MRR@10 0.6444→0.6278）→ **已去掉** |
| ② | 补 2~3 条"纯指代"题（先出草案） | ✅ **仅出草案未落地**：3 条候选 + 实测数字，见 `reports/batch24_pure_anaphora_candidates.md` |
| ⑤ | 清 11 个 Redis 残留 key | ✅ 清前报备 11 条 → 删 11 → `DBSIZE 44 → 33`，白名单残留 0；**另发现 33 个未授权 key 未动**（见第六节） |
| ④ | `run_eval.py` 拆分（零逻辑改动 + 逐条一致） | ✅ 628 → **213 行**，拆出 5 个模块；源码逐字相同 + 7 份 payload 逐字节一致；全量 100 条 A/B：**检索侧 99/100 逐条一致**（唯一 1 条已取证归因上游重排边界抖动）、**改写侧 5/5 逐字一致**，见第五节 |

**两个必须先说的**（与你的指令前提不一致，有实测证据）：

1. **③a 的前提不成立**：批次 23 报告里"该分支永不触发"这句话**说错了**。
   实测只有 `"？"` / `"?"` 两个标点项结构性不可达；7 个**词**后缀（吗/呢/怎么算/是什么/有哪些/如何/怎么回事）
   **是可达的** —— 输入"公司可以辞退怀孕的女员工吗"（无尾标点）就会命中，
   `backend/tests/test_query_rewrite.py` 有 **2 条用例正依赖它**。
   我按你的指令删了整段 → **2 条测试立刻变红**，且改写结果出现重复的「怎么算」→ **已回滚**。
   详见第二节。
2. **③b 去掉尾巴后，`run_eval.py` 里那句「怎么算」分支变成死代码**，我一并删掉了
   （去掉尾巴后，"怎么/如何/是什么/有哪些"分支与默认分支返回同一个字符串）。
   这与 ③a "删死代码"是同一条纪律，但属于你没点名的额外改动，特此声明。

---

## 一、① 锚点词改方案 A

```bash
# 只改 must_contain_any，题目一字未改
multiturn-003: ["女职工", "产假"]   → ["怀孕的女员工"]
multiturn-004: ["竞业限制"]         → ["离职后不去同行"]
```

**改动范围证明**（归一化行尾后逐条 JSON 对比）：

```
内容不同的条目数 = 2（只有 multiturn-003 / multiturn-004）
除 must_contain_any 外完全一致 = True（两条都是）
```

**重跑正向结果**（`reports/eval_20260921_144515_b24_mt_on.json`）：

| 指标 | 改前（批次 23） | 改后 |
|---|---|---|
| `rewrite_rate` | 0.5000（2/4） | **1.0000（4/4）** ✅ |
| `turn2_hit_at_5` | 1.0000（5/5） | 1.0000（5/5） |
| `context_carryover` | 4/5 | 4/5 |

逐题改写结果（原文）：

```
003 turn2 → '公司可以辞退怀孕的女员工吗 她能休多久 怎么算'   ← 含"怀孕的女员工" ✔
004 turn2 → '公司要我在离职后不去同行 需要给我钱吗 这个最长能约定几年 怎么算'  ← 含"离职后不去同行" ✔
```

> ⚠️ 一个**未申请但必须报备**的副作用：`eval_set_v1.jsonl` 原本是**混合行尾**
> （前 95 行 CRLF、批次 23 追加的 5 行 LF），编辑工具把整份文件归一成了 **LF**。
> **内容零变化**（已按上面的 JSON 逐条对比证明），但字节层面不只是那 2 个字段。
> 若要恢复 CRLF 混合状态，我可以还原（但更建议保持统一的 LF）。

---

## 二、③a 删"去疑问后缀"死分支 —— **指令前提被证据推翻，已回滚并改用证据版**

### 2.1 证据：哪些真死、哪些没死

`_extract_topic` 的执行顺序是「标点替换 → 判后缀」，所以**先**把 `[，。！？?；;：:、,.!]+` 换成空格，
**后**才判 `endswith`：

| 后缀 | 被标点规则改写为 | 改写后仍可能以它结尾？ | 结论 |
|---|---|---|---|
| `？` | `' '` | 否 | **结构性不可达（真死）** |
| `?` | `' '` | 否 | **结构性不可达（真死）** |
| 怎么回事 / 怎么算 / 是什么 / 有哪些 / 如何 / 吗 / 呢 | 原样 | 是 | **可达** |

实测可达/不可达（裸文本 vs 带尾标点）：

| 输入 | 是否命中 | 最终主题词 |
|---|---|---|
| `公司可以辞退怀孕的女员工吗`（**无**尾标点） | **命中「吗」** | `公司可以辞退怀孕的女员工` |
| `公司可以辞退怀孕的女员工吗？`（**有**尾标点） | 不命中 | `公司可以辞退怀孕的女员工吗` |
| `那试用期呢`（无尾标点） | **命中「呢」** | `那试用期` |
| `经济补偿怎么算`（无尾标点） | **命中「怎么算」** | `经济补偿` |
| `经济补偿怎么算？` | 不命中 | `经济补偿怎么算` |

即：**它是"被尾标点挡住、时灵时不灵"的活代码，不是死代码。**
（评测语料上命中 0 次 —— 因为 115 条语料里的"上一轮原文"尾字都是标点。**"未触发"≠"不可达"。**）

### 2.2 按指令删整段 → 两条既有单测变红

```
FAILED tests/test_query_rewrite.py::test_rewrites_implicit_reference_using_latest_user_topic
FAILED tests/test_query_rewrite.py::test_retrieval_uses_rewritten_query_when_short_term_context_is_available

AssertionError: assert '经济补偿怎么算 12 期 怎么算' == '经济补偿 12 期 怎么算'
                    ↑ 主题词里的"怎么算"没被剥掉，和尾部拼出来的重复了一次
```

删除整段是**行为变更 + 质量退化**（改写结果出现重复的「怎么算」），不是死代码清理。
→ **已回滚**（md5 回到 `f0e3e9494e99766383f33b615733e6db`）。

### 2.3 按证据只删真死项

改成：`_QUESTION_SUFFIXES = ("怎么回事", "怎么算", "是什么", "有哪些", "如何", "吗", "呢")`
（去掉 `"？"`、`"?"`），并在常量上方写明"别再往里加标点，加了也判不中"。

**零行为变化证明**：对 115 条语料 + 7 条会命中后缀的构造输入（共 122 条），
新旧 `_extract_topic` 输出**差异 0 条**；`tests/test_query_rewrite.py` **8 passed**。

> **请你确认**：这算"删掉了死分支"（真死的那两个）还是"没听指令"？
> 若你仍要删整段，我就删 —— 但需要同步改掉那 2 条单测的期望值，
> 并接受"……吗"这类无尾标点输入的主题词会多带一个疑问词。

---

## 三、③b 「怎么算」尾巴 A/B（10 条 followup_rewrite + 5 条 multi_turn）

做法：**不改生产代码**，在临时驱动里进程内替换 `QueryRewriter._compose_query` 这一个静态方法
（`TAIL_MODE=on/off`），其余（检索/问答/判定/汇总）完全是生产实现；加/不加**各跑三轮**。

| 指标 | 带尾巴（现状） | 不带尾巴 | 三轮是否复现 |
|---|---|---|---|
| Recall@5 | 1.0000 | 1.0000 | 三轮逐位一致 ✔ |
| **MRR@10** | **0.6278** | **0.6444** | 三轮逐位一致 ✔ |
| `rewrite_rate` | 1.0（分母4） | 1.0（分母4） | ✔ |
| `turn2_hit_at_5` | 1.0（分母5） | 1.0（分母5） | ✔ |

**逐题对照（6 次运行，全部一致）**：

| 题号 | 带尾巴 rank | 不带尾巴 rank |
|---|---|---|
| followup-025 / 026 / 027 / 028 / 029 / 030 / 069 / 070 / 071 / 072 | 1/3/1/1/3/2/1/2/1/3 | 完全相同 |
| multiturn-001 / 002 | 1 / 2 | 完全相同 |
| **multiturn-003** | **4** | **3** ⬆ |
| **multiturn-004** | **3** | **2** ⬆ |
| multiturn-005 | 3 | 完全相同 |

**结论（按"用数字决定"）**：13/15 逐题不变，2 题各提升 1 位，**无一条变差**，
MRR@10 提升 +0.0166，且三轮复现（不是采样波动）→ **去掉尾巴**。

已落地：
- `_compose_query` 去掉追加「怎么算」，并顺手删掉因此变成死代码的"疑问结构"分支；
- `tests/test_query_rewrite.py` 两条期望值随之更新（`经济补偿 12 期 怎么算` → `经济补偿 12 期`）；
- 生产代码复跑子集：15 条里 **14 条与不带尾巴三轮逐位一致**；
  唯一差异是 `followup-072`（3 → 2，方向更好）—— 该题是**已知漂移项**
  （批次 23 记录过 2→3→3），属上游波动，不是本改动引入。

---

## 四、② 纯指代题草案（**未落地**，交审）

3 条候选 + 逐条实测数字 → `reports/batch24_pure_anaphora_candidates.md`，
题目本体在 `data/evaluation/eval_set_v1_batch24_candidates.jsonl`（按约定放"候选"文件，**未**并入主集）。

设计过程中被数据推翻了两件事，都已写进草案：

1. **不能照抄「那这个呢？」**：`_compose_query` 剥指代词**只剥一个前缀就 break**，
   「那这个呢？」残留"这个"（`… 我可以解除劳动合同吗 这个`）；
   「**这个**呢？」才是干净版本（改写结果 = 上一轮原文）。这是个**可修的小缺陷**，登记待你裁决。
2. 我最初设计的 3 条**全部不达标**（正向 golden 漏召回）—— 改成"先用真实检索搜候选、
   再按 turn1排名/正向排名/反向排名 三条判据挑题"，最终三条实测：
   正向 rank **2 / 4 / 3**，反向**全部未召回**。
   → 加进去后反向 `turn2_hit_at_5` 预计从 0.8 掉到 **0.5**（判别力从 0.2 拉到 0.5）。

---

## 五、④ `run_eval.py` 按职责拆分（628 → 213 行）

| 文件 | 行数 | 职责 |
|---|---|---|
| `run_eval.py` | 628 → **213** | 路径常量 + `prepare_env` + CLI 主循环 + **再导出**（见下） |
| `legal_matching.py` | **95** | 法规名/条号匹配（检索侧判定原语） |
| `answer_judging.py` | **72** | 拒答口径（R2+R1′）+ 引用越界审计 |
| `item_runner.py` | **139** | 单问题题型执行（`with_retry` + `run_item`） |
| `aggregate.py` | **140** | `summarize` / `worst_samples`（数字口径唯一来源） |
| `render_report.py` | **88** | `render_markdown` |

（行数口径：`splitlines()`，与 `wc -l` 一致。）

新拆出的 5 个文件与 `run_eval.py` **全部 ≤300 行**。
`multi_turn_runner.py`(255) / `multi_turn_grading.py`(190) 也是既有的合规文件。

> ⚠️ 顺带发现：`evaluation/calibrate_refusal.py` **314 行**，超"单文件 ≤300 行"的约定。
> 它是**批次 24 之前的既有文件**（mtime 09-20 12:19，本批未动），
> 已登记到第七节待你决定要不要另开批拆。

### 5.1 零逻辑改动证明

**A. 源码逐字比较**（16 个函数 + 4 个常量，含 `main` 与 `prepare_env`）：

```
✔ prepare_env / normalize_article / canonical_title / title_matches / article_matches
✔ refused_by_text / refusal_failure_reason / citation_audit / first_golden_rank / absent_violation
✔ with_retry / run_item / summarize / worst_samples / render_markdown / main
✔ REFUSAL_PHRASES / CITATION_PATTERN / STATUTE_BASIS_PATTERNS / LAW_ALIASES
结论：源码/常量不一致项 = 无
```

> 为了做到"逐字相同"，搬移时**连 docstring 都不许加**（我第一版给 5 个函数加了 docstring，
> 被这个校验抓出来，已还原；新增说明一律写进模块 docstring）。

**B. 行为等价（重放真实 payload）**：拿 7 份历史报告（含批次 23 全量 100 条）喂给旧实现与新实现：

```
summarize       逐字节一致 = True
worst_samples   逐字节一致 = True
render_markdown 逐字节一致 = True（5191 字符）
附加重放 6 份 b24 报告 → 全部一致
```

**C. 全量 100 条 A/B（拆分前实现 vs 拆分后实现，同一份 backend）**：

- A 侧（拆分前实现）：`eval_20260921_153118_b24_split_before.json`（退出码 0）
- B 侧（拆分后实现）：`eval_20260921_152102_b24_split_after.json`
- 逐条对照：`compare_split_ab.py` → `reports/b24_split_ab_compare.txt`
  （脚本退出码 **1**：它的判据是"检索侧必须 100% 逐条一致"，本次检出 1 条差异故判不通过，
  该差异的归因取证见下方，结论仍为拆分零逻辑改动）
- 两侧题号集合一致（100 vs 100）

| 指标 | A 拆分前 | B 拆分后 | 一致 |
|---|---|---|---|
| Recall@5 | 0.9647 | 0.9647 | ✔ |
| MRR@10 | 0.7613 | 0.7601 | ✗ −0.0012 |
| 引用正确率 | 1.0000 | 1.0000 | ✔ |
| 拒答准确率 | 0.9333 | 0.9333 | ✔ |
| 误拒率 | 0.0824 | 0.1176 | ✗ +0.0352 |
| 计分题数 / 拒答题数 | 85 / 15 | 85 / 15 | ✔ |
| 越界引用 | 0 | 0 | ✔ |
| 引用总数 | 408 | 411 | ✗ |
| 无引用回答 | 7 | 10 | ✗ |

multi_turn 段**逐位一致** ✔（count 5 / asserted 4 / graded 5 / rewrite_rate 1.0 / 分母 4 /
turn2_hit_at_5 1.0 / 分母 5 / carryover 4-of-5 / `context_writeback=[True]`；
`per_item` 五条的 `rewritten_query` **逐字相同**）。

逐条分类：

| 类别 | 条数 | 归因 |
|---|---|---|
| 检索侧（golden_rank / hit_at_5 / 越界引用 / errors） | **1** | 见下方归因取证 |
| 改写侧（rewrite_hit / rewritten_query） | **0** | — |
| 回答侧 `refusal_by_text` 翻转 | 11 | LLM 采样 |
| 仅回答文本不同 | 71 | LLM 采样 |

回答侧形态自证：11 条翻转里 **A 侧有→B 侧无 = 4 条、A 侧无→B 侧有 = 7 条、两组重叠 = 0 条**
→ 无系统性方向偏移，符合逐题独立随机。

**那 1 条检索侧差异的归因（已取证，不归因拆分）**：

`confuse-085` 两侧 `retrieval_stats` **完全相同**（vector 13 / keyword 20 / fused 31 / reranked 10），
rank 1~9 的（法名, 条号, 分数）**逐位相同**，只有第 10 位换了候选：

```
A 侧 第10位：中华人民共和国劳动合同法 第三十九条  score 15.6716
B 侧 第10位：中华人民共和国劳动合同法 第三十八条  score 0.6176
```

用**同一份后端代码**把这道题单独重放 6 次（`probe_confuse085_rank10.py` →
`reports/b24_confuse085_rank10_probe.txt`）：

```
 轮 | 第10位              | 含第三十九条
  1 | 第三十九条(15.6716) | 有
  2 | 第三十八条(0.6176)  | 无
  3 | 第三十九条(15.6716) | 有
  4 | 第三十九条(15.6716) | 有
  5 | 第三十八条(0.6176)  | 无
  6 | 第三十九条(15.6716) | 有
含第三十九条的轮次：4/6（retrieval_stats 六轮全程一致）
```

→ 同码 6 次重放，**4 次复现 A 侧取值、2 次复现 B 侧取值，两侧都在重放分布内**。
该题 `hit_at_5` 两侧同为 False，只有 `mrr10` 由 0.1 变 0 —— 对应 MRR@10 的 −0.0012
（= 0.1 ÷ 85 计分题，精确吻合），归因**重排上游在 top10 边界上的抖动**，与拆分无关。

**结论**：拆分"零逻辑改动"在"全量 100 条"这一层同样成立 ——
检索侧 **99/100 逐条一致** + 1 条已归因上游；改写侧 **5/5 逐字一致**；
指标层差异（MRR@10 / 误拒率 / 引用总数 / 无引用）全部由上述两类上游随机性解释，
**无一项可归因拆分**。

### 5.2 顺手抓到并修掉的一个真实断点

拆分后 `run_eval.py` 只从 `legal_matching` 导入了 `first_golden_rank`，
结果 **`diagnose_rerank.py` 直接 ImportError**：

```
ImportError: cannot import name 'canonical_title' from 'run_eval'
```

查全后确认 **6 个脚本**仍写 `from run_eval import ...`
（`calibrate_refusal` / `compare_synonym_ablation` / `compare_window` / `diagnose_rerank` /
`faithfulness` / `selfcheck_faithfulness`）。
→ 在 `run_eval.py` 里把这批名字**再导出**（re-export）并注明"不要当未使用清理"，
**调用方零改动**（符合 `docs/目录与命名约定.md`「拆分时保持对外函数名不变」）。

逐个 import 验证：6/6 全部可导入 ✔（`scripts/check_imports.py` 只校验 `app.*`，
这类同级模块的失效导入是它的已知盲区，所以这里单独验了一遍）。

---

## 六、⑤ Redis 残留清理

### 6.1 清前报备（白名单，逐条列出后删）

```
删除前 DBSIZE = 44
白名单待删（11 个）：
  short_memory:ablation_runner:ablation_followup-025..030:messages   10 个（TTL ≈ 2.3~2.4h，list len=6）
  short_memory:u-verify-5c:s-5c:messages                             1 个（TTL = -1 永不过期，list len=1）
```

### 6.2 清后

```
请求 11 个，实际删除 11 个
删除后 DBSIZE = 33
白名单残留（应为 0）= 0  []
```

### 6.3 ⚠️ 顺带发现：还有 **33 个 key 不在你的授权范围，我没动**

你批的是"11 个残留"，但把全库扫一遍后发现另外 33 个：

| 命名空间 | 个数 | 是什么 | 风险 |
|---|---|---|---|
| `session:*` | 28 | **真实登录会话令牌**（`{token} → {user_id, is_admin}`，TTL 剩 6.2~7.4h） | 删了会把在线用户**踢下线** |
| `user_sessions:*` | 5 | 每个用户的 token 集合（`revoke_all_sessions` 反查用） | 同上 |

归属明细：

```
user_id=cef5c9236e4456b1f51586e45f21a36f  is_admin=True   22 个
user_id=cef5c9236e4456b1f51586e45f21a36f  is_admin=False   1 个
user_id=f8d8b20fd973c6564176e9aeb7412a17  is_admin=False   3 个
user_id=probe-user-0001                   is_admin=False   1 个
user_id=probe-admin-0001                  is_admin=True    1 个
user_sessions:pending                     TTL=-1，成员 19 个（都是已过期的悬空 token）
```

- 本机**当前没有后端在跑**（8010 无监听），所以这些不是活跃连接；
- 但它们是**登录态**，不是"评测残留"，删与不删应当由你决定；
- 另有一个**真缺陷**值得登记：`user_sessions:pending` **TTL = -1 且攒了 19 个悬空 token**
  （`session:*` 已过期但集合成员没清），说明存在一处"集合永不过期"的泄漏路径
  （`SessionStore.create_session_token` 里 `expire(user_sessions_key, ttl)` 是设了的，
  所以 `pending` 这个键多半来自更早的代码/人工操作）。

**要清我就清**（会再报备一次清单）；不清就先留着。

---

## 七、待你裁决 / 需你处理

1. **③a 的改法确认**（唯一卡住"是否算按你指令办"的一条）：
   我按证据只删了 `？`/`?` 两个真死项，**保留了可达的词后缀循环**（删整段会让 2 条单测变红 + 改写结果重复「怎么算」）。
   **认可 / 还是仍要删整段**（那样需要一起改 2 条单测期望值）？
2. **③b 附带删掉的"疑问结构"分支**（去掉尾巴后它已死）—— 报备，若你要保留这个分支结构我做回来。
3. **`eval_set_v1.jsonl` 行尾被归一成 LF**（原本 95 CRLF + 5 LF 混合）—— 保持 LF / 还原混合？
4. **② 的 3 条纯指代草案**：是否采用（采用 → 100 条变 **103** 条）；
   以及 turn2/turn1 **同 golden** 是否接受。
5. **`_compose_query` 只剥一个前缀的小缺陷**（「那这个呢？」残留"这个"）—— 修不修？
   修会改变既有 10 条 followup_rewrite 的改写结果，按规矩要单独 A/B。
6. **33 个非授权 Redis key**（28 个登录会话 + 5 个集合）—— 清 / 不清？
7. **`docs/开发路线图.md:304` 有过期表述**（不改 `docs/` 是你的边界，所以只报备不动）：
   ```
   当前：✅ 已完成（实测改写为"经济补偿 12 期 怎么算"，召回第四十七条）
   现在：改写结果不再带"怎么算" → 建议改为"实测改写为'经济补偿 12 期'，召回第四十七条"
   ```
8. **`evaluation/calibrate_refusal.py` 314 行**，超"单文件 ≤300 行"约定。
   本批未动（mtime 09-20 12:19，既有文件）。**要不要另开批拆？**

---

## 八、复现命令与证据文件

```bash
PY=C:/Users/92842/anaconda3/envs/rag/python.exe   # 项目唯一正式环境（§七 规则 7）

# ① 锚点词改后重跑正向
python evaluation/run_eval.py --only multiturn- --tag b24_mt_on

# ③b 「怎么算」尾巴 A/B（加/不加各三轮；驱动脚本在系统临时目录）
TAIL_MODE=on  python <临时目录>/ab_compose_tail.py --tag b24_tail_on
TAIL_MODE=off python <临时目录>/ab_compose_tail.py --tag b24_tail_off

# ③a 死分支可达性证据
python <临时目录>/evidence_suffix_branch.py

# ② 纯指代候选搜索 + 定点验证
python <临时目录>/search_pure_anaphora.py
python <临时目录>/verify_candidates.py

# ⑤ Redis 残留清理（白名单，支持 --dry-run）
python <临时目录>/cleanup_redis_residue.py --dry-run

# ④ 拆分等价性（源码逐字 + 7 份 payload 重放）
python <临时目录>/verify_split_equivalence.py
# ④ 全量 A/B
python <临时目录>/run_full_before_split.py --tag b24_split_before
python evaluation/run_eval.py --tag b24_split_after
# ④ 全量 A/B 逐条对照
python <临时目录>/compare_split_ab.py
# ④ confuse-085 第10位抖动重放（同码 6 次）
ROUNDS=6 python <临时目录>/probe_confuse085_rank10.py

# 终验
cd backend && PYTHONPATH= python -m pytest tests -q --no-header -p no:cacheprovider \
  --basetemp="C:/Users/92842/AppData/Local/Temp/pytest_b24_full"
python scripts/check_services.py
python scripts/check_imports.py
```

| 证据文件 | 内容 |
|---|---|
| `reports/b24_suffix_branch_evidence.txt` | ③a 后缀分支可达性（真死项 vs 可达项 + 语料命中计数 + 四方案对比） |
| `reports/b24_ab_on.log` / `b24_ab_off.log` / `b24_ab_r2.log` / `b24_ab_r3.log` | ③b 六次运行原始日志 |
| `reports/eval_*_b24_tail_{on,off}[_r2,_r3].json` / `.md` | ③b 六份明细（含逐题 golden_rank） |
| `reports/b24_prod_after_tail_fix.log` + `eval_*_b24_prod_after_tail_fix.json` | ③b 生产代码复跑子集 |
| `reports/b24_pure_anaphora_search.txt` | ② 候选搜索（5 条 turn1 × 3 种指代形式的 top10 对照） |
| `reports/b24_pure_anaphora_probe.txt` | ② 首版草案不达标记录（含我的两个设计错误） |
| `reports/batch24_pure_anaphora_candidates.md` | ② 草案正文（三条 + 实测数字 + 待裁决） |
| `data/evaluation/eval_set_v1_batch24_candidates.jsonl` | ② 三条候选题目（未并入主集） |
| `reports/b24_redis_cleanup.txt` | ⑤ 清前报备 / 清后剩余数 |
| `reports/b24_split_equivalence.txt` | ④ 源码逐字 + 行为等价（7 份 payload） |
| `reports/b24_split_before.log` / `b24_split_after.log` | ④ 全量 100 条 A/B 原始日志 |
| `reports/eval_*_b24_split_{before,after}.json` / `.md` | ④ A/B 明细 |
| `reports/b24_split_ab_compare.txt` | ④ 全量 100 条逐条对照（指标表 + 差异分类 + 形态自证） |
| `reports/b24_confuse085_rank10_probe.txt` | ④ 检索侧唯一差异的归因取证（同码 6 次重放，4/6 复现 A 侧、2/6 复现 B 侧） |
