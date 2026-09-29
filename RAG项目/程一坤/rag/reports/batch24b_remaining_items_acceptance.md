# 批次 24 剩余八项 + 25-2 两项 收尾验收报告

**执行时间**：2026-09-21
**本批范围**：用户对「批次 24 八项 / 批次 26 五问 / 25-2 三项」的裁决中，**"现在能做的"**那一部分。
（批次 26 回填、批次 27 答辩材料按顺序在后续批次执行，不在本报告内。）
**边界遵守**：未改 `docs/`；临时脚本全部落在系统临时目录 `%TEMP%/b24_bak/`，未进入 `backend/`。

---

## 〇、一句话结论

| 项 | 裁决 | 本批动作 | 状态 |
|---|---|---|---|
| ③a 只删 `？`/`?` 保留词后缀 | 认可 | 核对：**批次 24 已落地**，本轮零改动 | ✅ 已符合 |
| ③b「怎么算」尾巴 | 保持删除，补 A/B 数字 | 重跑 A/B 三格，**去尾巴 MRR@10 0.6444 > 留尾巴 0.6278** → 维持删除 | ✅ 有数字 |
| 行尾 | 保持 LF | 无需动作（现文件已是 LF） | ✅ 无需动 |
| ④ 3 条纯指代题 | 不采用 | 无需动作，草案留在 `reports/` | ✅ 无需动 |
| ⑤ `_compose_query` 循环剥前缀 | 修，先 A/B | A/B 后**已落地**（无回归，且修掉注释声明的过时行为） | ✅ 已落地 |
| ⑥ `user_sessions:pending` TTL=-1 | 修 | 代码加"清理悬空成员" + 存量补 TTL | ✅ 已修 |
| ⑦ `docs/开发路线图.md:304` | 用户已改 | 无需动作 | ✅ 无需动 |
| ⑧ `calibrate_refusal.py` 314 行 | 拆，排最后 | 拆成 3 文件（127/116/117 行），等价性证明 4/4 PASS | ✅ 已拆 |
| 25-2 ① 三个 299 行文件覆盖率 | 可接受，不拆 | 无需动作 | ✅ 无需动 |
| 25-2 ② `.dev_bak25b/` 清理 | 批准 | 已从项目内移除（先转入临时回收位置） | ✅ 已清 |

**验收三项**：`598 passed` / `check_services` 退出码 0 / `check_imports` 103 文件全绿，退出码 0。

---

## 一、③a 核对：批次 24 已落地，本批零改动

用户认可的改法是"只删 `？`/`?` 两个真死项、保留 7 个可达词后缀"。核对现状（`backend/app/retrieval/query_rewrite.py:17-19`）：

```python
# 疑问后缀：_extract_topic 用它剥掉主题词尾部的疑问词（"经济补偿怎么算" → "经济补偿"）。
# 注意别再往里加 "？"/"?"：上一步已把标点替换成空格，标点类后缀永远判不中（批次 24 删除死项）。
_QUESTION_SUFFIXES = ("怎么回事", "怎么算", "是什么", "有哪些", "如何", "吗", "呢")
```

`for suffix in _QUESTION_SUFFIXES:` 循环**完整保留**，7 个词后缀都在，`"？"/"?"` 已不在列。

与改动前版本（`%TEMP%/b24_bak/query_rewrite.py.before_b24`）的 diff 只含两处：
① 删除 `"？", "?"` 两项 + 改写注释；② `_compose_query` 去尾巴（即 ③b）。
→ **③a 本身无需再动。**

---

## 二、③b A/B：去尾巴 vs 留尾巴（10 followup_rewrite + 5 multi_turn）

驱动器：`%TEMP%/b24_bak/ab_compose_modes.py`（进程内替换 `QueryRewriter._compose_query`，其余全走生产实现）。
数据集：`%TEMP%/b24_bak/b24_ab_subset.jsonl` = 10 条 `followup_rewrite` + 5 条 `multi_turn`（共 15 题）。
自检：`tail=off + strip=single` 与生产实现**逐字一致**（9 条探针全 OK），确认复刻没跑偏。

| 组合 | Recall@5 | **MRR@10** | 引用正确率 | 拒答准确率 | multi_turn 改写命中率 | 第2轮 top5 |
|---|---|---|---|---|---|---|
| tail=**on**（留尾巴 = 改前行为） | 1.0000 | **0.6278** | 1.0000 | 0.0000 | 1.0（分母 4） | 1.0（分母 5） |
| tail=**off**（去尾巴 = 现网基线） | 1.0000 | **0.6444** | 1.0000 | 0.0000 | 1.0（分母 4） | 1.0（分母 5） |

- **去尾巴比留尾巴高 0.0166（0.6444 vs 0.6278），无一项指标变差** → 按裁决 **维持删除**。
- 两个数值与批次 24 报告里记录的完全一致（0.6278 / 0.6444），**可复现**。
- 原始报告：`reports/eval_20260921_161933_b24b_on_single.json`、`reports/eval_20260921_162200_b24b_off_single.json`
- 汇总：`reports/b24b_ab_modes_summary.txt`，全量日志 `reports/b24b_ab_modes.log`

---

## 三、⑤ `_compose_query` 循环剥前缀：A/B → 已落地

### 3.1 改的是什么

原实现只剥**一个**前缀（`for ... break`），所以代码注释里写的
「剥完指代后什么都不剩（如"那这个呢？"）→ 整问都是主题」这条分支**实际走不到**：

```
单层剥离："那这个呢？" → 剥"那" → "这个呢？" → 清标点/语气词 → remainder="这个" → 返回 "主题 这个"
循环剥离："那这个呢？" → 剥"那" → 剥"这个" → remainder=""  → 返回 "主题"
```

### 3.2 A/B 结果（改前 / 改后，同样 15 题）

| 组合 | Recall@5 | **MRR@10** | 引用正确率 | 拒答准确率 |
|---|---|---|---|---|
| strip=**single**（改前） | 1.0000 | **0.6444** | 1.0000 | 0.0000 |
| strip=**loop**（改后） | 1.0000 | **0.6444** | 1.0000 | 0.0000 |

**指标中性**（无增益也无损失）。子集内**只有 2 道 multi_turn 的第 2 轮改写串发生变化**，且两题的第 2 轮 golden 排名与 top5 命中**都未变**：

| 题号 | single（改前）改写串 | loop（改后）改写串 | 改前 rank | 改后 rank |
|---|---|---|---|---|
| multiturn-001 | 签三年期的劳动合同 试用期最长不能超过多久 **这个**期间工资最低不能低于多少 | 签三年期的劳动合同 试用期最长不能超过多久 期间工资最低不能低于多少 | 1 | 1 |
| multiturn-004 | 公司要我在离职后不去同行 需要给我钱吗 **这个**最长能约定几年 | 公司要我在离职后不去同行 需要给我钱吗 最长能约定几年 | 2 | 2 |

→ **已落地**：改动方向与注释声明的设计意图一致（去掉残留指示代词），15 题无一条变差。

### 3.3 落地产物

`backend/app/retrieval/query_rewrite.py` 的 `_compose_query`：

```python
        prefixes = sorted(_REFERENCE_PREFIXES, key=len, reverse=True)
        while True:
            hit = next((p for p in prefixes if remainder.startswith(p)), None)
            if hit is None:
                break
            # 每次至少吃掉 1 个字符，循环必然终止
            remainder = remainder[len(hit) :].strip()
```

落地后逐条核对：生产实现的输出与 A/B 里的 loop 变体**11 条探针全一致（0 条不一致）**。

### 3.4 ⚠️ 残留边界（如实登记，供你判断是否要再收）

循环剥离会把"那 + 非前缀字"也一并剥掉，产生单字残渣。**注意 `那些情况…` 这类单层剥离本来就有同样残渣，不是本次引入**：

| 输入 | 单层（改前） | 循环（改后） | 说明 |
|---|---|---|---|
| `那这个呢？` | `T 这个` | `T` | ✅ 本次修复目标 |
| `那这个期间怎么算？` | `T 这个期间怎么算` | `T 期间怎么算` | ✅ 去掉残留代词 |
| `那那那个怎么算` | `T 那那个怎么算` | `T 个怎么算` | ⚠️ 退化输入，留单字 |
| `前面那个员工的工资怎么算？` | `T 那个员工的工资怎么算` | `T 个员工的工资怎么算` | ⚠️ 留单字，改后略差 |
| `那些情况怎么算` | `T 些情况怎么算` | `T 些情况怎么算` | 改前改后相同（既有问题） |

若要彻底消除单字残渣，需给"那"加"后面必须紧跟前缀词才剥"的约束——那是另一条规则改动，本批未做，等你定。

---

## 四、⑥ `user_sessions:pending`（TTL=-1）修复

### 4.1 根因

`user_sessions:<user_id>` 集合的 TTL 每次登录都被刷满，但成员对应的 `session:<token>` 会各自到期
→ **只要用户持续登录，集合只增不减，旧 token 永久堆积**。
`user_sessions:pending` 是历史上"注册时用 `pending` 当占位 user_id"那个 bug 的遗留（代码早已修好，
`tests/test_auth_api.py:223` 有专门断言，但那条断言跑在**假 Redis** 上，清不掉真实 Redis 里的残留）。
它 19 个成员**全部悬空**、TTL=-1（永不过期）。

### 4.2 代码修复（最小改动）

`backend/app/auth/session_store.py`：`create_session_token` 里在 `sadd` 之后、`expire` 之前加一步清理，
新增私有方法 `_prune_dangling_tokens`。用 `get` 判存在而非 `exists`——测试替身（`tests/conftest.py`、
`tests/test_session_store.py`）只实现了 `get`，这样不必为测试新增方法。

### 4.3 证据（`reports/b24b_redis_pending_before.txt` / `_after.txt`）

**[A] 机制对比**（可控时钟的假 Redis，连续登录 3 次、中间跨过 TTL）：

```
   修复前（旧实现）：集合成员=3  其中有效会话=1  悬空=2  集合 TTL=7200
      悬空 token 示例：8dF-jr98me4CBKLu2zCP8xiE…（session 已过期，却仍占着集合成员位）
   修复后（新实现）：集合成员=1  其中有效会话=1  悬空=0  集合 TTL=7200
   结论：修复前攒下 2 个悬空 token，修复后 0 个 —— 符合预期
   两侧集合 TTL 均被续期为 7200s（TTL 机制未被改动，只多了一步清理）
```

**[B] 真实 Redis 现状**（`redis://127.0.0.1:6379/0`，DBSIZE=33）：

```
   user_sessions:cef5c9236e4456b1f51586e45f21a36f   成员=23  悬空=0   有效=23  TTL=21254
   user_sessions:f8d8b20fd973c6564176e9aeb7412a17   成员=3   悬空=0   有效=3   TTL=21278
   user_sessions:pending                            成员=19  悬空=19  有效=0   TTL=-1
   user_sessions:probe-admin-0001                   成员=1   悬空=0   有效=1   TTL=17201
   user_sessions:probe-user-0001                    成员=1   悬空=0   有效=1   TTL=17201
   合计：5 个 user_sessions:* 键，悬空 token 19 个
   TTL<0（永不或不存在）的键：['user_sessions:pending']
```

**[C] 存量处置**（按裁决"加 TTL"，**不删 key**）：

```
   user_sessions:pending：TTL -1 → 7200（+7200s）
```

**修复后复核**：

| 检查项 | 结果 |
|---|---|
| `user_sessions:pending` TTL | **7195**（已生效，不再是 -1） |
| 该键成员数 | 19（**未删成员**） |
| 全库 key 总数 | 34（未减少；比修复前 33 多的 1 个是当时正在跑的 A/B 评测新建的会话键） |
| 全库 TTL=-1 的 key 数 | **0**（修复前含 `user_sessions:pending` 为 1） |

> 说明：按你的裁决 **33 个 key 一个没清**，`user_sessions:pending` 也是"设过期"而非"删除"。
> 若你觉得这个键干脆删掉更干净，说一声我再处理（它是 100% 悬空的历史残留）。

### 4.4 回归

`backend/tests/test_session_store.py` + `test_auth_api.py`：**18 passed**（含既有断言
`test_user_sessions_key_has_ttl`、`test_register_binds_real_user_id_not_pending`）。

---

## 五、⑧ `evaluation/calibrate_refusal.py`（314 行）拆分

### 5.1 拆法（同 `run_eval` 标准：按职责拆、函数体逐字搬移、对外函数名不变）

| 文件 | 行数 | 职责 |
|---|---|---|
| `evaluation/calibrate_refusal.py` | **314 → 127** | 只留装配 + CLI；`collect_scores` / `scan_thresholds` / `render_markdown` 三个名字**再导出**（注明"不要当未使用清理"） |
| `evaluation/refusal_scoring.py` | **116**（新） | 分数采集 `collect_scores` + 阈值扫描 `scan_thresholds` |
| `evaluation/refusal_report.py` | **117**（新） | Markdown 渲染 `render_markdown`（只排版，不重算指标） |

三个文件均 ≤300 行。总行数 314 → 360（+46 是模块 docstring 与 import 成本）。
调用方零改动：没有任何其他代码 import 这三个函数；**脚本路径 `python evaluation/calibrate_refusal.py` 不变**。

### 5.2 等价性证明（`reports/b24b_calibrate_split_equivalence.txt`，退出码 0）

```
[1] AST 逐对象比对（剥 docstring 后逐字相同）
    拆分前顶层对象 4 个：['collect_scores', 'main', 'render_markdown', 'scan_thresholds']
    拆分后顶层对象 4 个：['collect_scores', 'main', 'render_markdown', 'scan_thresholds']
    [PASS] collect_scores / [PASS] main / [PASS] render_markdown / [PASS] scan_thresholds

[2] 运行时对拍（同一 payload，旧实现 vs 新实现）—— 样本 reports/latest_refusal_calibration.json（85 行）
    render_markdown 输出：旧 1871 字符 / 新 1871 字符 → 逐字节相同
    calibrate_refusal.render_markdown 是否仍指向同一函数：True
    [PASS] scan_thresholds('top1_vector', 85 行) → 最优阈值=0.6323 平衡准确率=0.9524
    [PASS] scan_thresholds('top1_rerank', 85 行) → 最优阈值=0.3451 平衡准确率=0.9167
    [PASS] scan_thresholds('字段名不存在', 85 行) → 分数样本不足，无法扫描阈值
    [PASS] scan_thresholds('top1_rerank', 15 行)  → 分数样本不足，无法扫描阈值
    [PASS] scan_thresholds('top1_rerank', 0 行)   → 分数样本不足，无法扫描阈值
    calibrate_refusal.collect_scores 是否仍指向同一函数：True
    总结论：全部通过
```

---

## 六、25-2 ②：`.dev_bak25b/` 清理

先核对"这确实是改动前副本"，再清理：

```
  app/core/config.py                bak=fa9cb480 now=7434b85b  DIFF（符合预期：是改动前副本）
  app/models/mineru.py               bak=8ce0b268 now=637516fc  DIFF
  app/models/qwen_vl.py              bak=b2b98de7 now=f89c6af9  DIFF
  app/retrieval/keyword_search.py    bak=8a29771f now=1744b5f6  DIFF
  app/retrieval/vector_search.py     bak=0507ab40 now=1a3f5abe  DIFF
```

- 已**复制到临时回收位置** `%TEMP%/b24_bak/retired_25b/`（5 个文件），再从项目内移除。
- 项目根当前 **已无 `.dev_bak*` 目录**。
- 说明：没有直接 `rm`，是"先转存再移除"，2 小时内可回捞；确认不需要后我再彻底清掉临时那份。

---

## 七、本批改动清单（共 5 个文件：2 改 + 2 新增 + 1 新模块）

| 文件 | 改动 |
|---|---|
| `backend/app/retrieval/query_rewrite.py` | ⑤ `_compose_query` 改为循环剥前缀（+注释） |
| `backend/app/auth/session_store.py` | ⑥ 新增 `_prune_dangling_tokens` 并在 `create_session_token` 调用 |
| `evaluation/calibrate_refusal.py` | ⑧ 314 → 127 行，改为装配 + CLI + 再导出 |
| `evaluation/refusal_scoring.py` | ⑧ 新增（116 行） |
| `evaluation/refusal_report.py` | ⑧ 新增（117 行） |

改动前备份（md5 已核对，均在 `%TEMP%/b24_bak/`）：
`session_store.py.before_b24b`、`query_rewrite.py.before_b24b`、`calibrate_refusal.py.before_b24b`。

**未改动**：`docs/`（用户已自行改路线图那行）、`data/`、`backend/tests/` 内容、任何 `.env`。

---

## 八、验收三项（真实输出）

```
$ PYTHONPATH= python -m pytest backend/tests -q
598 passed, 3 warnings in 13.91s                      exit=0

$ python scripts/check_services.py
=== 结果 ===  ✅ 全部通过（退出码 0）                   exit=0

$ python scripts/check_imports.py
扫描文件数: 103（backend/app 由 pytest 覆盖，此处跳过）
✅ 全部 app.* 导入均可解析（模块存在 + 符号存在）        exit=0
```

扫描数 101 → **103**，正好是 ⑧ 新增的两个 `evaluation/` 模块。

证据文件：`reports/b24b_final_pytest.log`、`reports/b24b_check_services.txt`、`reports/b24b_check_imports.txt`

---

## 九、遗留 / 待裁决

1. **⑤ 的单字残渣边界**（见 3.4）：`那那那个…` / `前面那个员工…` 会残留单字。要不要再加"那后面必须紧跟前缀词"的约束？本批未做。
2. **`user_sessions:pending` 是否直接删**：已按裁决"加 TTL"处理（不删）。若认为该键（19 个成员全悬空）应直接清掉，说一声。
3. **本批新增的评测报告文件**：`reports/eval_20260921_16*_b24b_*.json/md` 共 6 个 + 3 个 `latest_eval_b24b_*.json/md`，是否保留？
4. **行尾 LF**：本批未触碰任何 `.jsonl`，`eval_set_v1.jsonl` 保持批次 24 归一后的 LF。

---

## 十、下一批（按你的顺序）

- **批次 26 重做取证 + 回填**：按 Q1/③/④ 口径——[05][06][08] 填 `status=effective` + 备注写修订沿革（日期+文号）；[01] 整部 `effective` + 第三十二条第一款废止写入备注；[02] 用新取值 `not_applicable`（并确认前端/接口不因此报错）；其余留空；**禁止规则推定**，产出《人工确认清单》（含 [05][06] 的日期口径不一致问题）。
- **批次 27**：答辩讲解材料，单份文档交付到桌面。
