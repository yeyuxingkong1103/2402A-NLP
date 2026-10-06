# 批次 26（重做取证 + 回填）验收报告
### 附：批次 24 剩余 4 项裁决的执行记录

**执行时间**：2026-09-21
**本批范围**：① 你裁决的 4 个收尾项（⑤ 单字残渣 / Redis pending / 评测报告瘦身 / 临时回收位清理）＋ ② 批次 26 重做取证与回填 ＋ ③《人工确认清单》。
**边界遵守**：未改 `docs/`；临时脚本全部落在 `%TEMP%/b26_bak/`（未进 `backend/`）；未动 `data/`、`.env`。
**不在本批**：批次 27 答辩讲解材料（等你放行后开跑）。

---

## 〇、一句话结论

| 项 | 要求 | 结果 | 状态 |
|---|---|---|---|
| ①-1 ⑤ 单字残渣约束 | 加"剥完剩余不以单字助词开头，否则不剥"＋单测＋A/B | 已加约束；+13 条单测；A/B **15/15 逐条完全一致**（中性） | ✅ |
| ①-2 单测锁"不产生单字残渣开头" | 加 | 2 组参数化用例（7 + 6） | ✅ |
| ② `user_sessions:pending` 19 悬空成员 | 删；删前证明无对应 `session:*`，删后贴 key 数与 TTL 分布 | **19/19 全部悬空** → SREM 19，键被自动回收；DBSIZE 32、TTL=-1 的键 **0** | ✅ |
| ③ 评测报告瘦身 | 保 3 个 `latest_eval_b24b_*`，删 6 个中间 `eval_*` | 6 个中间文件与 latest **逐字节相同**后删除 | ✅ |
| ④ `retired_25b/` | 彻底清 | 已 `rm -rf`，项目根无 `.dev_bak*` | ✅ |
| 批次26-A 接口零破坏 | 管理端列表 + 检索接口各一次真实调用 | 两次 **HTTP 200 / code=0**，`effective`/`not_applicable` 未出现在响应里 | ✅ |
| 批次26-B 回填 | [05][06][08] `effective`+修订沿革；[01] 整部 effective + 条文废止入备注；[02] `not_applicable`；其余留空 | **5 填、6 留空**，逐条 `rowcount=1` | ✅ |
| 批次26-C 人工确认清单 | 每条给 ①法规名 ②最佳证据 ③建议值 ④待确认那句话 | `reports/batch26_manual_confirmation_list.md`（6 条法规 + 7 条跨条目） | ✅ |

**验收三项**：`611 passed` / `check_services` 退出码 0 / `check_imports` 103 文件退出码 0。

---

## 一、① ⑤ 单字残渣约束

### 1.1 改了什么

`backend/app/retrieval/query_rewrite.py` 的 `_compose_query`：循环剥前缀时增加"切碎词就放弃这一刀"的判据。

```python
# 单字碎片首字：这些字不能单独成词（"个""的"），只能粘在别的字后面用（"那个""是的"）。
_FRAGMENT_LEAD_CHARS = ("个", "的", "那", "这")
```

```python
        while True:
            hit = next((p for p in prefixes if remainder.startswith(p)), None)
            if hit is None:
                break
            candidate = remainder[len(hit) :].strip()
            # 这一刀会把词切成单字碎片就放弃（"那个员工" -"那"→ "个员工"）。
            # 例外："那这个呢？" 剥完得到的 "这个呢？" 虽然也以"这"开头，但它本身
            # 就是另一个前缀，能整词剥干净，不该拦 —— 拦了会退回 "主题 那这个"。
            if candidate[:1] in _FRAGMENT_LEAD_CHARS and not candidate.startswith(
                _REFERENCE_PREFIXES
            ):
                break
            remainder = candidate
```

### 1.2 ⚠️ 两件必须报告的事

**(1) 按你的字面规则直接实现，会把 ⑤ 的成果打回去**

你的规则是"剥完后剩余不得以单字助词（个/的/那/这）开头，否则不剥"。字面执行的话：
`那这个呢？` → 剥"那" → 剩余 `这个呢？` 以 **"这"** 开头（"这"在你的清单里）→ **不剥** → 结果退回 `主题 那这个`，而 ⑤ 刚落地成 `主题`。

所以我在判据里加了一个**例外**：**剩余串本身是另一个前缀的开头时不拦**（`这个呢？` 能整词剥干净）。这样两条要求同时成立：`那这个呢？` 仍是 `经济补偿`，而 `那个员工…` 不再被切成 `个员工…`。

**(2) 自检脚本抓出生产代码一个真 bug**

第一版复刻脚本 + 生产代码都写成 `candidate.startswith(prefixes)`，而 `prefixes` 是 **`sorted()` 返回的 list** —— `str.startswith()` **不接受 list**，运行期直接 `TypeError`。是 A/B 驱动器的自检（要求复刻与生产逐字一致）先炸出来才发现的。已改成元组 `_REFERENCE_PREFIXES`，并留了注释说明别再写回 list。

### 1.3 探针对照表（`reports/b24c_fragment_probe.txt`）

三档并列：`single`＝更早的"只剥一层"、`loop`＝改前（本批要修）、`guard`＝改后（＝生产）。

| 输入 | single（更早） | loop（改前） | **guard（改后）** |
|---|---|---|---|
| `那这个呢？` | `经济补偿 这个` | `经济补偿` | **`经济补偿`** ✅ ⑤ 成果保住 |
| `那这个期间怎么算？` | `经济补偿 这个期间怎么算` | `经济补偿 期间怎么算` | **`经济补偿 期间怎么算`** |
| `那那那个怎么算` | `经济补偿 那那个怎么算` | `经济补偿 `**`个`**`怎么算` ⚠️ | **`经济补偿 那个怎么算`** ✅ |
| `前面那个员工的工资怎么算？` | `经济补偿 那个员工的工资怎么算` | `经济补偿 `**`个`**`员工的工资怎么算` ⚠️ | **`经济补偿 那个员工的工资怎么算`** ✅ |
| `那个员工的工资怎么算？` | `经济补偿 `**`个`**`员工的工资怎么算` ⚠️ | `经济补偿 `**`个`**`员工的工资怎么算` ⚠️ | **`经济补偿 那个员工的工资怎么算`** ✅ |
| `那这个最长能约定几年？` | `竞业限制补偿 这个最长能约定几年` | `竞业限制补偿 最长能约定几年` | **`竞业限制补偿 最长能约定几年`** |
| `那些情况怎么算` | `经济补偿 些情况怎么算` | `经济补偿 些情况怎么算` | `经济补偿 些情况怎么算`（**残渣仍在，见下**） |
| `上述情况如何处理？` / `这个怎么算` / `那 12 期呢` | — | 同 guard | 不受本约束影响 ✅ |

`loop vs guard 有差异的探针：3 / 11`（正是要修的三条）。

**⚠️ 残留边界（如实登记）**：`那些情况怎么算` → `经济补偿 些情况怎么算`，**"些"不在你给的集合（个/的/那/这）里**，所以没被拦住。它属同一类"只能粘着用的字"（些/份/种…），**要不要把集合扩到"凡是被切出来的单字粘着语素"**？本批按你给的集合原样实现，未自作主张扩大。

### 1.4 新增单测（`backend/tests/test_query_rewrite.py`）

- `test_compose_query_does_not_split_words_into_fragment` —— 7 组参数化，逐条锁死**具体期望串**（含 ⑤ 的 `那这个呢？→ 经济补偿` 不被误伤）
- `test_rewrite_never_starts_remainder_with_fragment` —— 6 组参数化，锁**不变量**：结果"主题"之后的部分**不得以 `个`/`的` 开头**（测试里独立写了一份 `{"个","的"}`，不复用生产常量，避免同义反复）

单测数：**8 → 21**（+13）。

### 1.5 A/B（10 `followup_rewrite` + 5 `multi_turn`，改前 vs 改后）

驱动器：`%TEMP%/b26_bak/ab_compose_guard.py`（同批新写；进程内只替换 `QueryRewriter._compose_query`，其余全走生产实现）。
自检：`STRIP_MODE=guard` 与生产实现 **11 条探针逐字一致**才放行。

| 组合 | Recall@5 | **MRR@10** | 引用正确率 | 拒答准确率 | multi_turn 改写命中率 | 第2轮 top5 |
|---|---|---|---|---|---|---|
| loop（改前） | 1.0000 | **0.6444** | 1.0000 | 0.0000 | 1.0（分母 4） | 1.0（分母 5） |
| **guard（改后）** | 1.0000 | **0.6444** | 1.0000 | 0.0000 | 1.0（分母 4） | 1.0（分母 5） |

**逐条对照**（`reports/b24c_guard_ab_compare.txt`）：**检索侧完全一致 15 / 15**；`golden_rank`/`hit_at_5` 差异 **0**；`retrieval_stats` 差异 **0**；multi_turn 改写串差异 **0**。
→ **完全中性**，按你的裁决"中性也保留"照常落地（消除脏输出本身有价值）。
原始报告：`reports/latest_eval_b24c_strip_loop.json` / `latest_eval_b24c_strip_guard.json`（成对保留为基线快照）。

---

## 二、② 删 `user_sessions:pending` 的 19 个悬空成员

### 2.1 删前证据（`reports/b24c_redis_pending_delete.txt`）

逐个成员核对 `session:<token>` 是否存在 ——

```
【删前证据】user_sessions:pending
  成员数：19   键 TTL：5557
      逐个成员核对 session:<token> 是否存在：
      -8UO8urNOdFjjopj0HlL   悬空（无 session 键）
      …（19 行全部为"悬空（无 session 键）"）…
  小结：悬空 19 / 有对应会话 0 / 共 19
```

**19 个成员 100% 悬空，一个有效会话都没有** —— 符合你给的删除条件。

### 2.2 执行与删后复核

```
【执行删除】仅 srem 这 19 个悬空成员（不触碰其他任何键）
  SREM 返回（实际移除数）：19

【删后复核】
  user_sessions:pending 是否仍存在：False
      集合已空 → Redis 自动回收该键（无残留空集合）
  全库 key 总数（DBSIZE）：32
  TTL 分布：4~6h   32 个
  TTL=-1（永不过期）的键：无
  按前缀归类：session:*  28 个 ／ user_sessions:*  4 个

【其他 user_sessions:* 未被改动复核】
  user_sessions:cef5c9236e4456b1f51586e45f21a36f   成员=23   悬空=0   TTL=19586
  user_sessions:f8d8b20fd973c6564176e9aeb7412a17   成员=3    悬空=0   TTL=19610
  user_sessions:probe-admin-0001                   成员=1    悬空=0   TTL=15533
  user_sessions:probe-user-0001                    成员=1    悬空=0   TTL=15533
```

**要点**：① 其余 4 个 `user_sessions:*` 一个字符没动，悬空数全为 0；② 全库 **TTL=-1 的键从 1 → 0**；
③ 空集合被 Redis 自动回收，没有留下"空壳键"；④ 本批只发了 1 条 `SREM`，没有 `DEL` / `FLUSH`。

---

## 三、③④ 文件清理

### ③ 评测报告瘦身

删前先证明"删了不丢信息" —— 6 个中间文件与对应 `latest_*` **md5 逐一对上**：

```
  同  eval_20260921_161933_b24b_on_single.json == latest_eval_b24b_on_single.json  (f8de76a29345d4580b31eff2ba47eecc)
  同  eval_20260921_161933_b24b_on_single.md   == latest_eval_b24b_on_single.md    (b65dad605baa954a28a6eac7a5a0fc78)
  同  eval_20260921_162200_b24b_off_single.json == latest_eval_b24b_off_single.json (26961a7b98796aee4532ce86924ec75e)
  同  eval_20260921_162200_b24b_off_single.md   == latest_eval_b24b_off_single.md   (cd5ae0c39a4787db4fbec02e8471cd06)
  同  eval_20260921_162436_b24b_off_loop.json   == latest_eval_b24b_off_loop.json   (882eb2f168024b3c66dbfa48c9f18fd2)
  同  eval_20260921_162436_b24b_off_loop.md     == latest_eval_b24b_off_loop.md     (c5ff4afa546179bcdb64b4731a8d0912)
```

已删 6 个，保留 3 组 `latest_eval_b24b_*`（`on_single` / `off_single` / `off_loop`，每组 `.json` + `.md`，共 6 个文件）。

> **口径说明**：你说的"保留 **3 个** `latest_eval_b24b_*`"我按 **3 组**执行（每组含 `.json` + `.md`）—— 只留 `.md` 或只留 `.json` 会缺一半证据。若你要的是字面 3 个文件，说一声我删掉其中一半。

`b24b_*` 的证据文件（`b24b_ab_modes.log`、`b24b_calibrate_split_equivalence.txt`、`b24b_redis_pending_*.txt`、验收日志等 8 个）**全部保留**，未动。

本批新产出的 A/B 也按同一规则瘦身：保留 `latest_eval_b24c_strip_loop/guard`（2 组），删掉 4 个时间戳副本。

### ④ 临时回收位

```
删除前：%TEMP%/b24_bak/retired_25b/  存在（含 app/ 子树）
删除后：No such file or directory   ← 已彻底清
项目根复查：No such file or directory (.dev_bak*)   ← 干净
```

---

## 四、批次 26-A · 接口零破坏验证（真实调用）

### 4.1 静态面：`status` 有无硬校验

| 位置 | 读 `law_versions.status` 吗 | 有枚举校验吗 | 结论 |
|---|---|---|---|
| `GET /api/v1/admin/documents`（管理端列表） | ❌ 不带该字段 | 它的 `status` 入参校验的是 **`document_versions.version_status`**（`pending_review`/`approved`/`rejected`，单一来源 `app/db/version_status.py`），与 `law_versions.status` **完全无关** | 不受影响 |
| `GET /api/v1/admin/documents/{id}/detail` | ❌ 只取 `version_status` | — | 不受影响 |
| `POST /api/v1/legal/search`（检索接口） | 间接：`resolve_current_status(row.status, …)` | ❌ 无校验；只与**中文** `"已废止"/"已失效"` 比较 | 不报错；见 4.3 |
| `law_models.py:97` | `String(32) nullable` | ❌ 无 CHECK / 无 Enum | 新取值可存 |
| `legal_metadata_writer.py:76` | `status=metadata.status  # 留空表示待补录` | — | 设计上就允许留空 |

### 4.2 动态面：真实调用（起真实 uvicorn + 真 Redis 会话 + 真 MySQL/Milvus）

脚本 `%TEMP%/b26_bak/b26_api_probe.py`：起真实 `app.main:app`（端口 **8123**，避开被 Docker 占用的 8000）→ 用生产 `SessionStore` 往**真实 Redis** 写管理员令牌 → 真调两个接口 → 用完撤销令牌 → 关服务。

**回填前**（`reports/b26b_api_probe_before.txt`）：

```
[4] GET /api/v1/admin/documents?status=approved&page=1&page_size=3
    HTTP 200   code=0   message=success
    data 键：['items','page','page_size','total']   条目数：3 / total=11
[5] POST /api/v1/legal/search  {"query":"经济补偿怎么算","top_k":5}
    HTTP 200   code=0   message=success
    命中条数：5
    stats：{"vector_recall_count":12,"keyword_recall_count":20,"fused_count":28,"reranked_count":5,...}
      - 中华人民共和国劳动合同法 / 第四十七条 … is_current=None
```

**回填后**（`reports/b26b_api_probe_after.txt`）：

```
[4] HTTP 200   code=0   message=success        条目数：3 / total=11
[5] HTTP 200   code=0   message=success        命中条数：5
      - 中华人民共和国劳动合同法 / 第四十七条 … is_current=True    ← 变化
      - 中华人民共和国劳动合同法实施条例 / 第二十七条 … is_current=None
[6] 响应里是否出现 status 的取值本身（按 JSON 字符串值精确匹配）
    "not_applicable"  管理端：False｜检索：False
    "effective"       管理端：False｜检索：False
    "amended"         管理端：False｜检索：False
    "repealed"        管理端：False｜检索：False
    "superseded"      管理端：False｜检索：False
[8] 结论：两个接口 HTTP 状态均为 200 / 200，无 5xx、无异常信封
```

**→ 新取值 `not_applicable` 与 `effective` 都不会出现在任何接口响应里**（`status` 根本没被序列化；前端能拿到的只有 `is_current` 布尔值）。

### 4.3 唯一可观察到的变化：`is_current`

用**生产函数** `app.retrieval.context_builder.resolve_current_status`（不打桩）对 11 部各算两遍（`reports/b26b_is_current_impact.txt`）：

| # | 法规名 | 库内 status | 回填前 is_current | 回填后 is_current |
|---|---|---|---|---|
| 01 | 解释（一） | `effective` | None | **True** |
| 02 | 典型案例 | `not_applicable` | None | **True** |
| 05 | 劳动合同法 | `effective` | None | **True** |
| 06 | 劳动法 | `effective` | None | **True** |
| 08 | 工伤保险条例 | `effective` | None | **True** |
| 03/04/07/09/10/11 | （留空 6 部） | `NULL` | None | None（**不变**） |

- **影响面只有本批写入的 5 部**，其余 6 部两遍完全相同。
- **方向一律 `None`（未知）→ `True`（现行）**——把"不知道"变成"知道"。
- **没有任何一部从 True/None 变成 False** → 不存在"条文被静默隐藏"的风险（检索侧只在 `is_current is False` 时过滤，见 `app/retrieval/keyword_search.py:268`）。

> ⚠️ **一个值得你判断的语义细节**：[02] 是"案例材料"，`not_applicable` 也会算出 `is_current=True`，字面读作"现行有效"。功能上无害（它就保持可检索，与回填前的 `None` 一样），但语义上不太贴。若你希望 `not_applicable` 输出 `is_current=None`，需要在 `resolve_current_status` 里加一条判断 —— **属代码改动，本批未动**。

**还有一个反直觉但重要的发现**：`status` 的词表其实是**中文**的 —— `resolve_current_status`（`context_builder.py:193`）与索引侧（`vector_index_service.py:234`）只认 `"已废止"/"已失效"`。本批写入的英文 `effective` 落进"不属于已废止"分支，**行为上等同于"未标注废止"，与写入前一致**。这正是接口零报错的根本原因，也意味着**英文取值目前对检索行为不起作用**（只作台账用）。已登记为清单 C4。

---

## 五、批次 26-B · 回填（只写 5 行）

### 5.1 写入前 → 写入后（`reports/b26b_backfill_before_after.txt`）

```
【写入前】非空 status：0 / 11      非空 revision_note：0 / 11
  5 条 UPDATE 全部 rowcount=1，已提交。
【写入后】非空 status：5 / 11      非空 revision_note：5 / 11

【逐条核对】写入前 → 写入后
  最高人民法院关于审理劳动争议案件适用法律问题的解释（一）   status: None → 'effective'
  最高法发布劳动争议典型案例                        status: None → 'not_applicable'
  中华人民共和国劳动合同法                          status: None → 'effective'
  中华人民共和国劳动法                            status: None → 'effective'
  工伤保险条例                                 status: None → 'effective'

  其余保持 NULL 挂账：['最高法发布劳动争议司法解释（二）和典型案例', '中华人民共和国劳动争议调解仲裁法',
                     '工资支付暂行规定', '中华人民共和国劳动合同法实施条例',
                     '女职工劳动保护特别规定', '职工带薪年休假条例']
```

### 5.2 逐条取值与证据（详见 `reports/batch26_manual_confirmation_list.md`）

| # | status | 备注主线（原文级） |
|---|---|---|
| 01 | `effective` | 法释〔2020〕26号、2021-01-01 施行；第三十二条第一款被解释（二）（法释〔2025〕12号）废止 |
| 02 | `not_applicable` | 案例材料（`document_type=案例材料`/接口枚举 `case`），无施行条款、无效力状态概念 |
| 05 | `effective` | 根据 2012-12-28《关于修改〈中华人民共和国劳动合同法〉的决定》修正；页面"文号"栏＝**无** |
| 06 | `effective` | 第一次修正 2009-08-27、第二次修正 2018-12-29；页面"文号"＝**无**、"公布日期"＝**空** |
| 08 | `effective` | 根据 2010-12-20《国务院关于修改〈工伤保险条例〉的决定》修订；修订决定文号页面未标注 |

### 5.3 ⚠️ 两点必须报告

**(1) "其余 7 部" 实际是 6 部。** 11 − (已填 5) = 6，留空的是 `[03][04][07][09][10][11]`。按你逐条点名的口径执行无误；差异应源于把 [02]（`not_applicable`，既非留空也非 effective）计入了"留空"。

**(2) 重新导入会把本批手工填的 `status` 冲掉。** `app/db/legal_metadata_writer.py:85-93` 在版本已存在时会用本次抽取结果**刷新** `promulgation_date / effective_date / expiration_date / status`；抽取器对这几部页面抽不到状态词 → `None` → **`status` 被写回 NULL**。（`revision_note` 不在刷新列表，不会被覆盖。）这是本批新发现的风险，已登记为清单 C5，**未改代码**。

### 5.4 写入范围与撤销

- 只写 `law_versions` 的 `status` + `revision_note` 各 5 处，逐条断言 `rowcount == 1` 才 `commit`。
- **未动**：`laws`、`articles`、`document_chunks`、`document_versions`，以及 Milvus / Redis。
- **可完整撤销**：`python %TEMP%/b26_bak/b26_backfill_status.py --rollback`（把 5 条复位为 NULL）。

---

## 六、批次 26-C ·《人工确认清单》

`reports/batch26_manual_confirmation_list.md`（四节）：

- **第一节**：本批已填 5 条的逐条原文证据
- **第二节**：留空 6 部，每条四栏 —— **①法规名 ②最佳证据（URL + 逐字原文片段）③建议值 ④待确认的那句话**
  - R1 [03] 司法解释（二）：证据最强的留空项（有施行条款 + "此前不一致者以本解释为准"）→ 问"有施行条款算不算规则推定"
  - R2 [04] 调解仲裁法 / R3 [07] 工资支付暂行规定 / R4 [09] 实施条例 / R5 [10] 女职工特别规定 / R6 [11] 年休假条例
- **第三节**：7 条跨条目问题 —— **C1 [05][06] 日期口径不一致**（你点名的）、**C2 [08] 同名双条目 LawID 950 vs 610**（你点名的）、C3 修订决定文号缺失、C4 词表中英两套、C5 重导入覆盖手工值、C6 条文级废止标记、C7 `not_applicable` 是否固化进枚举

**C1 的实质影响（附证据）**：`[05]` 的 `promulgation_date=2012-12-28` 与 `effective_date=2008-01-01` 取自**不同版本**（公布晚于生效近 5 年；`[06]` 差近 24 年）。而 `app/retrieval/filters.py:91-98` 用 Milvus 里的 `effective_date <= as_of_date` 做时效过滤 → **按时间点检索会出错**：`as_of_date=2010-06-01` 时，[05] 返回的是 2012 年才修正的文本（真实检索响应里《劳动合同法》第四十七条返回 `effective_date=2008-01-01`，见 `b26b_api_probe_after.txt`）。修法给了 (A)/(B) 两个选项，推荐 (A)。

---

## 七、验收三项（真实输出）

```
$ PYTHONPATH= python -m pytest backend/tests -q
611 passed, 3 warnings in 13.46s                                  exit=0

$ python scripts/check_services.py
=== 结果 === ✅ 全部通过（退出码 0）                                 exit=0
（documents=11  document_chunks=1627  law_versions=11  document_versions=11
  Milvus=1627 == MySQL.document_chunks=1627   approved=11）

$ python scripts/check_imports.py
扫描文件数: 103（backend/app 由 pytest 覆盖，此处跳过）
✅ 全部 app.* 导入均可解析（模块存在 + 符号存在）                     exit=0
```

**611 = 598 + 13**（本批新增的 13 条单测）。证据：`reports/b26b_check_pytest.txt` / `b26b_check_services.txt` / `b26b_check_imports.txt`。

---

## 八、本批改动清单

| 文件 | 改动 |
|---|---|
| `backend/app/retrieval/query_rewrite.py` | 新增 `_FRAGMENT_LEAD_CHARS` 常量；`_compose_query` 加"切碎词就不剥"的判据（+ 注释） |
| `backend/tests/test_query_rewrite.py` | 新增 2 组参数化单测（+13 例）、`import pytest`、2 个测试辅助 |
| **MySQL `law_versions`** | 5 行：`status` + `revision_note`（[01][02][05][06][08]） |

**删除**：`reports/eval_*_b24b_*.json/md`（6）、`reports/eval_*_b24c_*.json/md`（4，本批自产）、`%TEMP%/b24_bak/retired_25b/`（目录）、`user_sessions:pending` 的 19 个悬空成员。
**未改动**：`docs/`、`data/`、`.env`、`.git`(无)、`laws`/`articles`/`document_*` 表、Milvus、其余 Redis 键。
**备份**：`%TEMP%/b24_bak/query_rewrite.py.before_b24b2`（md5 `dcf3c44c0df4eedc2f57d126fa5a117c`，改动前）。

---

## 九、遗留 / 待你拍板

1. **`那些情况怎么算` 的 `些`** —— 不在你给的集合里，仍残留。要不要把 `_FRAGMENT_LEAD_CHARS` 扩成"凡被切出来的单字粘着语素"？
2. **[02] 的备注** —— 我多写了一条（你只要求 [05][06][08]+[01]）。要清掉说一声。
3. **"保留 3 个 latest"** —— 我按 3 组（6 文件）执行；若要字面 3 个文件，我删一半。
4. **`not_applicable` 的 `is_current`** —— 现算作 `True`（字面"现行有效"）。要不要让它输出 `None`？
5. **清单 C4（词表中英两套）/ C5（重导入覆盖手工值）** —— 都属代码改动，等你批准再动。
6. **清单 C1（[05][06] 日期口径）** —— 需要你选 (A) 还是 (B)；选了之后还要重跑向量索引才生效。

---

## 十、下一批

**批次 27：答辩讲解材料** —— 单份文档交付到桌面，六节结构（代码职责总览表 / 一条问答请求的旅程 / 一条数据的旅程 / 原理 7 节 / 5 分钟答辩台词本 / 20 条预测提问与答法），面向非技术背景的项目负责人，能照着讲、不用再查代码。
