# 批次 41：行为中性清理（12 处未使用 import + 3 个真死常量 + 承诺性注释修正 + 窗口常量绑定）

生成时间：2026-09-23 · 改动 17 个文件（+1 新增测试）· 全量测试 **744 passed**（741 基线 + 3 新增）

## 0. 一句话结论

按"是否改变行为"分类执行完毕：**15 个零引用符号已删、1 条承诺性注释已改为事实、1 处窗口常量已建立"单一来源 + 测试强制"绑定**；
行为中性由**逐函数 AST 对拍**证明（17 个文件里 13 个 AST 完全一致、3 个只差一个死常量赋值、1 个只差一个同值改名节点），
全量测试不减反增，评测基线锚点（`prompt_builder.py` md5）与基线归档值一致，**已归档基线仍然可比**。
另：本轮我自己的编辑脚本出过一次覆盖 bug（已回滚重做）与两次"预期设定错误"（已修正口径），都在 §7 如实记录。

---

## 1. 改动清单（精确到符号）

### 1.A 删 12 处未使用 import（用户清单第 1 项）

| # | 文件 | 删除的名字 | 形态 |
|---|---|---|---|
| 1 | `backend/app/chat/guard.py` | `CitationError` | 多行 import 块中的 1 行 |
| 2 | `backend/app/chat/chat_history.py` | `SessionAccessDenied` | `from … import A, B` 中的 1 个 |
| 3 | `backend/app/db/vector_index_service.py` | `Article` | 同上 |
| 4 | `backend/app/retrieval/assembly.py` | `DEFAULT_TABLE_PATH` | **函数内** import 块的 1 行 |
| 5 | `backend/app/memory/long_term.py` | `LongTermMemoryError` | 多行 import 块中的 1 行 |
| 6 | `backend/app/memory/memory_schema.py` | `from typing import Any` | 整行 |
| 7 | `backend/app/models/mineru_api.py` | `BytesTransport`、`Transport` | 多行 import 块中的 2 行 |
| 8 | `evaluation/compare_window.py` | `import os` | 整行 |
| 9 | `evaluation/faithfulness.py` | `import os` | 整行 |
| 10 | `scripts/e2e/fetch_pdf_samples.py` | `import sys` | 整行 |
| 11 | `scripts/migrations/backfill_law_dates.py` | `text` | `from sqlalchemy import a, b, c` 中的 1 个 |

共 **11 个文件 / 12 个名字**，与你"12 处"的计数一致。
`backfill_law_dates.py` 的 `Session` **未删**：它只出现在 `# type: Session` 注解里，属你排除的"TYPE_CHECKING 注解"那一类（删了会让该注解悬空）。

### 1.B 删 3 个真死常量（第 2 项）

| # | 位置 | 常量 | 连带修正 |
|---|---|---|---|
| 12 | `backend/app/ingest/cleaner.py` | `_SOURCE_PREFIX = "来源："` | 同文件 `_BOILERPLATE_PREFIXES` 已有同一字面量，纯重复定义 |
| 13 | `backend/app/ingest/law_metadata_extractor.py` | `CASE_MATERIAL_KEYWORDS = (...)` | 生效的是下一行 `CASE_MATERIAL_TITLE_KEYWORDS` |
| 14 | `backend/app/cli/create_admin.py` | `STATUS_NOT_REGISTERED = "not_registered"` | 注释「三种结果标记」→「两种」；docstring 的「返回」段删掉该条（该方法实际是 **抛 ValueError**，不返回这个标记） |

### 1.C 注释 / 常量类（第 3、4、5 项）

| # | 文件 | 改动 |
|---|---|---|
| 15 | `backend/app/models/mineru.py` | `MAX_PAGES` 注释由「这里提前拦下」改为事实：**当前未实现**页数拦截 + 指向本报告待办段（§5）。常量保留，**未补实现** |
| 16 | `backend/app/api/chat_persistence.py` | 新增模块常量 `SHORT_TERM_MAX_MESSAGES = 20`，写回侧改用 `max_messages=SHORT_TERM_MAX_MESSAGES`（原为字面量 20） |
| 17 | `evaluation/multi_turn_grading.py` | `SESSION_WINDOW = 20` **保留**；注释改为三点绑定说明 + 为何不用 import（见 §4） |
| 18 | `backend/tests/test_short_term_window_binding.py` **（新增）** | 3 个用例，把三处窗口钉死（见 §3） |

前端那行注释（第 5 项）**批次 40 已完成，本轮未再动**：`frontend/src/app/(auth)/password-reset/page.tsx:89`
= `// 仅 ENVIRONMENT=development 时后端会回显 debug_code（与 SMTP 是否接入无关），用于本地联调`，
与备份（旧文案）逐行 diff 仅此一行、行数 261 不变；当前 md5 `c1e5f76381a07f37a9f7fb9a2deb019a`。

---

## 2. 行为中性的证明（五重证据）

### 2.1 证据一：逐函数 AST 对拍（最强）

用 `%TEMP%/b41_bak/orig/` 的**改前备份**与改后文件逐函数反编译对比（剥掉 import 与 docstring，注释本就不在 AST 里）：

| 结果 | 文件数 | 含义 |
|---|---|---|
| `IDENTICAL` | **13** | AST 完全一致（改动只落在 import 名表 / 注释 / docstring） |
| `IMPORT/COMMENT ONLY` | **3** | 除 import 外，只各差**一个死常量赋值**的删除：`cleaner.py`（`_SOURCE_PREFIX`）、`law_metadata_extractor.py`（`CASE_MATERIAL_KEYWORDS`）、`create_admin.py`（`STATUS_NOT_REGISTERED`） |
| `REVIEW` | **1** | `chat_persistence.get_short_term_memory`：全函数 AST 只差一个节点，见下 |

唯一"函数体有差异"的那一处，AST 差异**只有一个节点**：

```
-        value=Constant(value=20))])),
+        value=Name(id='SHORT_TERM_MAX_MESSAGES', ctx=Load()))])),
```

且运行期取值核对：`SHORT_TERM_MAX_MESSAGES == 20 → True`。即「同一个值换了个名字」，不是改行为。

### 2.2 证据二：零引用扫描器（带对照自证）

按你的要求给扫描器加了**阳性/阴性 + 四类易误报对照**，先在样例上自证，再对真实文件下结论：

```
---- 对照样例（阳性 4 = 档A×3 + 档B×1 / 阴性 2 / 易误报 4）----
  期望判死  : ['LOCAL_UNUSED', 'MENTIONED_ONLY', '_os', 'json']
  实测判死  : ['LOCAL_UNUSED', 'MENTIONED_ONLY', '_os', 'json']   PASS
  档位      : {'json': 'A', '_os': 'A', 'LOCAL_UNUSED': 'A', 'MENTIONED_ONLY': 'B'}   PASS
  被救活    : ['EXPORTED <- 其他文件 from ctl_a import 消费（再导出）',
               'STRPATH  <- 被字符串形式的模块路径引用（如 monkeypatch）']
  误报检查  : PASS
```

自证过程本身抓出 **4 类误报**（若不做对照，"0 发现"无法区分"真干净"与"扫描器坏了"）：
① `from __future__ import annotations` 被当普通 import；② 公共名（本文件零引用但别的文件在用）被当死代码；
③ 只出现在**另一个常量定义行**里的名字（`DEFAULT_OUT_DIR = PROJECT_ROOT / …`）被当死代码；
④ "全仓文本命中即视为在用"过粗 —— 别人定义的同名类会把本文件这个未使用 import **救活（漏报）**，改成只认「从本模块 import 该名」。
另修正一处自引用漏报：常量赋值的**左值本身**是 `ast.Name`，不跳过会把自己算成一次引用。

扫描结果（17 个目标文件）：

| 时点 | 判死合计 | 明细 |
|---|---|---|
| 改前 | **18** | 档A（全文零提及）13 + 档B（仅注释/文档提及）5 |
| 改后 | **3** | 全部是**有意保留**：`SESSION_WINDOW`（档A，按你要求保留）、`MAX_PAGES`（按你要求保留）、`Session`（`# type:` 注解用） |

被"救活"（正确不判死）的还有 `REFUSAL_ANSWER`、`DEFAULT_COLLECTION_NAME`、`EVAL_USER_ID` —— 都是对外导出名。

### 2.3 证据三：残留引用核验（0 残留）

逐符号给出三列证据（被改文件内「文本提及 / 代码引用」+「是否有人 `from 本模块 import 该名`」）：

```
被改文件                                     被删名字                   文本提及  代码引用  再导出消费
backend/app/chat/guard.py                    CitationError                 1        0          0  OK  (仅文字提及)
backend/app/chat/chat_history.py             SessionAccessDenied           3        0          0  OK  (仅文字提及)
backend/app/db/vector_index_service.py       Article                       0        0          0  OK
backend/app/retrieval/assembly.py            DEFAULT_TABLE_PATH            0        0          0  OK
backend/app/memory/long_term.py              LongTermMemoryError           0        0          0  OK
backend/app/memory/memory_schema.py          Any                           0        0          0  OK
backend/app/models/mineru_api.py             BytesTransport                0        0          0  OK
backend/app/models/mineru_api.py             Transport                     0        0          0  OK
evaluation/compare_window.py                 os                            0        0          0  OK
evaluation/faithfulness.py                   os                            0        0          0  OK
scripts/e2e/fetch_pdf_samples.py             sys                           0        0          0  OK
scripts/migrations/backfill_law_dates.py     text                          0        0          0  OK
backend/app/ingest/cleaner.py                _SOURCE_PREFIX                0        0          0  OK
backend/app/ingest/law_metadata_extractor.py CASE_MATERIAL_KEYWORDS        0        0          0  OK
backend/app/cli/create_admin.py              STATUS_NOT_REGISTERED         0        0          0  OK
残留问题数：0
```

两个"文本提及=1/3"的是 docstring 里的文字（guard 的"其余 CitationError（…）"、chat_history 的异常说明），
代码引用 0、再导出消费 0 —— 文档措辞依然成立，不需要改。

另：18 个文件（17 改 + 1 新增）`py_compile` **18/18 通过**。

### 2.4 证据四：全量测试

```
744 passed, 3 warnings in 16.49s
```

741（批次 39 基线）+ 3（新增绑定测试）= 744，**不减**。

### 2.5 证据五：导入链未被破坏

- `multi_turn_grading` / `aggregate` / `render_report` 三者在**只把 `evaluation/` 放进 sys.path**（不设 backend）的情况下均可独立导入，
  且 `sys.modules` 中**没有** `app` —— 证明没有把 FastAPI/SQLAlchemy 拖进评测纯函数模块（这正是 §4 的选型依据）。
- `python evaluation/run_eval.py --help` 正常。
- `from app.api.chat_persistence import SHORT_TERM_MAX_MESSAGES` → `20`；`mineru.MAX_PAGES` → `200`、`MAX_FILE_BYTES` → `209715200`。

---

## 3. 新增绑定测试（`backend/tests/test_short_term_window_binding.py`）

锁定 **三处窗口取值一致**（此前只有注释声称一致，无任何绑定）：

| 用例 | 检查对象 | 强度 |
|---|---|---|
| `test_eval_window_equals_production_constant` | 评测侧 `SESSION_WINDOW` == `SHORT_TERM_MAX_MESSAGES` | 值一致 |
| `test_writeback_store_actually_uses_the_constant` | **真实装配出的 store** 的 `.max_messages` == 常量（不是看注释） | 值一致 |
| `test_retrieval_assembly_window_matches_the_constant` | `assembly.build_default_retrieval_service` 里 `max_messages=` 的 AST 实参（字面量或引用常量都接受） | 值一致 |

**证伪（退回旧实现看是否变红）** —— 4 例，全部 PASS 且改动均已还原（md5 前后一致）：

| 人为改坏 | 结果 |
|---|---|
| 检索侧 `max_messages` 20→40 | 1 failed，点名 `test_retrieval_assembly_window_matches_the_constant` |
| 写回侧常量取值 20→30 | 2 failed（两个"值一致性"用例） |
| 评测侧 `SESSION_WINDOW` 20→30 | 1 failed，点名 `test_eval_window_equals_production_constant` |
| 写回侧调用点 20→**40** | 1 failed，点名 `test_writeback_store_actually_uses_the_constant` |

顺手纠正一个我自己的误解：把调用点换成**同值**字面量（20）**不会**变红 —— 因为此时并不存在漂移。
该用例锁的是"**取值**一致"而非"写法必须用常量名"，这正是要防的东西。

---

## 4. `SESSION_WINDOW` 为什么没做成 import 绑定（**你的技术前提有误，已取证**）

你的原话是「改为从 chat_persistence 导入同一个常量」。取证后有三点必须报告：

1. **`chat_persistence.py` 里原本没有常量**，是字面量 `max_messages=20`（`L61`）；全库也不存在任何 `MAX_MESSAGES`/`SESSION_WINDOW` 定义 —— 没有东西可导入。→ 已补上常量 `SHORT_TERM_MAX_MESSAGES`。
2. **顶层 import 会直接炸**：`evaluation/run_eval.py` 在**模块级**（`L52-75`）导入 `aggregate` → `multi_turn_grading`，
   而 `backend` 目录是 `prepare_env()` 在**运行时**（`L101-102`）才插进 `sys.path`。
   即 `python -m evaluation.run_eval` 走到 `from multi_turn_grading import …` 时，`app` 还不可导入 → `ModuleNotFoundError`。
3. **代价不对称**：`multi_turn_grading` 是被 `aggregate` / `render_report` / `run_eval` 共同导入的**库模块**，
   给它加 `sys.path` 副作用 + API 层依赖，会让"纯函数、可直接单测"的 `aggregate`（指标汇总）也背上 FastAPI/SQLAlchemy；
   而且它还会引入一个**新的未使用名**（`SESSION_WINDOW` 本就零引用），刚删完 12 个未使用 import 又加回一个。

因此改为**测试强制绑定**（§3）：覆盖范围更大（连 `assembly` 的字面量一起钉住）、零架构代价、且能被证伪。
若你仍要 import 写法，改动量为 `multi_turn_grading.py` 顶部 6 行（sys.path 前置 + import + `# noqa`），说一声我就改。

仍未真正做到"单一来源"的是 `assembly.py:169` 自己的字面量 —— 它是**检索路径**，本轮按冻结令未动，
已由 §3 的测试兜住（改它就会红）。彻底统一（assembly 也引用同一常量）登记进部署后待办。

---

## 5. 待办登记（部署验收后处理）

### 5.1 `MAX_PAGES` 页数拦截实现（已建任务 #212）

- **为什么**：原注释承诺"超出会被服务端拒绝，这里提前拦下"，但只实现了文件大小拦截（`_build_payload()` 的 `MAX_FILE_BYTES`），
  页数拦截从未实现。后果：超 200 页的 PDF 白走一次 Mineru 服务端调用才被拒（浪费请求/配额与一次往返）。
  本轮按要求**不删常量、不补实现**，只把注释改成事实。
- **怎么做**：在 `MineruClient.parse()` 上传前预检，复用现成的 `app/models/mineru_result.py: resolve_page_count`；
  超限时按 `MAX_FILE_BYTES` 同款方式收敛（不新增异常类型）；`MAX_PAGES` 保持模块级常量以利 monkeypatch。
- **怎么验收**：(a) monkeypatch `MAX_PAGES=1` + 2 页 PDF → 断言**零 HTTP 请求**且错误信息含"页数"；
  (b) 负向：放大阈值后同一 PDF 走通主路径；(c) 全量测试不减；(d) 报告里贴出「拦截发生在 `_build_payload()` 之后、`request_json` 之前」的调用顺序证据。

### 5.2 `render_markdown` 两份是否等价（结论：**不等价，不建议合并**）

| 对比项 | `evaluation/refusal_report.py:18` | `evaluation/render_report.py:27` |
|---|---|---|
| 入参 payload 顶层键 | `rows` / `run_at` / `scans`（+`chosen_threshold` / `comparison` 用 `.get`） | `summary` / `run_at` / `eval_set` / `worst_samples` / `calibration` |
| 产出标题 | **拒答阈值校准报告** | **检索与回答质量评测报告** |
| 章节 | 一现状 / 二分数分布 / 三阈值分界点（含边界样本表）/ 四校准前后对比 | 一四指标 / 二分类型 / 三最差 5 条 / 四拒答阈值校准（JSON 块） |
| 调用方 | `calibrate_refusal.py`（3 处） | `run_eval.py`（2 处） |
| 依赖 | 仅 `statistics` | `json` + `run_config` + `multi_turn_grading` |

两者只是**同名**，输入输出完全不同的两份报表，不存在重复实现，**合并只会引入多余的 dispatch 参数**。
建议各处加一行模块注释注明"同名不同报表"（纯注释，按你"本轮只出结论不动代码"未动；一句话我就补上）。

---

## 6. 边界遵守情况

- **未改 `docs/`** ✅
- 除你列出的项外未动业务代码 ✅（17 个文件的改动逐条列在 §1，AST 对拍可复核）
- 易漂移配置：**本轮未改任何配置**（`.env` 族 mtime 均为本批次之前），故无需备份；改动的源码统一备份在 `%TEMP%/b41_bak/orig/`（17 份，含改前 md5，见附录）
- 改源码统一 `read_bytes()/write_bytes()`，改后逐文件核对：**17/17 纯 LF、无 BOM** ✅
- 临时脚本全部在 `%TEMP%/b41_bak/`，**未进 `backend/`** ✅

---

## 7. 本轮我自己的失误（如实记录）

1. **编辑脚本覆盖 bug（已回滚重做）**：首版按"每条编辑各自读磁盘原文再写盘"实现，
   同一文件有 2 条编辑时**后写覆盖先写** —— 结果是 `chat_persistence.py` 落了调用点却**丢了常量定义**（运行期会 `NameError`）、
   `create_admin.py` 落了 docstring 却**没删掉常量**。
   靠"改后立即复扫"抓到（扫描器报 `STATUS_NOT_REGISTERED` 仍存在），**已从备份整批回滚（17 份 md5 逐一核对还原）后重做**，并改成"按文件累积"。
   教训：多编辑脚本必须按文件累积、并在写盘后立刻跑独立校验；「命中数 == 1」只能保证单条正确，保不了同文件多条。
2. **两次"预期设定错误"**：证伪脚本里我把"改坏写回侧常量"的期望失败用例设成了"是否用到常量"那条（实际该由两条"值一致性"用例抓）；
   又把"调用点换回**同值**字面量"当成该变红的场景（同值时不存在漂移，不该红）。两次都是**我的预期错，不是被测对象错**，已修正口径。
   这正说明"证伪脚本本身也要能被怀疑"。

---

## 8. 遗留

| # | 项 | 状态 |
|---|---|---|
| 1 | `assembly.py:169` 仍是自有字面量（真正的"单一来源"未完成） | 检索路径，冻结期不动；已由 §3 测试兜住；并入 5.1 同期处理 |
| 2 | `backfill_law_dates.py` 的 `Session` | **有意保留**：`# type: Session` 注解需要它在作用域内（属你排除的注解类）。若要彻底干净可改 `if TYPE_CHECKING:`，属额外改动，未动 |
| 3 | `answer_judging.py` 的 `REFUSAL_PHRASES` | 注释写明"旧词表仅留参考/debug" → 有意保留，你未列入，未动 |
| 4 | 两份 `render_markdown` 加注释说明差异 | 待你一句话（纯注释） |
| 5 | `data/labor_law_processed/` 仍是批次 37 旧包（1627 块） | 部署阻塞项 1，**仍等你放行**（纯数据操作） |
| 6 | `install.sh` 四处补丁（发布步 / Node.js / 摘要表 / 法规元数据） | 部署阻塞项，等你选"改脚本 or 出操作手册" |

---

## 附录 A：备份与 md5（`%TEMP%/b41_bak/orig/`）

| 文件 | 改前 md5 | 改后 md5 |
|---|---|---|
| `backend/app/api/chat_persistence.py` | `5fed85eb6fef85df7648695e5130f0dc` | `29b89b754fa12d0c6562104caa54bf7b` |
| `backend/app/chat/chat_history.py` | `5a717a9ee37b33c22a9d2e133d6a158f` | `63ab0e80ebd598062eff75f8e74f1c1b` |
| `backend/app/chat/guard.py` | `7369a2812cd28f6cab570c739f4fec51` | `67fbbe8de2ddd3cfc8772e1128f72441` |
| `backend/app/cli/create_admin.py` | `6beb9f65e8630103b93a559776135c7d` | `41e0dd4a87e8d2707d81f6370e2d3e91` |
| `backend/app/db/vector_index_service.py` | `78d172cd3ec39a889a8a332bef5c2ec4` | `e3e8fe462d8037dd5e9f0766a9f4d1d6` |
| `backend/app/ingest/cleaner.py` | `d40c89e5d7e42e66f7d6eda4b354f0cb` | `62ebf719718b9ddf9a55195cfd0bb2ca` |
| `backend/app/ingest/law_metadata_extractor.py` | `fb94070dc3fb5a8df4cfefbcb03bf132` | `3292f848b392758dd5a7490a660e440d` |
| `backend/app/memory/long_term.py` | `2f1f5da49208a3a8911d9796cea847b7` | `e89ad431ff27d183d0aeb5150b1e035d` |
| `backend/app/memory/memory_schema.py` | `9a6f13dbb0b07f1804d7cb9b83efb8f9` | `8468c36ab721f1dba7d545c836bdb4fd` |
| `backend/app/models/mineru.py` | `637516fc2c4c86be35aa74d85c26c897` | `14e3c24a82dc3b4c9e1b400cc41f578d` |
| `backend/app/models/mineru_api.py` | `4b28b0cc2d0ea93ba41db910324a0fff` | `ad66095988475b232155663b0c6cbacb` |
| `backend/app/retrieval/assembly.py` | `783e3c0e7b946b5a93f46bcaa1966a6a` | `972fc9557b7fe1b54155d4b0f24c7b82` |
| `evaluation/compare_window.py` | `8e4e615fa566e876e8fc454b2c6285ca` | `25b36d379c75a1c658c14ed92847bf6b` |
| `evaluation/faithfulness.py` | `04911565a3beeb9a26744a045fdadf06` | `859f0f988705d5abc0852bec76b1ebae` |
| `evaluation/multi_turn_grading.py` | `a5eac62a6b484b1ca5e5c58fdb7df02f` | `7b5484d16d4712c0082e68fb24e676c0` |
| `scripts/e2e/fetch_pdf_samples.py` | `ff75e6ff6f6b9ee01b2098c76ecd6369` | `aff5992d1e85100871288958579265a9` |
| `scripts/migrations/backfill_law_dates.py` | `3f71fd2af40f8231a3d6e535d4d58099` | `252c58cddf4889181829463628b73baa` |

新增：`backend/tests/test_short_term_window_binding.py`（88 行，纯 LF）。

## 附录 B：本轮临时脚本（`%TEMP%/b41_bak/`，未进仓库）

| 脚本 | 用途 |
|---|---|
| `b41_edit.py` | 备份 + 断言命中数 + 写盘 + 行尾复核（**已修覆盖 bug**） |
| `b41_check.py` | 零引用扫描器 v3（阳性/阴性 + 4 类易误报对照 + 档A/档B 判定） |
| `b41_residual.py` | 残留引用核验（文本提及 / 代码引用 / 再导出消费 三列） |
| `b41_ast_proof.py` | 逐函数 AST 对拍（行为中性主证据） |
| `b41_falsify.py` | 绑定测试证伪（4 例，改坏→跑测试→还原并核对 md5） |
