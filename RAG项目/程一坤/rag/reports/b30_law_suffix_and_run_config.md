# 批次 30 验收报告：评测报告记录当轮变量 + unknown_law 误报修复

**日期**：2026-09-22
**范围**：① 治理规则第 10 条落进 `run_eval`；①b 追查 `LLM_MODEL` 取值来源；② 修 `unknown_law` 检测器把文书名当法规名的误报
**结论**：① 已落地并验证；①b **来源不明**（证据链见下）；② 修复完成，20 题评测 `unknown_law` 计数 **0**；全量 **665 passed**、`check_services` exit 0

---

## 一、① 评测报告自动记录当轮变量

### 改了什么

| 文件 | 改动 |
|---|---|
| `evaluation/run_config.py` | **新增**（113 行）：`collect_run_config()` 采集当轮变量；`format_run_config_lines()` 负责渲染（采集与渲染同源，字段名只出现一次） |
| `evaluation/run_eval.py` | import 一行；`payload` 加顶层字段 `"run_config"`（+4 行，217 行） |
| `evaluation/render_report.py` | 报告头插入 `format_run_config_lines(...)`（+3 行，91 行） |

设计要点：
- **只读不写**：不碰任何状态、不改配置；读不到的项如实写 `None`，报告里显示 `unknown`，
  配置整体读不到时附 `error` 字段——**不猜、不用缺省值填**（报告里一个假值比空值更误导）。
- `PROMPT_FILES` 元组声明要记 md5 的提示词文件，以后提示词再拆文件只需往元组加路径，
  报告字段与渲染都不用改。渲染按元组遍历，当前是 `prompt_builder.py` 一个。
- 兼容旧 payload：`render_markdown` 拿不到 `run_config` 时渲染
  "未记录（旧版报告，先于治理规则第 10 条）"，不会炸。
- 我只改了 `run_eval` 的报告（你说的是它）；`calibrate_refusal` 用的是另一个渲染器
  `refusal_report.py`，未动。

### 实测（用的是真实产出的报告，不是构造的 payload）

`reports/latest_eval_b30_lawsuffix.json` 顶层：

```json
"run_config": {
  "llm_model": "deepseek-flash",
  "llm_temperature": 0.2,
  "llm_max_tokens": 4096,
  "llm_timeout_seconds": 120.0,
  "recall_vector_limit": 20,
  "recall_keyword_limit": 20,
  "rerank_candidate_limit": 20,
  "refusal_min_vector_score": 0.6304,
  "eval_set_md5": "cbf12c3f9370ce1c5311d4bb3f204a73",
  "prompt_files": {
    "backend/app/chat/prompt_builder.py": "f43ee63672ff2b7e136a968441bac640"
  }
}
```

`reports/latest_eval_b30_lawsuffix.md` 报告头：

```
# 检索与回答质量评测报告

- 运行时间：2026-09-22 17:02:21 +0800
- 评测集：`...data/evaluation/eval_set_v1.jsonl`（20 条）
- 检索口径：向量召回 20 + 关键词召回 20 → RRF 融合前 20 → 重排；指标基于重排 top10
- 回答口径：生产默认 top5 上下文 + 真实 LLM
- 当轮变量（`docs/目录与命名约定.md` 第 10 条；JSON 里同源字段 `run_config`）：
  - LLM：`deepseek-flash`（temperature 0.2 / max_tokens 4096 / timeout 120.0s）
  - 提示词：`backend/app/chat/prompt_builder.py` md5 `f43ee63672ff2b7e136a968441bac640`
  - 召回窗口：向量 20 + 关键词 20 → 融合前 20 → 重排
  - 拒答阈值：REFUSAL_MIN_VECTOR_SCORE = 0.6304
  - 评测集 md5：`cbf12c3f9370ce1c5311d4bb3f204a73`
```

验收脚本断言：JSON 六个必备字段缺失 = 无 ✅；Markdown 头四类必备行缺失 = 无 ✅
（见 `reports/b30_eval_accept.txt`）。

**两处超出你字面清单的补充，讲清楚好让你裁决**：
1. 除你点名的四项外，我还记了 `llm_temperature` / `llm_max_tokens` / `llm_timeout_seconds`。
   理由：这三个是生成参数，回答侧指标（无引用数、引用条数）对它们同样敏感，漏了就没法解释跨轮差异。
2. 记了 `eval_set_md5`。理由：题目集内容变了但文件名没变，是同一类"悄悄换变量"，
   光记路径 `eval_set` 挡不住。不需要的话说一声，各删一行即可。
3. `evaluation/` 目录**没有测试目录**（`pytest.ini` 的 `testpaths = tests` 只收 `backend/tests`），
   所以 ① 我没加单测，改用"对真实产出的报告断言必备字段"来验收。

---

## 二、①b `.env` 的 `LLM_MODEL=deepseek-flash` 来源

**结论：来源不明 —— 无法确定何时、因何改成该值。**

排查过的证据源（全部落空）：

| 证据源 | 结果 |
|---|---|
| `.env` 本文件 | mtime = **2026-09-22 15:09:02**（整文件被重写/触碰），`LLM_MODEL` 行**上方没有注释说明**为什么取该值 |
| `.env` 的历史副本 | `%TEMP%` 各批次备份目录、项目内、桌面范围（`find` 全量）**都不存在** `.env` 副本 → 前值不可恢复 |
| 其它记录该键的文件 | 全盘只有 3 处：`.env`、`backend/.env.example`（09-21 11:38，值 `Qwen2.5-14B-Instruct`）、`docs/部署文档.md:216`（占位符 `<大模型名称>`） |
| 数据库痕迹 | `chat_messages.model` 列**全为 NULL** —— 落库处 `backend/app/api/chat_persistence.py:136` 写死 `model=None`，数据库根本不记录服务模型（共 8 行，最早/最晚 09-22 01:07–01:33） |
| `docs/接口文档.md:719` 的 `"model": "deepseek-v4-flash"` | **不能作证据**：落库既写死 NULL，该示例就不是真实抓包，是手写的；且它写的是 `deepseek-v4-flash`，与当前 `deepseek-flash` 也不是同一个字符串 |
| 文档/报告/日志 | 无任何"换模型""模型切换"的记录；`models/llm.py` 只在 docstring 里举例 `deepseek-chat`；`scripts/check_services.py:395` 只是错误提示里提到 `api.deepseek.com` |

可以确定的只有：**09-22 15:09 这次改动早于当天 b28 评测（16:15）、晚于 b29 之前的全部评测**，
与"回答侧行为突变"的时间窗口相邻。因此它在批次 29 报告里被我列为"另一个独立诱因"，
但**无法证实**它当时是否改的正是模型名（也可能只是改了别的键、或仅保存了一次）。

**顺带一个值得修的口子（本次未动，留你裁决）**：`chat_messages.model` 写死 `None`
导致"这个回答是哪个模型生成的"在库里查不到，正好是治理规则第 9/10 条想防的那类漂移。
建议后续让 `persist_turn` 从 `settings.llm_model` 取值写入——那是改行为，等你说。

---

## 三、② `unknown_law` 误报修复

### 问题

回答里写"交警出具的《道路交通事故认定书》是最关键的一份材料"，`《([^》]+)》` 把
**文书名**捞出来当作法规名，与法源清单比对无果 → 抛 `UnknownLawCitationError`
→ 回答被追加"提到清单外法规"警示。护栏只是加警示、不替换回答，但属于纯误报。

### 改法（`backend/app/chat/citation_check.py`，299 行，未超 300 行上限）

新增模块级常量 `LAW_NAME_SUFFIXES`（法规名后缀白名单）+ 私有判定函数
`_is_law_name_candidate()`，`check_citations` 里 `mentioned_laws` 改为**只收白名单命中的候选**：

```python
mentioned_laws = {
    law for law in law_name_pattern.findall(answer) if _is_law_name_candidate(law)
}
```

常量带完整注释说明"为什么用后缀判定"：书名号在回答里不止用于法规（还有《…认定书》《…通知书》），
穷举法规全名不可行（法规无限多、还有简称），但中文规范性文件的通名是封闭集合、
文书名的通名也是（书/证/表/单），故按后缀落在哪一边区分；并写明"加词 / 删词都会改变护栏行为，
改表要配回归测试"。

### ⚠️ 两处我偏离了你给的字面清单，理由是硬证据，请你裁决

**1. 白名单里加了 `法典`。**

你给的列表是 `法 / 条例 / 规定 / 办法 / 解释 / 决定 / 规则 / 细则 / 通则 / 准则 / 标准 / 批复 / 复函`，
按字面实现时 —— 你同时要求的"《民法典》仍要警示"**会失效**：`"民法典".endswith("法")` 是 `False`
（末字是"典"），它会被静默排除，而既有用例
`test_unknown_law_still_rejected_after_normalization`（断言《民法典》必须报错）会立刻变红。
我实测确认了这一点，因此补上 `法典`（注释里写明原因）。若你更希望"保证《民法典》被拦"用别的方式实现，
说一声我改。

**2. 判后缀前先剥掉结尾的序号括号（`_is_law_name_candidate` 里的 `LAW_NAME_SERIAL_SUFFIX`）。**

依据是**库里的真实标题**：11 部法规中有
`最高人民法院关于审理劳动争议案件适用法律问题的解释（一）`，它以 `）` 结尾。
不剥的话它会被当成文书名静默跳过 —— 那就从"误报"换成了"真法规漏判"，方向更糟。
`（一）` 之外也覆盖半角 `(1)`。

### 回归测试（`backend/tests/test_citation_check.py`，+5 用例 → 20 passed）

- `test_document_title_not_reported_as_unknown_law` —— direct-013 场景：回答含《道路交通事故认定书》→ 不抛异常
- `test_common_document_titles_not_reported_as_unknown_law` —— 6 种文书/材料名（…书/…证/…表/…单）全部不参与比对
- `test_real_law_names_are_still_law_candidates` —— 白名单覆盖库里真实名称形态（含 `…解释（一）`、民法典）
- `test_document_titles_are_not_law_candidates` —— 文书名不得被识别为候选
- `test_unknown_law_with_serial_suffix_still_reported` —— 带序号括号的真法规仍报"清单外"

**既有断言全部保留且仍绿**（真提到清单外法规仍要警示）：`test_citation_undefined_source`、
`test_unknown_law_still_rejected_after_normalization`（《民法典》）、
`test_implementation_regulation_not_confused_with_law`（《劳动合同法实施条例》）、
`test_labor_law_and_labor_contract_law_are_different`（《劳动法》≠《劳动合同法》）。

### 反向证明（项目纪律）

把 `mentioned_laws` 那一处**只退回旧行为**（白名单函数定义保留）：

```
新实现： exit=0 | 2 passed, 18 deselected
旧行为： exit=1 | 2 failed  ← test_document_title_not_reported / test_common_document_titles_not_reported
生产代码还原: 字节一致 ✅   （md5 142581ce0396b783fda15e707d0025cf）
还原后复跑: exit=0 | 20 passed
```

> 反向证明脚本第一版直接整文件退回旧版 → 测试在**收集期**就 `ImportError`（新用例 import 了新函数），
> 报的是"import 不到"而不是"行为没锁住"，证明力不足；改成只退回行为那一处才对。
> 同一版脚本还踩了另一个坑：`Path.write_text` 在 Windows 上默认做 `\n → \r\n` 翻译，
> 还原后整文件被改成 CRLF、md5 变了（299 处 CRLF），**已按字节修回并核对 md5 一致**。
> 教训：脚本改源码一律 `read_bytes` / `write_bytes`。

### 真实回答回放（不依赖 LLM 随机性）

拿 b29 两轮里 direct-013 的**真实回答**喂给新旧两版校验器（`reports/b30_replay_direct013.txt`）：

| 轮次 | 回答长度 | 含《道路交通事故认定书》 | 新实现 | 旧行为 |
|---|---|---|---|---|
| b29_cite1 | 1058 | 是 | 通过 ✅ | `UnknownLawCitationError` ← 误报 |
| b29_cite2 | 659 | 是 | 通过 ✅ | `UnknownLawCitationError` ← 误报 |
| b28_after2 | 609 | 否 | 通过 | 通过（对照组：本来就不触发） |

---

## 四、验收结果

| 验收项 | 要求 | 实测 |
|---|---|---|
| ① 报告头记录当轮变量 | 机器可读 | JSON `run_config` 六字段齐全 ✅；MD 头四类行齐全 ✅ |
| ② `unknown_law` 计数 | 0 | **0**（`reports/b30_eval_accept.txt`） |
| ② 引用率不回退 | —— | 无引用回答 **0/20**、引用正确率 1.0、越界 0、Recall@5 1.0 |
| 全量测试 | passed 不减 | **665 passed**（660 基线 + 5 新用例） |
| `check_services` | exit 0 | **exit 0**（Milvus 1627 == MySQL 1627、approved=11） |

护栏分布（20 题）：`citation_check_passed 17` / `citation_warning_uncited_conclusion 3` /
`guardrails_applied 20` / **`citation_warning_unknown_law 0`**（上一轮是 1）。

---

## 五、已知覆盖边界（未改，供你决策）

后缀白名单是"按名称形态"判定，因此**案例类标题**会被排除在"清单外法规"比对之外。
库里 11 部法规里有 2 部的标题属于这一类：
`最高法发布劳动争议典型案例`、`最高法发布劳动争议司法解释（二）和典型案例`（都以"典型案例"结尾）。
影响面：如果回答提到一个**不在本轮清单里**的案例材料名，现在不会再给"提到清单外法规"警示。
实测影响为零 —— b29 两轮回答里出现的 44 处《…》，结尾只分布为 法(43)/例(14)/书(3)/定(10)，
其中"例"全是 `条例 / 实施条例 / 带薪年休假条例`，没有一处是"典型案例"。
若你希望案例材料也纳入比对，往白名单加 `典型案例` 即可（会连带让该词结尾的一切《…》进比对）。

另：`direct-013` 那题的法源清单是 5 条《工伤保险条例》（不同条号），所以误报信息里
"只能使用清单内的法规"把同一部法规列了 5 遍，看着有点怪但无害（清单是按条目而非按法规去重的）。

---

## 六、下一批（你已裁决，本批未做）

`guard.resolve_citation_outcome(answer, error) -> (answer, event_name)`：一处实现、
`service.py` 与 `streaming.py` 两处调用、行为逐字不变、补测试锁住两条路径结果一致。

---

## 七、证据文件

| 文件 | 内容 |
|---|---|
| `reports/b30_eval_accept.txt` | 20 题评测输出 + 报告头字段断言 + 护栏分布 |
| `reports/b30_replay_direct013.txt` | 真实回答回放（新旧校验器对照） |
| `reports/b30_reverse_proof.txt` | 反向证明（新实现绿 → 旧行为红 → 字节还原） |
| `reports/b30_pytest.txt` | 全量测试输出（665 passed） |
| `reports/b30_check_services.txt` | 服务健康自检（exit 0） |
| `reports/eval_20260922_170221_b30_lawsuffix.{json,md}` | 本轮评测报告（含 `run_config` 头） |

改前备份：`%TEMP%/b30_bak/*.before_b30`（4 文件含 md5）。临时脚本同样只在 `%TEMP%/b30_bak/`，
`backend/` 与 `evaluation/` 下无临时文件。未改动 `docs/`。
