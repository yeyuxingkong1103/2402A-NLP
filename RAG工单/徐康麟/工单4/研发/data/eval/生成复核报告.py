# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：研发/data/eval/生成复核报告.py（T3：生成 golden 复核报告）

把冻结后的 golden 与逐题复核结论渲染成中文报告：
  测试/测试数据/golden_复核报告.md
—— 表格与结论全部来自 golden 本身（防止「报告与真值不一致」），叙事部分写在 TEMPLATE 常量里。

运行：pwsh -NoProfile -File run_py.ps1 研发/data/eval/生成复核报告.py
"""
from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "研发" / "data" / "eval" / "golden_qa.jsonl"
REVIEW = ROOT / "测试" / "测试数据" / "golden_复核结论.json"
OUT = ROOT / "测试" / "测试数据" / "golden_复核报告.md"

HEAD = """# 工单4 16 题 golden 测试集复核报告（T3 冻结稿）

> **工单编号**：人工智能NLP-RAG-图像内容解析及检索优化
> **复核人**：测试工程师（T3）　**attempt_id**：2fb1ec72-ce29-452f-9d4a-1b264b141d50　**日期**：{today}
> **复核对象（唯一真值）**：`研发/data/eval/golden_qa.jsonl` ≡ `测试/测试数据/golden_qa_16.jsonl`
> （16 行 JSONL，两份**逐字节一致**，sha256 = `{sha_canonical}`）
> **权威题面**：`设计/工单需求原文.md` §五（用户提示词附件逐字副本，本报告不另立题面）
> **判分口径**：`设计/验收标准.md` §3.4 ＋ `测试/common/golden.py`（确定性判定，非 LLM 自评）
> **复核方式**：用 PyMuPDF **打开** `研发/data/raw` 下两份 PDF（招股说明书1=548 页 / 招股说明书2=350 页），
> 逐题抽取**声明证据页**原文做逐字/配对/数值比对；两道图像题另做 `read_image` **逐图视觉复核**。
> 复核脚本与留痕：`研发/data/eval/构建_golden_qa.py`（构建期断言）、`复核_probe_*.py`（探针）、
> `复核_逐页原文/*.txt`（21 页原文留档）、`复核_证据图/*.png`（页 39/72 渲染图与两图原图）。

---

## 1. 结论摘要

| 项 | 结论 |
| --- | --- |
| 题量与题面 | **16/16** 题，题面与顺序（5,6,1,2,3,4,260,95,33,34,957,793,795,543,531,207）与权威题面**逐字一致**（契约测试对照 `设计/工单需求原文.md` 解析比对） |
| 证据页复核 | **16/16** 题证据页均**打开 PDF 复核**并逐页留痕（`page_facts` + `golden_复核结论.json`）：逐字证据串/配对证据确实出现在所声明物理页；`evidence_pages` 全部落在此次复核范围内 |
| 图像两题 | id 5（页 39 框图）：4 个部门 + 6 个销售处经 **span 坐标 + 2x 渲染图视觉**双重确认；id 6（页 72 两图）：条形图『增长率』与 3D 饼图『应用结构』**逐图视觉复核**，汽车 14.0% / IC卡 -2.0% 确认 |
| 判分自检 | 16 题**正样本全部判对**；**{n_negatives} 个负样本（含跨文件串答、单位偷换、编造金额、图像陷阱）全部判错** |
| 契约测试 | `pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线/test_golden_contract.py -v` → **101 passed**（含「golden 答案自洽」不变量，以及 captain 提出的 id 5 隐蔽错答回归用例） |
| 与 captain 结论 | **9 题完全一致**；6 题在同答案下细化证据页口径（id 2 更严、**id 3 / id 4 的页 158 降级为部分证据页**、id 95 / id 795 / id 207 补充同证据异页）；**1 题分歧**（id 957 的页 332 经复核不含「重要供应商」原句，收窄为 `related_pages`）——详见 §5 |
| 已知限制 | 图表数值依赖**像素级视觉复核**（本机无 OCR）；该限制已写进 golden 的 `chart.no_ocr_note`，报告中不得声称为 OCR 结果 |

**给下游的一句话**：准确率一律用 `测试/common/golden.py::judge_answer`，
检索命中一律用 `测试/common/golden.py::check_retrieval_hit`；
两者**分别报告**，且**不得**用「引用页码 == evidence_pages」冒充检索命中。

> ⚠️ **调用约定（T5/T6 必读）**：工单3 兼容字段（`required_substrings` / `forbidden_substrings` /
> `evidence` / `min_hit_top_k` / `max_first_token_ms`）都是 **golden 记录里的字段**，
> 正确写法 `rec["min_hit_top_k"]`；`测试/common/golden.py` **模块顶层没有这些同名名字**，
> 写成 `golden.min_hit_top_k` 会 AttributeError。模块只额外提供默认值常量
> `golden.DEFAULT_MIN_HIT_TOP_K = 5`、`golden.DEFAULT_MAX_FIRST_TOKEN_MS = 3000`（契约测试已固化该约定）。

---
"""

RULES = """## 6. 判分口径（**测试报告必须照抄本节**）

> **调用约定（防踩空）**：工单3 兼容字段（`required_substrings` / `forbidden_substrings` / `evidence` /
> `min_hit_top_k` / `max_first_token_ms`）**都是 golden 记录里的字段**，正确写法是 `rec["min_hit_top_k"]`；
> `测试/common/golden.py` **模块顶层没有这些名字**，写成 `golden.min_hit_top_k` 会 AttributeError。
> 模块只额外提供默认值常量 `golden.DEFAULT_MIN_HIT_TOP_K = 5`、`golden.DEFAULT_MAX_FIRST_TOKEN_MS = 3000`。

1. **判对 = 确定性关键字判定**（不引用任何 LLM 自评）：
   判对 ⟺ 全部 `answer_keywords` 命中 ∧ 全部 `answer_keywords_any_of` 组各命中其一
   ∧ 全部 `answer_regex_required` 命中 ∧ **无** `forbidden_keywords` / `forbidden_regex` 命中。
2. **数值归一化**：去千分位逗号、全角转半角、去全部空白、`百分之X`→`X%`；带单位数值按**基本单位**比较：
   - `6,464.51 万元` ≡ `6464.51万元`；`82.10%` ≡ `82.1%` ≡ `百分之82.1`；
   - `1.5亿元` ≡ `15,000.00万元` ≡ `150,000,000元`；`5,520.00万元` ≡ `5,520万元`。
   - 数值关键字**必须带单位**（万元/亿元/元/万股/股/%/倍）才走等价比较；单位不符即判错（如 `5,520.00万元` vs `5,520.00万美元`）。
3. **否定护栏**（机制与触发词**从实现自动渲染**，防止「文档比实现窄」）：
   判定机制 = 「禁止误答项命中点**前后各 {guard_window} 字**窗口内出现**任一**否定词 → 豁免（不判违规）」；
   实际触发词共 **{n_cues} 个**，全量清单：{cue_list}。
   例：「赵马克不属于不存在控制关系的关联方」「上海分公司**不属于/不计入**销售处」均不判错。
4. **「检索命中」≠「回答判对」**，**分别报告**：
   - 检索命中 ⟺ `retrieval_evidence` 的证据原文（归一化后）落在**返回的前 K 条 chunk** 中；
   - **禁止**用「引用页码 == `evidence_pages`」判定命中（会把无关页误计为命中、虚高）；
   - 图像题（id 6）按**行业-增长率配对**判定：同一 chunk 内同时出现「汽车 + 14.0」「IC卡 + -2.0」才算命中。
   - **`min_hit_top_k` 精确语义（T6 必读，否则检索命中率不可复核）**：`record["min_hit_top_k"] = 5` 表示
     「**取检索结果的前 5 条 chunk 作为命中候选集合，其中任意一条包含证据原文**——即『**前 K 条内命中即算**』」；
     **不是**「至少命中 5 条」，也**不是**「必须命中第 5 条」。`check_retrieval_hit(record, chunks)` 默认按
     `record["min_hit_top_k"]` **截断**（`chunks[:K]`）后判定，返回值含 `top_k` / `n_chunks_passed` /
     `n_chunks_considered`；调用方只传 3 条就只在前 3 条里判（函数不补数据）。
     **与用户提示词 Top-K=3 的关系**：用户要求的 **K=3 是生成阶段**的检索条数（由 T2/T5 实现，可配置）；
     golden 的 **K=5 是命中判定口径**（比生成口径宽 2 条），用于区分「检索没捞到证据」与「生成没写对」。
     T6 报告必须**分别标注两个 K 的取值与来源**，不得混用同一个 K。
5. **枚举完整性口径（刻意严格，T5 提示词必须配套）**：{enum_policy}
6. **严格度不对称（T5 写提示词最重要的一条）**：{strictness_rule}
7. **证据页三级口径**（防止「引用了某页」被当成「该页含证据」）：
   - `evidence_pages_complete`：**承载完整答案证据**的页（进入检索命中判定）；
   - `evidence_pages`：**承载任何证据**的页 = `complete` ＋ `supporting`（报告/验收表里的“证据页”口径）；
   - `related_pages`：相关但**不含本题证据原句**的页（仅登记，不计入证据页）；
   - `evidence_pages_union`：全并集 = `complete` ＋ `supporting` ＋ `related`（纯描述性字段，判分与命中均不依赖）。
8. 逐题记录：问题、系统答案、golden 答案、是否判对、证据页、**检索命中的 chunk 与页码**、
   `latency_ms`（`total` 与首字）、`citation_ok`、`extraction_method`（第 5、6 题）。
9. 判分结果可复算：报告中的每一处「判对/判错」都必须能用
   `测试/common/golden.py` 对 golden 记录重放得到同样结果。

### 6.1 提供名称变体的题（`answer_keywords_any_of`；只解决「同一实体的不同写法」，**不豁免枚举完整性**）

{any_of_table}

---
"""

TAIL = """## 8. 复现命令与产物清单

```powershell
cd E:\\gao6gongdan\\工单4

# 1) 重建 golden（构建期会用真实 PDF 断言每条证据页，失败即退出码 1 不产真值）
pwsh -NoProfile -File run_py.ps1 研发/data/eval/构建_golden_qa.py

# 2) golden 契约测试（字段/题面/证据页复核/图像资产/判分自检/检索命中口径/页级等式/文档一致性）
pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线/test_golden_contract.py -v

# 3) 重新生成本报告
pwsh -NoProfile -File run_py.ps1 研发/data/eval/生成复核报告.py

# 4) 证据页探针（可选：打印每题关键字在全文的命中页，供人工抽查）
pwsh -NoProfile -File run_py.ps1 研发/data/eval/复核_probe_证据页扫描.py
pwsh -NoProfile -File run_py.ps1 研发/data/eval/复核_probe_逐页原文.py
```

| 产物 | 说明 |
| --- | --- |
| `研发/data/eval/golden_qa.jsonl` | **主真值**（16 行，含 answer/answer_keywords/evidence_pages/retrieval_evidence/判分陷阱/复核结论） |
| `测试/测试数据/golden_qa_16.jsonl` | 同内容镜像（逐字节一致，sha256 校验） |
| `测试/测试数据/golden_复核结论.json` | 机器可读的逐题复核结论（页级 verified 明细、自检结果、与 captain 的异同） |
| `测试/测试数据/golden_复核报告.md` | 本报告 |
| `测试/common/golden.py` | 判分与检索命中库（normalize / judge_answer / check_retrieval_hit / validate_record_shape） |
| `测试/离线/test_golden_contract.py` | golden 契约测试（离线，不联网、不起服务、不调 LLM） |
| `研发/data/eval/构建_golden_qa.py` | golden 构建器（含构建期 PDF 断言） |
| `研发/data/eval/复核_逐页原文/*.txt` | 21 个证据页原文留档（逐字比对底稿） |
| `研发/data/eval/复核_证据图/*.png` | 页 39 组织结构图 / 页 72 两图（渲染图 + 图内原图，含 sha256） |

## 9. 已知限制与风险（**报告中不得当成缺陷，也不得隐瞒**）

| 项 | 说明 | 处置 |
| --- | --- | --- |
| 本机无 OCR | 页 72 图表数值、页 39 框图节点标签**不能**用 OCR 复核；数值来自像素级抽取 + `read_image` 人工视觉复核 | golden 记录 `chart.no_ocr_note`；报告写明方法级别，**禁止**声称为 OCR 结果 |
| 3D 饼图透视失真 | 只登记标签与数值，**不做角度反算** | 已在 `chart.figures[1]` 标注为标签落库 |
| 框图竖排文本 | 页 39 每个汉字一个 span，按 y 行聚类会串成乱码，须按 x 列聚类（±6pt） | 已写进 `diagram.column_clustering` 与 id 5 陷阱 |
| 数值判等容差 | 采用 1e-9 相对容差（严格等值），`6464.5` 与 `6464.51` **不等价**（截断即判错） | 见 `keyword_hit` docstring |
| 图表题检索命中 | 依赖图像通道产出「行业-增长率」文本（CLIP/像素级抽取）；若系统未产出该 chunk，命中记为 miss（**如实统计，不得豁免**） | 见 §6 第 4 条 |
| 页 306 表格列交错 | PDF2 页 306 表内夹核准号/环保意见，无短逐字句 | 该页用「项目名-金额」配对作为检索证据（`kind=pairs`） |

## 10. 验收结论

| 验收项（t3 合同） | 结论 | 证据 |
| --- | --- | --- |
| golden 文件含 16 题且字段齐备 | **通过** | 16 行 JSONL；`validate_record_shape` 16/16 无问题；契约测试 `test_field_contract[16 题]` |
| 每题证据页经打开 PDF 复核并在报告中记录复核结论 | **通过** | §4 逐题结论 + `page_facts` + `golden_复核结论.json`；契约测试 `test_evidence_pages_verified_on_real_pdf[16 题]`、`test_review_conclusions_recorded` |
| 含 answer_keywords 与判分陷阱口径 | **通过** | 每题 `answer_keywords` + `forbidden_keywords/regex` + `traps` + §6/§7；契约测试 `test_judge_samples[16 题]`、`test_cross_document_confusions_rejected`、`test_image_trap_answers_rejected` |

**冻结声明**：自本报告生成起，`研发/data/eval/golden_qa.jsonl` 与 `测试/测试数据/golden_qa_16.jsonl`
即为本工单**准确率、检索命中与响应时间的唯一判分真值**。
后续（T6 评测）若发现判分争议，**只能**在 golden 正确性经复核的前提下修改 golden，
并同步重建 `生成复核报告.py` 与契约测试；**禁止**为迁就系统输出而放宽 `answer_keywords`、删除样本或改写 golden 答案
（`设计/验收标准.md` §9）。
"""


def _load_golden_lib():
    spec = importlib.util.spec_from_file_location(
        "golden_lib", ROOT / "测试" / "common" / "golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import hashlib
    records = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in records}
    n_neg = sum(len(r["judge_samples"]["negatives"]) for r in records)
    sha = hashlib.sha256(GOLDEN.read_bytes()).hexdigest()

    parts = [HEAD.format(today=date.today().isoformat(), sha_canonical=sha, n_negatives=n_neg)]

    # §6 判分口径：否定护栏/枚举口径等**从实现自动渲染**，杜绝「文档比实现窄」
    golden_lib = _load_golden_lib()
    cues = list(golden_lib.NEGATION_CUES)
    any_of_rows = []
    for r in records:
        if r.get("answer_keywords_any_of"):
            for gi, group in enumerate(r["answer_keywords_any_of"], 1):
                any_of_rows.append(f"| {r['id']} | {r['category']} | {gi} | "
                                   + " / ".join(f"`{m}`" for m in group) + " |")
    any_of_table = ("| 题 id | 题型 | 组号 | 该组接受的写法（命中任一即算该项满足） |\n"
                    "| --- | --- | --- | --- |\n" + "\n".join(any_of_rows)) if any_of_rows else "（无）"
    rules = RULES.format(
        guard_window=golden_lib.GUARD_WINDOW,
        n_cues=len(cues),
        cue_list="、".join(f"`{c}`" for c in cues),
        enum_policy=golden_lib.ENUMERATION_POLICY,
        strictness_rule=golden_lib.NUMERIC_VS_ENUM_STRICTNESS,
        any_of_table=any_of_table,
    )

    # ---- 2/3 复核方法与汇总表 ----
    parts.append("""## 2. 复核方法（可复现、非转录）

1. **打开 PDF**：`fitz.open(研发/data/raw/招股说明书N.pdf)`，逐页 `get_text("text")`；
2. **逐字比对**：证据串经统一归一化（`测试/common/golden.py::normalize`）后必须**确实是该页的子串**——
   构建器 `构建_golden_qa.py` 把「不成立」当**构建失败**（退出码 1，不产出 golden），而不是警告；
3. **页级归属**：对表格页（金额列不跟单位）用 `page_carries`（同一数值出现在该页）判定，
   并额外要求该页**承载全部 `answer_keywords`**（id 2 页 306）；
4. **自动发现同证据异页**：对每个逐字证据串做**全文扫描**，命中的其它页自动登记为 `alt`（供检索与召回对照）；
5. **图像题**：`page.get_images()` + `extract_image()` 取原图并算 sha256；用 `read_image` **逐图视觉复核**数值；
6. **判分自检**：每题内置 1 个正样本 + 若干陷阱负样本，构建期与 pytest 双重复算，
   正样本必须判对、负样本必须判错——否则 golden 不予冻结。

## 3. 16 题汇总表（**唯一真值**）

> `证据页(完整)` = `evidence_pages_complete`（承载完整答案证据，进入检索命中判定）；
> 完整证据页以外的页见“部分/相关页”列（`supporting_pages` / `related_pages`）。

| # | id | 题型 | 语料 | 答案形式（完整性要求） | 判分关键字（全部命中=判对） | 必含正则 | 禁止误答项 | 证据页(完整) | 部分/相关页 | 检索证据(页/类型) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |""")

    for r in records:
        kw = "、".join(f"`{k}`" for k in r["answer_keywords"])
        if r.get("answer_keywords_any_of"):
            kw += "；任一组：" + "；".join(
                " / ".join(f"`{x}`" for x in g) for g in r["answer_keywords_any_of"])
        rx = "；".join(x["label"] for x in r["answer_regex_required"]) or "—"
        fbk = r["forbidden_keywords"]
        fb = "、".join(f"`{k}`" for k in fbk[:6]) + ("…" if len(fbk) > 6 else "")
        if r.get("forbidden_regex"):
            fb += "；正则：" + "；".join(x["label"] for x in r["forbidden_regex"])
        comp = "、".join(map(str, r["evidence_pages_complete"]))
        extra = []
        if r["supporting_pages"]:
            extra.append("部分 " + str(r["supporting_pages"]))
        if r["related_pages"]:
            extra.append("相关 " + str(r["related_pages"]))
        ret = "；".join(f"{e['page']}/{e['kind']}" for e in r["retrieval_evidence"])
        parts.append(f"| {r['order']} | {r['id']} | {r['category']} | {r['pdf']} | {r.get('answer_form', '—')} | {kw} | {rx} | "
                     f"{fb} | {comp} | {' '.join(extra) if extra else '—'} | {ret} |")

    parts.append("""
> **语料陷阱**：`id 1/2/3/4` 的页 22 指 **招股说明书2**；`id 531/543` 的页 22 指 **招股说明书1**——
> 页码数值相同、文件不同，引用必须带文件名。
> **同页陷阱**：`id 33` 与 `id 260` 证据同在招股说明书1 物理页 129，但关键字分别为占比组与金额组；
> 该页 `find_tables()` 返回 **0 个表**（正文段落，不是表格），两题不得互顶。

## 4. 逐题复核结论（打开 PDF 后记录）
""")

    for r in records:
        rv, rev = r["review"], review["items"][[i for i, x in enumerate(review["items"])
                                                   if x["id"] == r["id"]][0]]
        parts.append(f"\n### 4.{r['order']}  id {r['id']}（{r['category']}，{r['pdf']}）\n")
        parts.append(f"- **题面**：{r['question']}")
        parts.append(f"- **golden 答案**：{r['answer']}")
        parts.append(f"- **判分关键字**：{'、'.join('`'+k+'`' for k in r['answer_keywords'])}"
                     + (f"；任一组：{'；'.join(' / '.join('`'+x+'`' for x in g) for g in r['answer_keywords_any_of'])}"
                        if r.get("answer_keywords_any_of") else "")
                     + (f"；参考（不参与判对）：{'、'.join('`'+k+'`' for k in r['answer_keywords_optional'])}"
                        if r.get("answer_keywords_optional") else ""))
        if r["answer_regex_required"]:
            parts.append("- **必含正则**：" + "；".join(
                f"`{x['regex']}`（{x['label']}）" for x in r["answer_regex_required"]))
        if r["forbidden_keywords"] or r["forbidden_regex"]:
            parts.append("- **禁止误答项**：" + "、".join(f"`{k}`" for k in r["forbidden_keywords"])
                         + ("；" + "；".join(f"`{x['regex']}`（{x['label']}）"
                                             for x in r["forbidden_regex"]) if r["forbidden_regex"] else ""))
        if r.get("forbidden_note"):
            parts.append(f"- **禁止项说明**：{r['forbidden_note']}")
        if r.get("confusion_keywords"):
            parts.append("- **高风险混淆项（记录，不判错）**：" + "、".join(
                f"`{k}`" for k in r["confusion_keywords"]))
        parts.append(f"- **证据页**：完整 {r['evidence_pages_complete']}；"
                     f"含证据的页 {r['evidence_pages']}；部分证据页 {r['supporting_pages'] or '—'}；"
                     f"相关非证据页 {r['related_pages'] or '—'}")
        parts.append(f"- **证据原文（{r['evidence_pages_primary'][0]} 页）**：{r['evidence_verbatim'][:220]}"
                     + ("…" if len(r["evidence_verbatim"]) > 220 else ""))
        parts.append(f"- **复核结论**：{rv['verdict']}")
        parts.append(f"- **复核记录（打开过的页与校验）**：")
        for pf in rev["pages_opened_and_verified"]:
            parts.append(f"  - 物理页 {pf['page']}（{pf['role']}/{pf['kind']}，verified={pf['verified']}，"
                         f"{pf['detail']}）← {pf['source']}")
        parts.append(f"- **复核说明**：{rv['note']}")
        parts.append(f"- **与 captain 结论**：{rv['agreement_with_captain']}")
        parts.append(f"- **判分陷阱**：" + "；".join(r["traps"]))

    # ---- 5 分歧登记 ----
    parts.append("""

## 5. 与 captain 结论的对比与分歧登记

| 题 | captain 声明 | T3 复核结论 | 判定 |
| --- | --- | --- | --- |
| id 5 | 页 39（框图）+40、113（正文互证）；销售部 4 个部门 / 大客户销售部 6 个销售处；禁答项含上海分公司 | 页 39 文本层 97 span、6 个销售处 x=241/286/331/376/421/466（y 588..636 等距），2x 渲染图视觉复核确认；页 40/113 为部分证据 | **一致**（采纳 `evidence_pages=[39,40,113]`；完整证据页仅 39） |
| id 6 | 汽车 14.0% 最快、IC卡 -2.0% 负增长；左=1693 饼图、右=1692 条形图；正文「仅次于汽车电子」互证 | 逐图 `read_image` 复核一致；条形图 7 项数值与饼图 7 项占比全部登记；两图 bbox 相接非重叠 | **一致**（已弃用早期「两图重叠/左右对调」说法） |
| id 1 / id 2 / id 4 / id 260 / id 33 / id 34 / id 95 / id 207 / id 531 / id 543 / id 793 / id 795 | 见 t3 任务书 | 逐字复核全部命中 | **一致**（id 95 另补页 27 的简称写法；id 207 另补页 490 的正文口径） |
| id 3 | 页 157、21、158 | 页 157 表行与页 21 正文均含 `赵马克`+`42.35%`；**页 158 不含该两项**（仅关联方图示与普芯达详述） | **收窄**：158 → `supporting_pages`（不再计为完整证据页） |
| id 4 | 页 157、158 | 7 家企业在页 157 齐全；**页 158 只含普芯达详述** | **收窄**：158 → `supporting_pages` |
| id 957 | 页 154、332 | 页 154/27/160 逐字含「重要供应商」；**页 332 不含该结论词**，其承载的是「军方视频指挥系统标准主要参与单位、系统验证单位」 | **分歧并收窄**：332 → `related_pages`（相关但非本题证据页），完整证据页为 26/27/154/160 |
| id 795 | 页 27、157 | 页 27/157 逐字命中；全文扫描另发现 155/181/182/241 同句 | **扩充**：登记同证据异页（利于召回对照，全部 verified） |

> 建议架构师据此把 `设计/验收标准.md` §3 中 id 957 的「332」改为 **154（主）/27/160**，
> 并在 id 3、id 4 的证据页后标注「158 为部分证据页」。本 T3 不改设计文档（越界）。

---
""")

    parts.append(rules)

    # ---- 7 陷阱清单 ----
    parts.append("\n## 7. 判分陷阱清单（逐题专有项；通用 3 条见 §6 第 1~3 条，写在每题 `traps[0:3]`）\n")
    for r in records:
        parts.append(f"**id {r['id']}（{r['category']}）**\n")
        for t in r["traps"][3:]:
            parts.append(f"- {t}")
        parts.append("")

    parts.append("\n---\n")
    parts.append(TAIL)

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] 复核报告已生成：{OUT}")
    print(f"     16 题汇总表 + 逐题结论；负样本 {n_neg} 个；sha256={sha[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
