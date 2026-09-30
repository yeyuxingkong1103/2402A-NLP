# 症状挂卡覆盖层 实施计划（试点批）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让症状型问法（农户描述现象、不说病名）能命中正确的卡，且不引入编造——先采 5 条症状跑通全链路，验证有效再铺开。

**Architecture:** 症状以**独立覆盖层**存在（`data/processed/symptoms.jsonl`，逐条带字段级来源），卡本体一个字不动。切分时合并成 chunk 的 `sym_text`（喂编码）与 `symptoms`（供答案层）；`embed_text`/`bm25_text` 保持原样，避免污染检索层从 `bm25_text` 现算的实体表。生成层新增【症状对照】段，逐字不过模型，措辞是"与你描述相符"而非下诊断。

**Tech Stack:** Python 3 / pytest / Milvus Lite / BGE-M3 + BGE-reranker-v2-m3（本地权重）/ FastAPI（不在本批范围）

**Spec:** `docs/superpowers/specs/2026-09-28-symptoms-field-design.md`

## Global Constraints

- **逐字照抄**：`symptom_text` 不改写、不概括、不换算（核心约束 6）
- **检索层 `src/retrieve/search.py` 一行不改**（spec §4.3）
- **症状不参与实体表**：`_MENTIONED` / `_MENTION_CROPS` 仍从卡本体现算（spec §4.1）
- **247 张卡 / 203 chunk 两个数字不变**；症状条目数单列（spec §12）
- **症状文本不得与评估问句同源**：评估集的症状问句来自 12396，故**禁止**从 12396 相关页面抄症状文本（spec §8）
- **高危数值仍逐字来自卡片，模型不参与**（核心约束 1）
- **代码注释全中文**，与全库风格一致
- 每个任务结束必须 `python -m pytest -q` 全绿再提交

---

## 文件结构

| 文件 | 动作 | 职责 |
| --- | --- | --- |
| `data/processed/symptoms.jsonl` | 新建 | 症状覆盖层（一行一条，带来源） |
| `data/external/symptoms_raw/` | 新建 | 逐条原文快照，来源可回查 |
| `scripts/validate_symptoms.py` | 新建 | 覆盖层校验 CLI（可被 pytest 导入） |
| `scripts/make_symptoms_checklist.py` | 新建 | 生成 `症状挂卡清单.md`（人工复核用） |
| `tests/test_symptoms.py` | 新建 | 校验器 + 切分合并的回归锁 |
| `scripts/build_chunks.py` | 改 | 读覆盖层 → chunk 加 `sym_text`/`symptoms`/`meta.has_symptoms`；挂空硬校验 |
| `src/index/embed_chunks.py` | 改 | 编码输入拼接 `sym_text`；`embedding_meta` 留口径 |
| `src/index/build_milvus_index.py` | 改 | schema 加 `sym_text`/`has_symptoms`（不建倒排） |
| `src/generate/answer.py` | 改 | 【症状对照】段 + 对照口吻结论 |
| `tests/test_generate_template.py` | 改 | 加 5 个生成层用例 |
| `eval/run_eval.py` | 改 | `false_fb` 拆两个计数器 |
| `eval/datasets/collect.jsonl` | 改 | 加回 4 条 + 第 18 条改写 |
| `CLAUDE.md` / `README.md` / `docs/01` / `docs/06` | 改 | 口径与产物同步 |

---

## Task 1: false_fb 计数器拆分（先做，后面评估的数字才可信）

**Files:**
- Modify: `eval/run_eval.py:131-133`（桶初始化）、`:172`、`:178`、`:251-253`、`:257`

**Interfaces:**
- Consumes: 无
- Produces: 报告 JSON 的桶里新增 `false_fb_positive` 与 `false_fb_noplan` 两个键（`false_fb` 键消失）

- [ ] **Step 1: 改桶初始化**

`eval/run_eval.py:131-133`，把 `"false_fb": 0` 换成两个键：

```python
    buckets = defaultdict(lambda: {"n": 0, "recall": 0.0, "mrr": 0.0, "n_pos": 0,
                                   "false_fb_positive": 0, "false_fb_noplan": 0,
                                   "missed_fb": 0, "n_fb": 0,
                                   "num_ok": 0, "num_tot": 0, "num_fail": 0, "relaxed": 0})
```

- [ ] **Step 2: 改 noplan 分支（该放宽却兜底）**

`eval/run_eval.py:172`：

```python
                b["false_fb_noplan"] += 1  # 该放宽却硬兜底了
```

- [ ] **Step 3: 改正例分支（误兜底）**

`eval/run_eval.py:178`：

```python
                b["false_fb_positive"] += 1  # 误兜底：该答的答了「暂无」
```

- [ ] **Step 4: 改报告的两处读数**

`eval/run_eval.py:249-253`（误兜底行只读 positive）：

```python
        if b["n_pos"]:
            # 误兜底率：**只读正例分支的计数器**（false_fb_positive）。
            # 原实现一个 false_fb 被两处复用，noplan 分支一旦触发会污染这一行。
            ff = b["false_fb_positive"] / b["n_pos"]
            lines.append(f"   误兜底 {b['false_fb_positive']}/{b['n_pos']} = {ff:.1%}"
                         f"  {'✅' if ff < 0.05 else '❌ 应 <5%'}")
            if ff >= 0.05:
                ok_all = False
```

`eval/run_eval.py:257`（该放宽却兜底行只读 noplan）：

```python
            lines.append(f"   该放宽却兜底 {b['false_fb_noplan']}/{b['n_noplan']}"
                         f"（应返回正文+提示，不是硬兜底）")
```

- [ ] **Step 5: 确认没有残留**

Run: `grep -n "false_fb" eval/run_eval.py`
Expected: 只剩 `false_fb_positive` 与 `false_fb_noplan`，没有裸 `false_fb`

- [ ] **Step 6: 跑一次小样本评估验证报告结构**

Run: `python eval/run_eval.py --limit 5 --no-gen`
Expected: 正常运行；随后 `python -c "import json;print(sorted(json.load(open('eval/report.json',encoding='utf-8'))['buckets'].values())[0])"` 打印的键里含 `false_fb_positive` 与 `false_fb_noplan`，不含 `false_fb`

⚠️ 这一步会覆盖 `eval/report.json`。**先备份**：
`cp eval/report.json eval/report_baseline_20260928.json`

> 📌 **本步及下面两步是 Task 1 已执行的历史命令，文件名保持原样不改**——
> 该快照后由 **Task 7 `git mv` 为 `eval/report_before_symptoms.json`**（内容一字未动）。
> 现在要引用它请用新名。

- [ ] **Step 7: 恢复基线报告**

Run: `cp eval/report_baseline_20260928.json eval/report.json`（Task 7 之前不动基线；该文件现名 `eval/report_before_symptoms.json`）

- [ ] **Step 8: 提交**

```bash
git add eval/run_eval.py eval/report_baseline_20260928.json   # 当时名；Task 7 已 git mv 为 report_before_symptoms.json
git commit -m "fix: false_fb 拆成 false_fb_positive / false_fb_noplan，误兜底只读正例分支"
```

---

## Task 2: 覆盖层校验器 `validate_symptoms.py`

**Files:**
- Create: `scripts/validate_symptoms.py`
- Create: `tests/test_symptoms.py`
- Test: `tests/test_symptoms.py`

**Interfaces:**
- Consumes: `data/processed/symptoms.jsonl`、`data/chunks/chunks.jsonl`
- Produces:
  - `PART_ENUM: tuple[str, ...]` —— 九值枚举
  - `ROOT / PROCESSED / SYMPTOMS / CHUNKS` 路径常量
  - `load_cards() -> dict[str, dict]`（card_id → 卡，来自三份卡 JSONL）
  - `load_chunks() -> set[str]`（进索引的 chunk_id 集合）
  - `load_symptoms(path=None) -> list[dict]`
  - `validate(rows, card_ids, chunk_ids) -> list[str]`（错误列表，空=通过）
  - `collect(path=None) -> tuple[list, list, dict]`（rows, errs, stats；供 pytest）
  - `main() -> int`（CLI 退出码）

- [ ] **Step 1: 写失败测试**

Create `tests/test_symptoms.py`：

```python
# -*- coding: utf-8 -*-
"""症状覆盖层自检：校验器 + 切分合并的回归锁。

为什么单来一层：症状条目是**外部来源**（期刊/农科院/科普），挂在国标卡上。
挂空 = 症状静默丢失；挂错卡 = 把 A 病的症状安到 B 病头上，会给出错药。
不联网、不加载模型。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_symptoms as V  # noqa: E402  被测对象


def _ok_row():
    """一条合法样例。"""
    return {
        "symptom_id": "sym-test-01",
        "card_id": "CARD-A",
        "crop": "大蒜",
        "pest": "豌豆潜叶蝇",
        "symptom_text": "幼虫在叶片表皮下潜食叶肉，形成白色弯曲虫道",
        "part": ["叶片"],
        "source": {"kind": "journal", "title": "测试文献", "publisher": "《测试》2020(1)",
                   "url": None, "accessed": "2026-09-28", "quote": "原文"},
        "authority_level": 2,
        "needs_human_review": True,
    }


def test_合法样例通过():
    errs = V.validate([_ok_row()], {"CARD-A"}, {"CARD-A"})
    assert not errs, errs


def test_抓得住挂到不存在的卡():
    errs = V.validate([_ok_row()], set(), set())
    assert any("不存在" in e for e in errs), errs


def test_抓得住挂了不进索引的卡():
    """卡存在但没进索引（附录A / 光杆标题卡）——症状会静默丢失。"""
    errs = V.validate([_ok_row()], {"CARD-A"}, set())
    assert any("不进索引" in e for e in errs), errs


def test_抓得住空症状文本():
    r = _ok_row()
    r["symptom_text"] = "   "
    errs = V.validate([r], {"CARD-A"}, {"CARD-A"})
    assert any("symptom_text" in e for e in errs), errs


def test_抓得住非法part():
    r = _ok_row()
    r["part"] = ["叶子"]
    errs = V.validate([r], {"CARD-A"}, {"CARD-A"})
    assert any("part" in e for e in errs), errs


def test_抓得住挂错卡():
    """pest 必须是目标卡的独立词元；挂到别的病虫害卡上要拦下。"""
    cards = {"CARD-A": {"subtype": "蓟马", "source": {"quote": "蓟马｜单株2头~3头"}}}
    errs = V.validate([_ok_row()], {"CARD-A"}, {"CARD-A"}, cards=cards)
    assert any("独立词元" in e for e in errs), errs


def test_子串不算数_疫病挂不到早晚疫卡():
    """假阳性方向：「疫病」是「早疫病、晚疫病」的子串，必须判为对不上——
    挂上去就是把疫病症状安到早/晚疫方案卡上，等于给错药。"""
    r = _ok_row()
    r["pest"] = "疫病"
    cards = {"CARD-A": {"subtype": "早疫病、晚疫病",
                        "source": {"quote": "早疫病、晚疫病｜初见病叶｜方案一:…"}}}
    errs = V.validate([r], {"CARD-A"}, {"CARD-A"}, cards=cards)
    assert any("独立词元" in e for e in errs), errs


def test_正文卡按quote认病虫害():
    """假阴性方向：正文卡 subtype 就是章节号，病虫害名只在 quote 里——必须放行。"""
    r = _ok_row()
    r["pest"] = "疫病"
    cards = {"CARD-A": {"subtype": "6.2",
                        "source": {"quote": "主要防治对象为:猝倒病、立枯病、疫病、病毒病等。"}}}
    errs = V.validate([r], {"CARD-A"}, {"CARD-A"}, cards=cards)
    assert not errs, errs


def test_抓得住重复symptom_id():
    """两条同 id 会让复核清单与快照文件名互相覆盖。"""
    errs = V.validate([_ok_row(), _ok_row()], {"CARD-A"}, {"CARD-A"})
    assert any("symptom_id 重复" in e for e in errs), errs


@pytest.mark.skipif(not (ROOT / "data/processed/symptoms.jsonl").exists(),
                    reason="覆盖层未生成（Task 3 之前正常）")
def test_真实覆盖层全部通过():
    """真实文件必须零错误——有错就是挂错了卡或写漏了字段。"""
    rows, errs, _ = V.collect()
    assert not errs, "症状覆盖层校验未通过:\n" + "\n".join(f"  ✗ {e}" for e in errs)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_symptoms.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'validate_symptoms'`

- [ ] **Step 3: 写校验器**

Create `scripts/validate_symptoms.py`：

```python
# -*- coding: utf-8 -*-
"""症状覆盖层自检：data/processed/symptoms.jsonl 的挂卡与字段校验。

症状条目是**外部来源**（期刊/农科院/科普）挂在国标卡上的覆盖层，
三类错误都会造成实际损害：

  1. **挂空**（card_id 不存在）→ 症状静默丢失，农户问症状永远查不到
  2. **挂了不进索引的卡**（附录A 36 张 / 光杆标题 8 张）→ 同样静默丢失
  3. **挂错卡**（pest 与目标卡 subtype 对不上）→ 把 A 病的症状安到 B 病头上，
     检索命中 B 卡 → 农户拿到**错药**

用法：
    python scripts/validate_symptoms.py        # 退出码 0=通过，1=有错，2=缺文件
"""
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
CHUNKS = os.path.join(ROOT, "data", "chunks", "chunks.jsonl")
SYMPTOMS = os.path.join(PROCESSED, "symptoms.jsonl")
# 三份卡片文件（与 validate_cards.py 口径一致：247 张）
CARD_FILES = ["appendixB_cards.jsonl", "body_cards.jsonl", "appendixA_cards.jsonl"]

# part 九值枚举（spec §3.2）。扩枚举必须同步 tests/test_symptoms.py 与本表。
PART_ENUM = ("叶片", "茎", "茎基部", "根", "鳞茎", "果实", "心叶", "花", "全株")


def load_cards():
    """三份卡片 JSONL → {card_id: 卡}。"""
    cards = {}
    for fn in CARD_FILES:
        p = os.path.join(PROCESSED, fn)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            if line.strip():
                c = json.loads(line)
                cards[c["card_id"]] = c
    return cards


def load_chunks(path=None):
    """chunks.jsonl → 进索引的 chunk_id 集合（203 张里实际被检索的那些）。"""
    path = path or CHUNKS
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["chunk_id"] for l in open(path, encoding="utf-8") if l.strip()}


def load_symptoms(path=None):
    """症状覆盖层 → 条目列表；文件不存在返回空列表。"""
    path = path or SYMPTOMS
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# 词元切分：病虫害名在卡片里可能以 、｜: 等分隔（subtype 与 quote 都按此切）
_TOKEN_SPLIT = re.compile(r"[、,，;；:：|｜/]+")


def _tokens(text):
    """把一段卡片文本切成独立词元集合（去空白、丢空串）。"""
    return {t.strip() for t in _TOKEN_SPLIT.split(text or "") if t.strip()}


def validate(rows, card_ids, chunk_ids, cards=None):
    """校验条目，返回错误字符串列表（空列表 = 通过）。

    rows      : 症状条目列表
    card_ids  : 全部卡片 id 集合（247 张）
    chunk_ids : 进索引的 chunk id 集合（203 张）
    cards     : {card_id: 卡}，给「挂错卡」检查用；None 时跳过该项
    """
    errs = []
    for i, r in enumerate(rows):
        tag = r.get("symptom_id") or f"第 {i+1} 行"
        cid = r.get("card_id")
        # ① 卡号必须真实存在
        if cid not in card_ids:
            errs.append(f"{tag}: card_id {cid!r} 在卡片里不存在")
            continue
        # ② 卡必须真的进了索引，否则症状静默丢失
        if cid not in chunk_ids:
            errs.append(f"{tag}: card_id {cid!r} 不进索引（附录A 或光杆标题卡），症状会丢失")
        # ③ 症状文本非空
        if not (r.get("symptom_text") or "").strip():
            errs.append(f"{tag}: symptom_text 为空")
        # ④ part 必须在枚举内
        for p in r.get("part") or []:
            if p not in PART_ENUM:
                errs.append(f"{tag}: part {p!r} 不在枚举 {PART_ENUM}")
        # ⑤ pest 必须是目标卡里的**独立词元**（subtype 或 source.quote）
        #   为什么不用子串（2026-09-28 用户裁决）——子串两个方向都错：
        #     假阳性：「疫病」是「早疫病、晚疫病」的子串，会让疫病症状挂上早/晚疫
        #             方案卡，正检索错药
        #     假阴性：正文卡的 subtype 就是章节号（b6.2 → "6.2"），病虫害名只出现在
        #             quote 里，子串法会把合法条目全部拒收
        if cards is not None:
            card = cards.get(cid) or {}
            sub = card.get("subtype") or ""
            quote = (card.get("source") or {}).get("quote") or ""
            pest = (r.get("pest") or "").strip()
            if pest and pest not in (_tokens(sub) | _tokens(quote)):
                errs.append(f"{tag}: pest {pest!r} 不是目标卡的独立词元"
                            f"（subtype={sub!r}；quote 前 40 字={quote[:40]!r}）")
        # ⑥ source 必须是对象（来源不可回查的条目等于没有来源）
        if not isinstance(r.get("source"), dict):
            errs.append(f"{tag}: source 不是对象")
    # ⑦ symptom_id 不许重复（重复会让复核清单与快照文件名互相覆盖）
    dup = sorted(k for k, v in Counter(r.get("symptom_id") for r in rows).items()
                 if k and v > 1)
    if dup:
        errs.append(f"symptom_id 重复: {dup}")
    return errs


def collect(path=None):
    """给 pytest 用的整包：返回 (条目列表, 错误列表, 统计)。"""
    cards = load_cards()
    chunks = load_chunks()
    rows = load_symptoms(path)
    errs = validate(rows, set(cards), chunks, cards)

    # 统计时对 source 做类型防护：source 坏掉的条目走的是错误列表，不该拖垮统计
    def _kind(r):
        src = r.get("source")
        return src.get("kind") if isinstance(src, dict) else None

    stats = {"n": len(rows), "by_crop": dict(Counter(r.get("crop") for r in rows)),
             "by_kind": dict(Counter(_kind(r) for r in rows))}
    return rows, errs, stats


def main():
    """CLI：打印条目数与错误明细。返回 0=通过，1=有错，2=缺文件。"""
    if not os.path.exists(SYMPTOMS):
        print(f"缺 {SYMPTOMS}（先采条目，见实施计划 Task 3）")
        return 2
    rows, errs, stats = collect()
    print(f"症状条目 {stats['n']} 条")
    print(f"  按作物: {stats['by_crop']}")
    print(f"  按来源: {stats['by_kind']}")
    if errs:
        print(f"\n✗ 校验未通过，{len(errs)} 个错误：")
        for e in errs:
            print(f"  - {e}")
        return 1
    print("\n✓ 校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_symptoms.py -q`
Expected: PASS（6 条通过，真实的第 7 条 skip）

- [ ] **Step 5: 提交**

```bash
git add scripts/validate_symptoms.py tests/test_symptoms.py
git commit -m "feat: 症状覆盖层校验器（挂空/不进索引/空文本/part 枚举/挂错卡）"
```

---

## Task 3: 采集 5 条症状并落库（试点）

**Files:**
- Create: `data/processed/symptoms.jsonl`
- Create: `data/external/symptoms_raw/`（每条一个快照文件）
- Create: `scripts/make_symptoms_checklist.py`
- Create: `data/processed/症状挂卡清单.md`（脚本产物）

**Interfaces:**
- Consumes: `scripts/validate_symptoms.py`
- Produces: `data/processed/symptoms.jsonl`，字段见 spec §3

**⚠️ 铁律（spec §8）**：这批问句来自 12396，**症状文本一律不得取自 12396 相关页面**。
优先期刊文献（有刊名/年/期/页）；政府/农科院网站可作补充并**必须存快照**。

- [ ] **Step 1: 采集 5 条**（每条：打开原文 → 逐字抄 → 记出处 → 存快照）

**先让快照目录能入库**：`.gitignore` 第 6 行的 `data/external/` 把整个目录忽略了，
而快照是溯源唯一凭据（网页会漂），必须进版本控制。在那行之后加一行例外：

```gitignore
# 例外：症状来源快照要入库（网页会漂，这是溯源唯一凭据）
!data/external/symptoms_raw/
```

然后开始采条：

五个目标与候选来源（候选来自 2026-09-28 侦察，**用前需重新打开原文核对**）：

| # | 目标病虫害 | 挂到哪张卡 | 候选来源（需核对） |
| --- | --- | --- | --- |
| 1 | 辣椒 疫病 | `GBZ26583-2011-p6-b6.2`（正文卡） | 需新找：辣椒疫病症状（期刊/农科院/农牧局页面） |
| 2 | 大蒜 豌豆潜叶蝇 | `GBZ26578-2011-p20-c09` | 《河南农业》2023(13)《河南省大蒜田豌豆植潜蝇的发生及绿色防治》；《内蒙古农业科技》2003(1)；《中国蔬菜》2011(13) p24-25 |
| 3 | 大蒜 蓟马 | `GBZ26578-2011-p20-c10` | ⚠️ 侦察时最好找的是 12396 页面——**禁用**（#6 问句同源）。改找中国农业科技信息网（中国农科院）/期刊 |
| 4 | 辣椒 茶黄螨 | `GBZ26583-2011-p21-c05` | 《北方园艺》2008《朝天椒病毒病和茶黄螨的识别及防治》；中国农科院；通辽市农牧局 |
| 5 | 辣椒 猝倒病、立枯病 | `GBZ26583-2011-p20-c01` | 需新找：辣椒苗期猝倒病/立枯病症状 |

快照落盘：`data/external/symptoms_raw/<symptom_id>.md`，内容 = 出处信息 + 抄录原文 + 抓取时间。
网页类快照另存 `raw.html`（页面会漂，这是唯一凭据）。

- [ ] **Step 2: 写 `data/processed/symptoms.jsonl`**

一行一条，格式：

```json
{"symptom_id": "sym-lajiao-yibing-01", "card_id": "GBZ26583-2011-p6-b6.2", "crop": "辣椒", "pest": "疫病", "symptom_text": "（逐字抄录的原文）", "part": ["茎基部"], "source": {"kind": "journal", "title": "（篇名）", "publisher": "（刊名 年(期)，或机构名+日期）", "url": null, "accessed": "2026-09-28", "quote": "（原文 ≤200 字）"}, "authority_level": 2, "needs_human_review": true}
```

⚠️ `pest` 必须是目标卡里的**独立词元**，出现在 `subtype` **或** `source.quote` 里
（按 `、，:｜` 等切分，**子串不算数**）。三条示例：

- `GBZ26583-2011-p20-c01`（subtype「猝倒病、立枯病」）→ 填 `"猝倒病"` 或 `"立枯病"` 合法；
  `"辣椒猝倒病"` 不合法
- `GBZ26583-2011-p6-b6.2`（正文卡，subtype 是章节号 `6.2`）→ 病虫害名只在 quote 里，
  填 `"疫病"` 合法
- `GBZ26583-2011-p20-c03`（subtype「早疫病、晚疫病」）→ 填 `"疫病"` **不合法**：
  既是子串、又是两种不同的病，挂上去＝给错药

**落库四条约定（spec §3.1/§3.2）**：

1. **一条一行**：同一病虫害有多个来源 = 多行；同一来源写了多个病虫害 = 拆成多行
2. **挂卡规则**：该病虫害有附录B 方案卡 → 挂方案卡；只在正文提及（无方案）→ 挂正文卡
   （本批 5 条：疫病挂正文卡 `b6.2`，其余 4 条挂方案卡）
3. **能挂都挂**：一个病虫害有多张卡时（种蝇有 c03/c05/c06 三张），症状贴近哪张挂哪张，
   分不清就多行都挂，带虫态就挂对应虫态卡——本批 5 条各只有一张对应卡，暂不涉及
4. **找不到出处的目标不硬凑**：哪一条确实找不到可引用来源，把它记进
   `data/processed/症状挂卡清单.md` 的「未覆盖清单」一节，并在提交信息里说明

**快照文件命名**：`data/external/symptoms_raw/<symptom_id>.md`

- [ ] **Step 3: 写清单脚本**

（脚本末尾带「未覆盖清单」小节：把本批计划覆盖但没落库的目标列出来，供用户判断）

Create `scripts/make_symptoms_checklist.py`：

```python
# -*- coding: utf-8 -*-
"""把症状覆盖层导成人读的复核清单（与口语映射表的 build_colloquial_map.py 同一套路）。

用法：
    python scripts/make_symptoms_checklist.py
输出：
    data/processed/症状挂卡清单.md
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from validate_symptoms import collect  # noqa: E402  复用同一套加载口径

OUT = os.path.join(ROOT, "data", "processed", "症状挂卡清单.md")


def main():
    rows, errs, stats = collect()
    lines = ["# 症状挂卡清单（人工复核用，脚本现算）", "",
             f"- 条目数：**{stats['n']}**",
             f"- 按作物：{stats['by_crop']}",
             f"- 按来源：{stats['by_kind']}",
             f"- 校验：{'✗ ' + str(len(errs)) + ' 个错误' if errs else '✓ 通过'}", "",
             "> 逐条核对：症状原文是否与目标病虫害相符、出处是否可回查、"
             "是否与评估集问句同源（12396）。核对后把 `needs_human_review` 改成 false。", "",
             "| # | 作物 | 症状挂到 | 病虫害 | 症状原文 | 出处 | 复核 |",
             "| ---: | --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(rows, 1):
        src = r.get("source") or {}
        ref = " ".join(x for x in (src.get("title"), src.get("publisher")) if x)
        txt = (r.get("symptom_text") or "").replace("\n", " ")
        flag = "⬜ 待复核" if r.get("needs_human_review") else "✅ 已复核"
        lines.append(f"| {i} | {r.get('crop')} | `{r.get('card_id')}` | {r.get('pest')} | "
                     f"{txt[:60]} | {ref} | {flag} |")
    if errs:
        lines += ["", "## 校验错误", ""] + [f"- {e}" for e in errs]
    lines += ["", "## 未覆盖清单", "",
              "> 本批计划要覆盖、但没落库的目标（找不到可引用出处就不硬凑，逐条写明原因）。", ""]
    covered = {r.get("card_id") for r in rows}
    plan_targets = [
        ("GBZ26583-2011-p6-b6.2", "辣椒 疫病（正文卡）"),
        ("GBZ26578-2011-p20-c09", "大蒜 豌豆潜叶蝇"),
        ("GBZ26578-2011-p20-c10", "大蒜 蓟马"),
        ("GBZ26583-2011-p21-c05", "辣椒 茶黄螨"),
        ("GBZ26583-2011-p20-c01", "辣椒 猝倒病、立枯病"),
    ]
    miss = [(cid, name) for cid, name in plan_targets if cid not in covered]
    lines += ([f"- [ ] `{cid}` {name}" for cid, name in miss] if miss
              else ["- （无，5 条全部覆盖）"])
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"→ {OUT}（{len(rows)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑校验与清单**

Run:
```bash
python scripts/validate_symptoms.py && python scripts/make_symptoms_checklist.py
```
Expected: `✓ 校验通过`，条目 5 条；清单文件生成

- [ ] **Step 5: 请用户逐条复核**

把 `data/processed/症状挂卡清单.md` 给用户过一遍（每条：症状与病是否相符、出处是否可回查、是否与 12396 同源）。
复核通过后把对应条目的 `needs_human_review` 改成 `false`，重跑 Step 4。

- [ ] **Step 6: 跑全量自检**

Run: `python -m pytest -q`
Expected: 全绿（`tests/test_symptoms.py` 的真实文件用例由 skip 转为实跑）

- [ ] **Step 7: 提交**

```bash
git add data/processed/symptoms.jsonl data/processed/症状挂卡清单.md \
        data/external/symptoms_raw scripts/make_symptoms_checklist.py
git commit -m "data: 症状覆盖层试点 5 条（辣椒疫病/大蒜豌豆潜叶蝇/大蒜蓟马/辣椒茶黄螨/辣椒猝倒病）"
```

---

## Task 4: 切分层合并症状

**Files:**
- Modify: `scripts/build_chunks.py`（新增 `SYMPTOMS` 常量、`load_symptoms()`、`build_chunk()` 扩参、`main()` 挂空校验）
- Modify: `tests/test_symptoms.py`（追加切分合并用例）

**Interfaces:**
- Consumes: `data/processed/symptoms.jsonl`（Task 3）
- Produces: chunk 新增字段 `sym_text: str`、`symptoms: list[dict]`、`meta.has_symptoms: bool`

- [ ] **Step 1: 追加失败测试**

追加到 `tests/test_symptoms.py`：

```python
sys.path.insert(0, str(ROOT / "src" / "index"))
import build_chunks as B  # noqa: E402  被测对象：切分合并逻辑


def _card():
    """一张最小可用的附录B 卡。"""
    return {
        "card_id": "CARD-A", "crop": "大蒜", "std_no": "GB/Z 00000-0000",
        "subtype": "豌豆潜叶蝇", "title": "大蒜豌豆潜叶蝇防治方案",
        "knowledge_type": "病虫害防治", "type_group": "病虫草害", "pest_kind": "虫害",
        "growth_stage": None, "trigger": "始见虫道", "precautions": None,
        "is_high_risk": True, "needs_verification_hint": True,
        "chemicals": [{"product": "10%落灭津SC", "dose": "30 g/667m²", "dilution": None,
                       "method": "对水喷雾", "interval_days": "7 d",
                       "pre_harvest_interval": None, "max_uses_per_season": None,
                       "companion_control": None, "raw": "方案一:10%落灭津SC 30 g/667m²"}],
        "measures": [], "source": {"section": "附录B 表B.1", "page": 20,
                                   "channel": "vision", "quote": "豌豆潜叶蝇｜始见虫道"},
    }


def _sym():
    return {"symptom_id": "sym-x", "card_id": "CARD-A", "crop": "大蒜", "pest": "豌豆潜叶蝇",
            "symptom_text": "叶片表皮下潜食叶肉，形成白色弯曲的蛇形虫道",
            "part": ["叶片"],
            "source": {"kind": "journal", "title": "测试文献", "publisher": "《测试》",
                       "url": None, "accessed": "2026-09-28", "quote": "原文"}}


def test_症状进sym_text不进embed和bm25():
    """症状文本只进 sym_text——embed_text/bm25_text 必须保持卡本体口径，
    否则会污染检索层从 bm25_text 现算的实体表（spec §4.1）。"""
    ch = B.build_chunk(_card(), [_sym()])
    assert "蛇形虫道" in ch["sym_text"]
    assert "蛇形虫道" not in ch["embed_text"]
    assert "蛇形虫道" not in ch["bm25_text"]
    assert ch["meta"]["has_symptoms"] is True
    assert ch["symptoms"][0]["symptom_id"] == "sym-x"


def test_无症状的卡字段为空():
    ch = B.build_chunk(_card(), [])
    assert ch["sym_text"] == ""
    assert ch["symptoms"] == []
    assert ch["meta"]["has_symptoms"] is False


def test_挂空卡直接报错():
    """症状挂到不存在的卡 → 必须拦下，不能静默丢。"""
    codes = B.check_orphans({"CARD-NOPE"}, {"CARD-A"})
    assert codes == ["CARD-NOPE"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_symptoms.py -q`
Expected: FAIL —— `build_chunk() takes 1 positional argument but 2 were given` / `check_orphans` 未定义

- [ ] **Step 3: 改 `build_chunks.py`**

顶部常量区（`CARDS = [...]` 之后）加：

```python
# 症状覆盖层：外部来源的症状条目，按 card_id 挂到卡上（卡本体不动）
SYMPTOMS = os.path.join(P, "symptoms.jsonl")
```

新增加载函数（放在 `load_cards()` 之后）：

```python
# 【症状覆盖层】读 data/processed/symptoms.jsonl → {card_id: [条目, ...]}
def load_symptoms(path=None):
    """读症状覆盖层，按目标卡分组。文件不存在返回空 dict（覆盖层是可选的）。"""
    path = path or SYMPTOMS
    if not os.path.exists(path):
        return {}
    by_card = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            s = json.loads(line)
            by_card.setdefault(s["card_id"], []).append(s)
    return by_card


# 症状挂到不存在的卡（或该卡不进索引）→ 返回孤儿 card_id 列表
def check_orphans(symptom_card_ids, chunk_card_ids):
    """挂空的症状会静默丢失，必须在切分层拦下。两个参数都是 card_id 集合。"""
    return sorted(set(symptom_card_ids) - set(chunk_card_ids))
```

改 `build_chunk(c)`：

```python
# 把一张卡组装成一个 chunk：两个检索字段 + 症状覆盖层 + 过滤用 meta + 原始 card
def build_chunk(c, symptoms=None):
    """symptoms：挂到本卡的覆盖层条目列表（可空）。

    ⚠️ 症状只进 `sym_text`（编码时才与两路拼在一起），**不写进 embed_text /
    bm25_text**——检索层会用 bm25_text 现算「语料里提到过」的实体表，
    混入外部来源会把兜底档位改掉（spec §4.1）。
    """
    symptoms = symptoms or []
    sym_text = " ".join((s.get("symptom_text") or "").strip() for s in symptoms).strip()
    return {
        "chunk_id": c["card_id"],
        "card_id": c["card_id"],
        # ---- 两路检索字段（已做全角->半角归一）----
        "embed_text": halfwidth(embed_text(c)),
        "bm25_text": halfwidth(bm25_text(c)),
        # ---- 症状覆盖层（外部来源，转半角后落盘；symptoms 保留原文供答案层引）----
        "sym_text": halfwidth(sym_text),
        "symptoms": symptoms,
        # ---- 过滤/排序/展示用元数据（不参与相似度）----
        "meta": {
            "crop": c["crop"],
            "std_no": c["std_no"],
            "knowledge_type": c["knowledge_type"],
            "type_group": c["type_group"],
            "pest_kind": c.get("pest_kind"),
            "subtype": c["subtype"],
            "growth_stage": c.get("growth_stage"),
            "is_high_risk": c["is_high_risk"],
            "needs_verification_hint": c["needs_verification_hint"],
            "source_section": c["source"]["section"],
            "source_page": c["source"]["page"],
            "channel": c["source"]["channel"],
            "has_symptoms": bool(symptoms),
        },
        # ---- 生成阶段要用的完整卡（自带，省一次 join）----
        "card": c,
    }
```

改 `main()` 的组装段（原 `chunks = [build_chunk(c) for c in cards]`）：

```python
    # 症状覆盖层挂卡；挂空 = 静默丢失，直接拦下
    syms = load_symptoms()
    orphan = check_orphans(syms, {c["card_id"] for c in cards})
    if orphan:
        print(f"✗ 症状挂到了不存在的卡（或该卡不进索引）: {orphan}")
        sys.exit(1)
    chunks = [build_chunk(c, syms.get(c["card_id"])) for c in cards]
```

统计段加一行（原 `print(f"  高危卡: ...")` 之后）：

```python
    print(f"  带症状的 chunk: {sum(1 for ch in chunks if ch['meta']['has_symptoms'])}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_symptoms.py -q`
Expected: PASS

- [ ] **Step 5: 重新切分并核对数字**

Run:
```bash
python src/index/build_chunks.py
```
Expected: 输出 `203 个 chunk`（**数字不变**）、`带症状的 chunk: 5`、无孤儿报错

- [ ] **Step 6: 提交**

```bash
git add scripts/build_chunks.py tests/test_symptoms.py data/chunks/chunks.jsonl
git commit -m "feat: 切分层合并症状覆盖层（sym_text 单开一路，不动 embed/bm25）+ 挂空硬校验"
```

---

## Task 5: 向量化与建库

**Files:**
- Modify: `src/index/embed_chunks.py`（编码输入拼接 `sym_text`；`embedding_meta` 记口径）
- Modify: `src/index/build_milvus_index.py`（schema + `record()`）
- 产物：`data/chunks/chunks_embedded.jsonl`、`data/index/milvus_rag.db`

**Interfaces:**
- Consumes: chunk 的 `sym_text`（Task 4）
- Produces: Milvus collection 新增 `sym_text`(VARCHAR 2048) 与 `has_symptoms`(BOOL) 两个字段

- [ ] **Step 1: 改 `embed_chunks.py` 的编码输入**

把 `main()` 里这两行：

```python
    # 稠密路输入 = embed_text（只含语义线索）
    dense_in = [c["embed_text"] for c in chunks]
    # 稀疏路输入 = bm25_text（含数值）；字段缺失时退回 embed_text
    sparse_in = [c.get(SPARSE_FIELD) or c["embed_text"] for c in chunks]
```

改为：

```python
    # 症状覆盖层：只在**编码这一刻**与两路基底拼在一起，chunk 文件里两个字段保持卡本体口径
    def _with_sym(c, base):
        sym = (c.get("sym_text") or "").strip()
        return f"{base} {sym}".strip() if sym else base

    # 稠密路输入 = embed_text（只含语义线索）+ 症状文本
    dense_in = [_with_sym(c, c["embed_text"]) for c in chunks]
    # 稀疏路输入 = bm25_text（含数值）+ 症状文本；字段缺失时退回 embed_text
    sparse_in = [_with_sym(c, c.get(SPARSE_FIELD) or c["embed_text"]) for c in chunks]
```

`embedding_meta` 里补一行口径：

```python
            rec["embedding_meta"] = {
                "model": "BAAI/bge-m3",
                "dense_dim": dim,
                "dense_source_field": "embed_text + sym_text",
                "sparse_source_field": SPARSE_FIELD + " + sym_text",
                "sparse_type": "lexical_weights",
                "symptom_source_field": "sym_text",
            }
```

文件头部的双路表也补一行说明（`| 稠密 | embed_text + sym_text | ... |`）。

- [ ] **Step 2: 改 `build_milvus_index.py`**

常量区（`INT_FIELDS` 之后）加：

```python
# 症状覆盖层字段：存一份供调试/展示，**不建倒排索引**（没有任何路由过滤用它）
RAW_FIELDS = [("sym_text", 2048)]
RAW_BOOL_FIELDS = ["has_symptoms"]
```

`build_schema()` 末尾（`bm25_text` 那两行之后）加：

```python
    # 症状覆盖层：只存不索引
    for name, ln in RAW_FIELDS:
        s.add_field(name, DataType.VARCHAR, max_length=ln)
    for name in RAW_BOOL_FIELDS:
        s.add_field(name, DataType.BOOL)
```

`record()` 的返回 dict 末尾加：

```python
        # 症状覆盖层（只存不索引）
        "sym_text": (r.get("sym_text") or "")[:2048],
        "has_symptoms": bool(m.get("has_symptoms")),
```

⚠️ `build_index()` **不动**——`RAW_FIELDS` 不进倒排循环。

- [ ] **Step 3: 向量化**

Run: `python src/index/embed_chunks.py`
Expected: `稠密维度 1024 ｜ 稀疏非空 203/203`

- [ ] **Step 4: 重建索引**

Run: `python src/index/build_milvus_index.py --recreate`
Expected: 实体数 `203`（**不变**）；生效索引清单里**没有** `sym_text` / `has_symptoms`

- [ ] **Step 5: 抽查症状确实进了索引**

Run:
```bash
python -c "
from pymilvus import MilvusClient
c = MilvusClient('data/index/milvus_rag.db')
r = c.query('agri_knowledge', filter='has_symptoms == true',
            output_fields=['chunk_id','sym_text'], limit=10)
print(len(r)); [print(x['chunk_id'], x['sym_text'][:40]) for x in r]
"
```
Expected: 5 条，`sym_text` 非空

- [ ] **Step 6: 提交**

```bash
git add src/index/embed_chunks.py src/index/build_milvus_index.py data/chunks/chunks_embedded.jsonl
git commit -m "feat: 编码期合并 sym_text + Milvus 存症状字段（不建倒排）"
```

（`data/chunks/chunks_embedded.jsonl` **在版本控制内**，向量变了必须一起提交；
`data/index/milvus_rag.db` 被 gitignore，不进提交。）

---

## Task 6: 生成层【症状对照】段

**Files:**
- Modify: `src/generate/answer.py`
- Modify: `tests/test_generate_template.py`（追加 5 个用例）

**Interfaces:**
- Consumes: chunk 的 `symptoms` 列表（Task 4）
- Produces:
  - `_SYMPTOMS: dict[str, list]`（chunk_id → 症状条目）
  - `SYM_HEAD: str` 对照口吻结论句
  - `_is_symptom_query(q) -> bool`
  - `render_symptoms(symptoms) -> str`
  - `render_high_risk(cards, crop, symptoms=None)` / `render_high_text(cards, crop, symptoms=None)` 扩参

- [ ] **Step 1: 写 5 个失败测试**

追加到 `tests/test_generate_template.py`：

```python
# ---------------------------------------------------------------- 症状覆盖层
@pytest.fixture(scope="module")
def sym_rows(rows):
    """挂了症状的 chunk；一条都没有时跳过（覆盖层未生成）。"""
    got = [r for r in rows if r.get("symptoms")]
    if not got:
        pytest.skip("症状覆盖层未生成（先跑 scripts/build_chunks.py）")
    return got


def test_有症状卡渲染症状段(sym_rows):
    """症状问法 + top-1 有症状 → 必须出【症状对照】，且排在用药方案之前。"""
    r = sym_rows[0]
    c = r["card"]
    out = A.compose("叶子上有白色的虫道，是什么虫", c["crop"], [{"chunk_id": r["chunk_id"]}], dry_run=True)
    assert "【症状对照】" in out, out
    if "【用药方案】" in out:
        assert out.index("【症状对照】") < out.index("【用药方案】"), "症状段必须在用药方案之前"


def test_无症状卡不渲染症状段(rows):
    """没挂症状的卡，症状问法也不该凭空冒出【症状对照】。"""
    bare = [r for r in rows if not r.get("symptoms")]
    assert bare, "语料里没有无症状的卡，本用例失去意义"
    r = bare[0]
    out = A.compose("叶子上有白色的虫道，是什么虫", r["card"]["crop"],
                    [{"chunk_id": r["chunk_id"]}], dry_run=True)
    assert "【症状对照】" not in out, out


def test_病名问法答案零变化(sym_rows):
    """点了病名的问法，答案与加症状前一致：不出现症状段、结论句仍是原措辞。

    这条最关键——它直接验证「全量回退」风险（spec §11.2）。
    """
    checked = 0
    for r in sym_rows:
        c = r["card"]
        # 只挑能确定抽出实体的问法（subtype 里第一个病虫害名）
        pest = (c.get("subtype") or "").split("、")[0].strip()
        if len(pest) < 2:
            continue
        out = A.compose(f"{c['crop']}{pest}打什么药", c["crop"],
                        [{"chunk_id": r["chunk_id"]}], dry_run=True)
        assert "【症状对照】" not in out, f"{r['chunk_id']} 病名问法冒出了症状段"
        assert not out.startswith(A.SYM_HEAD), f"{r['chunk_id']} 病名问法用了对照口吻"
        checked += 1
    assert checked, "没有可测的病名问法，本用例失去意义"


def test_症状段逐字等于原文(sym_rows):
    """症状文本不过模型、不改写——答案里必须逐字等于 symptoms.jsonl 的内容。"""
    for r in sym_rows:
        out = A.render_symptoms(r["symptoms"])
        for s in r["symptoms"]:
            assert s["symptom_text"] in out, f"{r['chunk_id']} 症状段没有逐字照抄"


def test_二级来源标注正确(sym_rows):
    """症状来源必须标出来，且明示非国标（与卡片来源分开）。"""
    for r in sym_rows:
        out = A.render_symptoms(r["symptoms"])
        for s in r["symptoms"]:
            src = s.get("source") or {}
            assert (src.get("publisher") or src.get("title")) in out, "来源没标"
        assert "非国标" in out
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_generate_template.py -q`
Expected: FAIL —— `AttributeError: module 'answer' has no attribute 'SYM_HEAD'`

- [ ] **Step 3: 改 `answer.py`**

导入区之后（`_CARDS` 建立处）改为同时建症状表：

```python
# chunk_id -> 完整卡片（模板渲染高危字段要用结构化 chemicals[]，
# Milvus 里只存了拼平的 bm25_text，取不回字段边界）
# chunk_id -> 症状覆盖层条目（外部来源，答案里要与卡片来源分开标）
_CARDS = {}
_SYMPTOMS = {}
for _line in open(CHUNKS, encoding="utf-8"):
    _c = json.loads(_line)
    _CARDS[_c["chunk_id"]] = _c["card"]
    _SYMPTOMS[_c["chunk_id"]] = _c.get("symptoms") or []
```

话术常量区（`VERIFY_HINT` 之后）加：

```python
# 症状问法的结论句：**不做诊断，只做对照**——「与你描述相符」而不是「这是X病」。
# 农户描述可能是疫病也可能是别的，一旦用肯定口吻下诊断，后面的药方就建立在一个
# 未经确认的判断上；对照口吻把最后一道校验交回农户（spec §5.3）。
SYM_HEAD = "【结论】你描述的症状与下面这张卡记载的相符，请对照确认："
# 症状来源的类型标签（外部来源，与卡片的一级国标来源分开标）
SYM_KIND = {"journal": "期刊文献", "gov_site": "政府/农科院网站",
            "book": "图书", "hotline": "农技热线"}
```

新增三个函数（放在 `render_high_risk` 之前）：

```python
# 症状问法判据：问句里没有任何病虫害实体（农户在描述现象、没说出病名）
def _is_symptom_query(q):
    """⚠️ 延迟导入检索层：生成层测试不该被 Milvus/BGE 依赖拖下水。
    导入失败保守返回 False——宁可少渲染一段症状，也不能让生成层挂掉。"""
    if not q:
        return False
    try:
        from search import extract_entities
        return not extract_entities(q)
    except Exception:
        return False


# 【症状对照段】逐字引用外部来源的症状描述，**不过模型**
def render_symptoms(symptoms):
    """症状条目 → 【症状对照】段。数值/文本一字不改，来源逐条标。"""
    out = ["【症状对照】"]
    for s in symptoms:
        out.append(f"  {s['symptom_text']}")
        src = s.get("source") or {}
        ref = "　".join(x for x in (src.get("title"), src.get("publisher")) if x)
        kind = SYM_KIND.get(src.get("kind"), "外部来源")
        out.append(f"  —— 来源：{ref}（{kind}，非国标）")
    return "\n".join(out)


# 取出 top-1 卡的症状条目（只在症状问法下取）
def _top_symptoms(query, results):
    """返回 top-1 卡的症状列表；非症状问法 / 无症状 → 空列表。"""
    if not results or not _is_symptom_query(query):
        return []
    return _SYMPTOMS.get(results[0]["chunk_id"]) or []
```

`render_high_risk` 扩参并改结论句：

```python
def render_high_risk(cards, crop, symptoms=None):
```

（函数签名改这一行，docstring 补一句"症状问法时用对照口吻并插入【症状对照】段"。）

函数体里结论句那段：

```python
    out = []
    if symptoms:
        # 症状问法：不下诊断，先让农户对照确认，再看药（spec §5.2）
        out.append(SYM_HEAD)
        out.append(render_symptoms(symptoms))
    elif len(stages) > 1:
        out.append(f"【结论】{crop}的{sub}：{'、'.join(stages)}均可施药，方案如下")
    elif stages:
        out.append(f"【结论】{crop}的{sub}，{stages[0]}时可用下列药剂")
    else:
        out.append(f"【结论】{crop}的{sub}，可用下列药剂")
```

`render_high_text` 同样扩参、在 head 之后插入：

```python
def render_high_text(cards, crop, symptoms=None):
```

```python
    first = cards[0]
    topic = first.get("knowledge_type") or "要求"
    out = []
    if symptoms:
        out.append(SYM_HEAD)
        out.append(render_symptoms(symptoms))
    else:
        head = f"【结论】{crop}的{topic}要求如下"
        if len(cards) == 1 and first["source"].get("section"):
            head += f"（第 {first['source']['section']} 条）"
        out.append(head)
```

⚠️ `render_high_risk` 里那句 `return render_high_text(cards, crop)` 要改成
`return render_high_text(cards, crop, symptoms=symptoms)`。

`compose()` 里插入症状块（`groups` 组装前后的位置见下）：

```python
    # 症状问法（问句里没有病虫害实体）→ 只给 top-1 卡渲染【症状对照】，
    # 且**全篇只渲染一次**；若 top-1 落不到任何高危组（普通卡/正文卡），
    # 整块提到所有高危块之前，保证它永远排在【用药方案】之前。
    # （2026-09-28 用户裁决，修计划初稿的「无条件叠加」——那会让症状段重复出现，
    #   或落到【用药方案】之后。原写法用 card_id 比 chunk_id 也比错了命名空间。）
    top_syms = _top_symptoms(query, results)
    top_card = _CARDS.get(top_id)      # 用对象同一性判组，不依赖 card_id ≡ chunk_id

    groups = {}
    for c in high:
        groups.setdefault((c["crop"], c["subtype"]), []).append(c)

    blocks = []
    sym_attached = False
    for g in groups.values():
        g_syms = top_syms if (top_syms and any(c is top_card for c in g)) else None
        if g_syms:
            sym_attached = True
        blocks.append(render_high_risk(g, g[0]["crop"], symptoms=g_syms))
    # top-1 不在任何高危组（普通卡/正文卡）→ 症状块独立提前，必须在【用药方案】之前
    if top_syms and not sym_attached:
        blocks.insert(0, SYM_HEAD + "\n" + render_symptoms(top_syms))
```

（`top_id` 在卡片取回之后定义：`top_id = results[0]["chunk_id"] if results else None`。）

普通卡分支**不再挂症状块**（症状块已在上面统一处理，挂这里就会重复）：

```python
    if normal:
        blocks.append("【相关农事操作】\n" + call_model(query, normal, dry_run))
```

**多卡混合路径的回归测试（必做）**——初版新测试都只喂单条结果，所以上面两个缺陷
一路绿灯溜过去了。追加两条，用真实 chunks 数据：

```python
def test_多卡混合时症状段只出现一次(sym_rows, rows):
    """top-1 是高危症状卡、结果里还夹着普通卡时，【症状对照】必须只有一份。"""
    r = next(x for x in sym_rows if x["card"]["needs_verification_hint"])
    bare = [x for x in rows if not x["card"]["needs_verification_hint"]]
    assert bare, "语料里没有普通卡，本用例失去意义"
    out = A.compose("叶子上有白色的虫道，是什么虫", r["card"]["crop"],
                    [{"chunk_id": r["chunk_id"]}, {"chunk_id": bare[0]["chunk_id"]}],
                    dry_run=True)
    assert out.count("【症状对照】") == 1, out
    assert out.count("【结论】") == 1, out


def test_症状段永远排在任何用药方案之前(sym_rows, rows):
    """top-1 是普通症状卡、结果里另有带药高危卡时，症状段仍必须在【用药方案】之前。"""
    r = next(x for x in sym_rows if not x["card"]["needs_verification_hint"])
    chem = [x for x in rows if x["card"].get("chemicals")]
    assert chem, "语料里没有带药卡，本用例失去意义"
    out = A.compose("叶子上有白色的虫道，是什么虫", r["card"]["crop"],
                    [{"chunk_id": r["chunk_id"]},
                     {"chunk_id": next(c for c in chem if c["card"]["crop"] == r["card"]["crop"])["chunk_id"]}],
                    dry_run=True)
    assert "【用药方案】" in out and "【症状对照】" in out, out
    assert out.index("【症状对照】") < out.index("【用药方案】"), out
```

（`rows` 就是测试文件里既有的模块级 fixture，直接当参数取用；两条用例都用真实 chunks 数据。）

核实提示补一句（`if high:` 的两个分支里，`VERIFY_HINT` 之前）：

```python
                    f"{VERIFY_HINT}。症状描述来自外部文献，是否为你田里的情况请对照确认。"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_generate_template.py -q`
Expected: PASS（原 10 条 + 新 5 条）

- [ ] **Step 5: 端到端抽一条**

Run: `python src/generate/answer.py --query "蒜苗叶子上有一条条白色弯曲的斑道，是什么危害？" --crop 大蒜 --dry-run`
Expected: 出现【症状对照】+ 豌豆潜叶蝇来源；用药方案数值逐字来自卡片

- [ ] **Step 6: 提交**

```bash
git add src/generate/answer.py tests/test_generate_template.py
git commit -m "feat: 生成层症状对照段（逐字不过模型，对照口吻不下诊断）"
```

---

## Task 7: 评估——快照、数据集、全量重跑

**Files:**
- Copy: `eval/report.json` → `eval/report_before_symptoms.json`
- Modify: `eval/datasets/collect.jsonl`（加回 4 条 + 第 18 条改写）

**Interfaces:**
- Consumes: 全部上游改动
- Produces: 改前/改后两份报告，用于对比三项指标

- [ ] **Step 1: 存改前快照**

```bash
cp eval/report.json eval/report_before_symptoms.json
```

- [ ] **Step 2: 加回 4 条**

追加到 `eval/datasets/collect.jsonl`（一行一条，字段与既有条目一致）：

```json
{"query": "蒜苗叶子上有一条条白色弯曲的斑道，是什么危害？", "crop": "大蒜", "source": "collected", "origin": "12396", "note": "症状描述豌豆潜叶蝇，未提虫名｜症状挂 c09（豌豆潜叶蝇），补 symptoms 后按正例加回", "expect_cards": ["GBZ26578-2011-p20-c09"], "expect_fallback": false, "expect_noplan": false, "is_high_risk": false, "annotated_by": "人工复核（用户 2026-09-27）"}
{"query": "大蒜叶上有小白点密密麻麻，心叶扭卷，是什么虫害？", "crop": "大蒜", "source": "collected", "origin": "12396", "note": "症状描述蓟马，未提虫名｜症状挂 c10（蓟马），补 symptoms 后按正例加回", "expect_cards": ["GBZ26578-2011-p20-c10"], "expect_fallback": false, "expect_noplan": false, "is_high_risk": false, "annotated_by": "人工复核（用户 2026-09-27）"}
{"query": "辣椒新叶变小发硬、叶背有锈色细毛，落花落果严重，怎么回事？", "crop": "辣椒", "source": "collected", "origin": "12396", "note": "症状描述茶黄螨，未提虫名｜症状挂 c05（茶黄螨），补 symptoms 后按正例加回", "expect_cards": ["GBZ26583-2011-p21-c05"], "expect_fallback": false, "expect_noplan": false, "is_high_risk": false, "annotated_by": "人工复核（用户 2026-09-27）"}
{"query": "辣椒苗床里小苗成片倒伏死苗，怎么办？", "crop": "辣椒", "source": "collected", "origin": "12396", "note": "苗期症状描述，对应猝倒病/立枯病，未提病名｜症状挂 c01，补 symptoms 后按正例加回", "expect_cards": ["GBZ26583-2011-p20-c01"], "expect_fallback": false, "expect_noplan": false, "is_high_risk": true, "annotated_by": "人工复核（用户 2026-09-27）"}
```

- [ ] **Step 3: 改写第 18 条（辣椒疫病）**

把 `eval/datasets/collect.jsonl` 里 `query` 为「辣椒茎基部或枝条处出现黑褐色病斑，整株萎蔫落叶，几天就死了，是什么病？」
那一行的 `note` 与 `expect_noplan` 改为：

```json
"note": "复核需改（2026-09-27 二次复核）｜用户判定：症状（茎基黑褐病斑+全株萎蔫+几天速死）指向**疫病**、非叶部病害，辣椒疫病在语料中只在 6.2 列名无方案 → **目标行为＝b6.2 正文 + 不带别家药方**。symptoms 字段已落地（2026-09-28），按此口径回改验收：expect_noplan=true",
"expect_noplan": true,
```

（`expect_cards` 保持 `["GBZ26583-2011-p6-b6.2"]`，其余字段不动。）

- [ ] **Step 4: 确认检索层确实没被改（spec §4.3 的硬约束）**

Run: `git diff --stat HEAD -- src/retrieve/search.py`
Expected: **无输出**。有输出说明检索判定逻辑被动了，先回退再往下走——
本批的所有效果都必须来自「症状文本进了向量」，而不是来自改了判定。

- [ ] **Step 5: 全量重跑**

Run: `python eval/run_eval.py`
Expected: 运行完成（GPU 约 1–2 分钟）；输出报告

- [ ] **Step 6: 对比三项指标**

Run:
```bash
python -c "
import json
b = json.load(open('eval/report_before_symptoms.json', encoding='utf-8'))['buckets']
a = json.load(open('eval/report.json', encoding='utf-8'))['buckets']
for k in sorted(set(a) | set(b)):
    x, y = b.get(k, {}), a.get(k, {})
    print(k, '  改前 recall/mrr/false+ :', x.get('recall'), x.get('false_fb_positive', x.get('false_fb')), '->',
          y.get('recall'), y.get('false_fb_positive'))
"
```
Expected: **auto 两个桶的 recall 不回退**；collected 普通/高危 recall 上升

- [ ] **Step 7: 逐条看四个新加回的用例**

Run: `python eval/run_eval.py --dataset eval/datasets/collect.jsonl`
Expected: 记录 #5/#6/#8/#9 与辣椒疫病条（第 18 条）的实际状态

**若 c03（早疫/晚疫方案卡）仍压在 b6.2 之上** → 不要改标注蒙混，按 spec §13.3 记录到
`eval/失败复核清单.md` 的"待修项"，处置预案＝症状指向更精确的卡，或给 c03 负向权重。

- [ ] **Step 8: 提交**

```bash
git add eval/datasets/collect.jsonl eval/report.json eval/report_before_symptoms.json
git commit -m "eval: 症状覆盖层落地后重跑（加回 4 条症状用例 + 辣椒疫病条回改 expect_noplan）"
```

---

## Task 8: 文档同步与收口

**Files:**
- Modify: `CLAUDE.md`（六·3 口径改写、六·2 元数据表、五节目录、二节进度）
- Modify: `README.md`、`docs/01-需求分析.md`、`docs/06-全链路代码索引.md`
- Modify: `tests/test_docs_consistency.py`（若锁了新数字）

**Interfaces:**
- Consumes: 前面所有任务的产物
- Produces: 文档与实现一致，`python -m pytest` 全绿

- [ ] **Step 1: 改写 CLAUDE.md 六·3 的症状口径**

把「症状描述型问法（2026-09-27 定稿口径，取代原「建症状映射表」计划）……」那一段，
改为记录**新的**定稿口径（**保留推翻痕迹**，不能悄悄删）：

```markdown
- **症状描述型问法（2026-09-28 改口径，取代 2026-09-27 的「不造症状映射、等扩语料」）**：
  农户描述症状时，**靠语料扩层 + 语义匹配**，不靠映射表——新增
  `data/processed/symptoms.jsonl` 覆盖层（外部二级来源，逐条带字段级来源），
  切分时并入 `sym_text` 喂编码；卡片本体一个字不动，`symptoms` 字段仍为 null。
  ⚠️ 症状**绝不进** `configs/colloquial_map.json`（那张表只做词形替换，写症状＝造诊断）。
  不改口径：症状问句仍不允许返回**其他**病虫害的附录B 用药方案。
  spec 见 `docs/superpowers/specs/2026-09-28-symptoms-field-design.md`
```

- [ ] **Step 2: 补齐其余文档**

- `CLAUDE.md` 五节：`data/processed/` 清单加 `symptoms.jsonl` 与 `症状挂卡清单.md`；
  `data/external/` 加 `symptoms_raw/` 快照区；`tests/` 8 → 9 个文件（加 `test_symptoms.py`）；
  `scripts/` 加 `validate_symptoms.py` / `make_symptoms_checklist.py`
- `CLAUDE.md` 二节：进度勾选加一条症状覆盖层（试点 5 条）
- `README.md`：同口径改写（第 43 行那句"造映射等于引入语料外知识"要改），产物清单同步
- `docs/01-需求分析.md`：凡有症状口径表述处按新口径改写
- `docs/06-全链路代码索引.md`：阶段③ 补 symptoms.jsonl 与校验脚本

- [ ] **Step 3: 全量自检**

Run: `python -m pytest -q`
Expected: 全绿

- [ ] **Step 4: 提交**

```bash
git add CLAUDE.md README.md docs/ tests/test_docs_consistency.py
git commit -m "docs: 症状覆盖层口径回写（六·3 改口径并留推翻痕迹）+ 产物与自检清点同步"
```

---

## 铺开（试点通过后再做，不在本计划内）

试点验证有效后，按 spec §6 把症状覆盖层铺到全部病虫害（25 张附录B 卡 + 正文提及的），
另开一份计划。铺开前先确认：**auto 集三桶无回退**、**五个症状用例达标**、
**`eval/report.json` 的误兜底/漏兜底仍为 0**。
