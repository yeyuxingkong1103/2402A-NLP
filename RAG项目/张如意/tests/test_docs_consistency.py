# -*- coding: utf-8 -*-
"""文档一致性单测：锁定 claude.md / docs/01 / docs/04 / README / sources.md 与语料之间的一致性。

这些用例防的是「文档写着小麦、语料是蔬菜」这类漂移（本项目真发生过）。
其中语料事实（份数/页数/字形码页数/本期页数/关键词命中）由 PDF 现算，不依赖人工记录。
"""
import ast
import json
import re
from pathlib import Path

import fitz
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLAUDE = (ROOT / "claude.md").read_text(encoding="utf-8")
DOC01 = (ROOT / "docs/01-需求分析.md").read_text(encoding="utf-8")
DOC04 = (ROOT / "docs/04-语料清单-农业国标PDF.md").read_text(encoding="utf-8")
SOURCES = (ROOT / "data/sources.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
PROMPT = (ROOT / "configs/prompts/parse_pdf_to_knowledge.md").read_text(encoding="utf-8")

CROPS = ("黄瓜", "辣椒", "大蒜")
STD_NOS = ("GB/Z 26581-2011", "GB/Z 26583-2011", "GB/Z 26578-2011")
SCOPE_STD = ("26578-2011", "26581-2011", "26583-2011")     # 本期 3 份（文件名片段）
KEYWORDS = ("农药混用", "限用", "主要有害生物防治方案", "安全间隔期", "最多使用次数")
# 视觉通道产物与解析卡片：缺口复核结论必须能从这两处现算（不是人工记录）
VISION_A = ROOT / "data/processed/vision_transcribe_appendixA.md"
VISION_B = ROOT / "data/processed/vision_transcribe_appendixB.md"
CARDS = ("body_cards.jsonl", "appendixA_cards.jsonl", "appendixB_cards.jsonl")


@pytest.fixture(scope="module")
def corpus():
    """一次扫描全语料，返回 {份数, 总页数, 需视觉通道页数, 本期页数, 本期视觉页数, 关键词命中文件数}。"""
    pdfs = sorted(RAW.glob("*.pdf"))
    if not pdfs:
        pytest.skip("语料不在")
    total = vision = scope_pages = scope_vision = 0
    hits = dict.fromkeys(KEYWORDS, 0)
    for p in pdfs:
        d = fitz.open(p)
        total += d.page_count
        low = sum(1 for i in range(d.page_count)
                  if len(re.findall(r"[\u4e00-\u9fff]", d[i].get_text())) < 60)
        vision += low
        if any(s in p.name for s in SCOPE_STD):
            scope_pages += d.page_count
            scope_vision += low
        text = "".join(d[i].get_text() for i in range(d.page_count))
        for k in hits:
            hits[k] += k in text
        d.close()
    return {"pdfs": len(pdfs), "pages": total, "vision": vision,
            "scope_pages": scope_pages, "scope_vision": scope_vision, "hits": hits}


def test_本期作物四处文档一致():
    """CLAUDE.md / docs/01 / README / docs/04 / sources.md 五处写到的本期作物与标准号必须一致，且不残留「待确认」占位。"""
    assert "作物：黄瓜、辣椒、大蒜" in CLAUDE
    assert "本期作物已确定：**黄瓜、辣椒、大蒜**" in DOC01
    assert "黄瓜、辣椒、大蒜" in README
    assert "【待确认】" not in README
    for crop in CROPS:                       # 语料清单里也要能看到三种作物
        assert crop in DOC04, crop
    for std in STD_NOS:                      # 来源记录里必须能查到对应标准号
        assert std in SOURCES, std


def test_本期作物在_data_raw_里真实存在():
    """文档登记的每份本期标准都必须能在 data/raw/ 里找到对应 PDF（防「文档说有、磁盘上没有」）。"""
    for std in STD_NOS:
        assert list((ROOT / "data/raw").glob(std.replace("/", "_") + "*.pdf")), std


def test_文档不再把小麦当本期作物():
    """旧口径「本期只做小麦」必须在所有文档里清干净，防止检索层还按老范围兜底。"""
    assert "本期聚焦 **【待确认" not in DOC01
    assert "作物：小麦" not in CLAUDE
    assert DOC01.count("不做小麦") >= 1 or "不含小麦语料" in CLAUDE
    for doc in (DOC04, SOURCES, README):     # 旧口径「本期只做小麦」不得残留
        assert "只做小麦" not in doc


def test_语料缺口结论与文档一致():
    """『农药混用/限用在本批语料中不存在』这个结论必须写在两份文档里（决定兜底策略）。"""
    assert "农药混用" in CLAUDE and "0 次" in CLAUDE
    assert "农药混用" in DOC01 and "0 次" in DOC01
    assert "口径" in CLAUDE and "口径" in DOC01     # 统计口径必须声明（175 页字形码页未参与统计）


def test_元数据字段覆盖解析_schema_全部字段():
    """docs/01 的 7.4 元数据表必须能对上提示词 schema 的每个字段名（严格口径，无豁免）。"""
    block, = re.findall(r"```json\s*(\[.*?\])\s*```", PROMPT, re.S)
    card = json.loads(block)[0]
    sec74 = DOC01.split("### 7.4")[1].split("### 7.5")[0]
    assert [k for k in card if f"`{k}`" not in sec74] == []
    for sub in ("dose", "dilution", "pre_harvest_interval", "max_uses_per_season"):
        assert sub in sec74, sub


def test_高危清单覆盖真实语料里的高危字段():
    sec44 = DOC01.split("### 4.4")[1].split("### 4.5")[0]
    for kw in ("安全间隔期", "每茬最多使用次数", "防治指标", "农药混用"):
        assert kw in sec44, kw
    assert "双人审核" in sec44


def test_地图时效规则写的是标准发布年():
    """时效字段（有效期起算）必须锚定标准发布年（2009–2022 区间），不能写成别的口径。"""
    assert "标准发布年" in DOC01
    assert "2009–2022" in DOC01 or "2009-2022" in DOC01


def test_解析要求已写入需求文档():
    """docs/01 的 7.6 必须明确解析硬要求：文本/视觉双通道、只抽取不改写、按 PDF 物理页、OCR 不猜字。"""
    sec76 = DOC01.split("### 7.6")[1]
    for kw in ("双通道", "视觉", "只抽取不改写", "PDF 物理页", "不猜字"):
        assert kw in sec76, kw


@pytest.mark.skipif(not list(RAW.glob("*.pdf")), reason="语料不在")
def test_语料事实与文档写的数字一致(corpus):
    """文档里的 26 份 / 433 页 / 175 页需视觉通道，必须能被 PDF 现算复现。"""
    assert corpus["pdfs"] == 26 and corpus["pages"] == 433
    assert corpus["vision"] == 175
    assert "433" in DOC01 and "433 页" in CLAUDE
    assert "175（约 40%）" in DOC01 and "175 页（40%）" in CLAUDE


@pytest.mark.skipif(not list(RAW.glob("*.pdf")), reason="语料不在")
def test_本期语料页数与文档一致(corpus):
    """本期 3 份 = 62 页，其中 24 页需视觉通道；三份文档都要写对（锚点锚到具体数字，别被别处的「62 页」蒙混）。"""
    assert (corpus["scope_pages"], corpus["scope_vision"]) == (62, 24)
    sec22 = DOC01.split("### 2.3")[0]                       # 2.2 合计行
    assert "**62**" in sec22 and "**24（约 39%）**" in sec22
    assert "62 页，需视觉通道 24 页（约 39%）" in DOC01        # 7.2 实测行
    assert "3 份 / 62 页，其中 24 页需视觉通道" in CLAUDE
    assert "62 页 / 需视觉通道 24 页" in DOC04


@pytest.mark.skipif(not list(RAW.glob("*.pdf")), reason="语料不在")
def test_语料缺口统计与文档结论一致(corpus):
    """『农药混用/限用 0 次命中』这个兜底依据必须与实测一致。"""
    assert corpus["hits"]["农药混用"] == 0 and corpus["hits"]["限用"] == 0
    assert corpus["hits"]["主要有害生物防治方案"] == 8
    assert "0 次" in CLAUDE and "0 次" in DOC01


@pytest.fixture(scope="module")
def vision_pages():
    """由卡片 source.channel 现算本期 3 份实际走视觉通道的 (标准号, 页码) 集合。"""
    paths = [ROOT / "data/processed" / f for f in CARDS]
    if not all(p.exists() for p in paths):
        pytest.skip("解析产物不在")
    vis = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            card = json.loads(line)
            src = card.get("source") or {}
            if src.get("channel") == "vision":
                vis.add((card["std_no"], src["page"]))
    return vis


@pytest.mark.skipif(not VISION_B.exists(), reason="视觉转录产物不在")
def test_视觉转录层关键词复核与文档结论一致():
    """2026-09-27 视觉层复核：混用/限用/禁用仍 0；大蒜的安全间隔期由「统计不到」改判为「有」。

    这些结论直接决定兜底策略（约束 8），锁住防止文档回退到「未复核」口径，
    也防止有人把"视觉层 0 命中"顺手写成"三份都没有安全间隔期"。
    """
    a = VISION_A.read_text(encoding="utf-8")
    b = VISION_B.read_text(encoding="utf-8")
    for kw in ("农药混用", "限用", "禁用"):
        assert kw not in a and kw not in b, f"视觉转录层出现了「{kw}」，缺口结论必须重写"
    assert b.count("安全间隔期") > 0 and b.count("最多使用次数") > 0
    assert "已过视觉层复核" in CLAUDE, "CLAUDE.md 第三节未回写视觉层复核结论"
    assert "结论变化" in DOC01 and "视觉层复核" in DOC01, "docs/01 2.2 未记录复核前后差异"


@pytest.mark.skipif(not list(RAW.glob("*.pdf")), reason="语料不在")
def test_视觉通道转录页数与文档一致(vision_pages):
    """实际走 vision 的页数由卡片现算，与文档写的 16 页对得上。

    防的是把「24 页需视觉通道」（CJK<60 的启发式上界）当成转录口径写进文档：
    实际 16 页——启发式标的 24 页里 12 页文本层可读（记录表表头能抽字），
    另有 4 页（黄瓜·辣椒附录B p20/p21）启发式没标出但按约束 7 走了视觉。
    """
    assert len(vision_pages) == 16, sorted(vision_pages, key=str)
    assert "**16 页**" in DOC01 and "**16 页**" in CLAUDE


def test_里程碑表已随实际进度更新():
    """docs/01 §11 里程碑表不得再写「待开始/待跑批/进行中」——各阶段产物都已落库。"""
    sec11 = DOC01.split("## 11.")[1].split("## 12.")[0]
    assert "☐" not in sec11, "§11 还有未勾选项"
    for stale in ("待开始", "待跑批", "进行中"):
        assert stale not in sec11, f"§11 里程碑表残留「{stale}」"


def test_语料清单写的是当前份数与页数():
    """docs/04 的头部结论行与清单标题必须写现在的份数/页数（锚到具体行，别被正文里的同数字蒙混）。"""
    head = DOC04.split("## 二、")[0]
    assert "**26 份** PDF / **433 页**" in head, "docs/04 头部份数/页数过期"
    assert "26 个 PDF，全部经校验" in DOC04
    assert "175" in DOC04
    assert "只做小麦" not in DOC04


def test_来源记录引用的文件真实存在():
    """sources.md 里写成 data/<dir>/xxx.pdf 或 `GB_Z xxx.pdf` 的引用必须真实存在（防「文档说有、磁盘上没有」）。"""
    refs = re.findall(r"data/(?:raw|external)/[^`\n|]+?\.pdf", SOURCES)
    refs += re.findall(r"`(GB_Z [^`\n]+?\.pdf)`", SOURCES)      # 文件名带空格，不能用 \S+
    assert len(refs) >= 4, f"应至少抽到 3 份本期语料 + 1 份外部附件，实际 {refs}"
    for rel in refs:
        p = (ROOT / rel) if rel.startswith("data/") else (RAW / rel)
        assert p.exists(), f"{rel} 被 sources.md 引用但磁盘上没有"


def test_外部来源不入语料目录():
    """协会公告等外部附件必须放 data/external/，不得混进 data/raw（否则污染 26/433 口径）。"""
    assert not list(RAW.glob("*CCPIA*")) and not list(RAW.glob("*农药工业协会*"))
    assert (ROOT / "data" / "external").is_dir()
    assert "data/external" in CLAUDE and "data/external" in SOURCES


def test_claude_md_格式规范():
    """首行畸形与嵌套缩进会让整个文档渲染错乱，锁住。"""
    assert not CLAUDE.startswith("- #")
    assert "  - ## " not in CLAUDE
    assert all(f"\n## {s}" in CLAUDE for s in
               ("一、项目目标", "三、本期范围", "四、核心约束", "六、知识设计底线规则", "七、常用指令与自检"))


def test_claude_md_记录的命令真实存在():
    """CLAUDE.md 第七节列出的每个脚本都必须在仓库里真实存在（防文档引用了已删除/改名的脚本）。"""
    for script in ("scripts/list_openstd.py", "scripts/verify_pdfs.py",
                   "src/ingest/download_gb_standards.py", "src/ingest/download_ccpia.py"):
        assert script in CLAUDE and (ROOT / script).exists(), script
    assert "python -m pytest" in CLAUDE


@pytest.mark.skipif(not list(RAW.glob("*.pdf")), reason="语料不在")
def test_清单与语料份数一致():
    """data/raw/_下载清单.csv 的行数必须等于实际 PDF 份数（防止清单过期）。"""
    import csv
    pdfs = list(RAW.glob("*.pdf"))
    rows = list(csv.DictReader(open(RAW / "_下载清单.csv", encoding="utf-8-sig")))
    assert len(rows) == len(pdfs) == 26


# ---------------------------------------------------------------- 口语映射表 / 意图表
def _load_json(rel):
    """读取仓库内 JSON 配置文件并解析为 Python 对象。"""
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_口语映射表条数与_configs_现算一致():
    """CLAUDE.md 第三节的条数必须由 configs 两个表现算复现。

    防的是「2026-09-17 人工筛定删了 7 条、迁走 19 条，但 CLAUDE.md 忘回写」那一类漂移
    （真发生过：文档写 76/54/22，代码实际 69/48/21）。
    """
    m = _load_json("configs/colloquial_map.json")
    it = _load_json("configs/intent_map.json")
    ver = [x for x in m if x["status"] == "verified"]
    pen = [x for x in m if x["status"] == "pending"]
    noc = [x for x in pen if x["pending_reason"] == "no_card"]
    amb = [x for x in pen if x["pending_reason"] == "ambiguity"]
    assert len(m) == len(ver) + len(pen), "status 只能取 verified/pending"
    assert len(pen) == len(noc) + len(amb), "每个 pending 必须带 pending_reason"

    sec3 = CLAUDE.split("### 3. 农户口语")[1].split("### 4. 来源权威性分级")[0]
    assert f"`configs/colloquial_map.json` | **术语归一化**：农户口语 → 标准术语 | **{len(m)}** |" in sec3
    assert f"术语表 {len(m)} 条：`verified` **{len(ver)}** / `pending` **{len(pen)}**" in sec3, \
        "CLAUDE.md 第三节术语表条数过期"
    assert f"`pending: no_card` **{len(noc)} 条**" in sec3
    assert f"`pending: ambiguity` **{len(amb)} 条**" in sec3

    n_int = len({x["intent"] for x in it})
    assert f"| **{len(it)}**（对应 **{n_int}** 个意图） |" in sec3, "CLAUDE.md 意图表条数过期"


def test_意图表每条问法都有意图_且每个意图被检索层接住():
    """intent_map 的意图必须要么被 search.py 接了路由，要么在免路由白名单里有名有姓。

    防的是「往意图表加了新意图，但 search.py 没接」——它不会报错，
    只会静默退化成全库检索（路由表达式为 None），召回质量悄悄下降。
    """
    it = _load_json("configs/intent_map.json")
    assert all(x["pattern"] and x["intent"] for x in it), "pattern / intent 不得为空"

    src = (ROOT / "src/retrieve/search.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    def _keys(name):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == name for t in node.targets):
                return {k.value for k in node.value.keys}
        raise AssertionError(f"search.py 里找不到 {name}")

    routable = _keys("ROUTE_BY_TITLE") | _keys("ROUTE_BY_KT") | _keys("INTENT_TO_ROUTE")
    # 字段级意图：问的是「剂量/间隔期/最多几次」这类字段值，没有对应小节可拦，
    # 有意不加路由（退回全库检索）。新增免路由意图必须加进这里并写明理由。
    UNROUTED_OK = {"dose", "max_uses_per_season", "pre_harvest_interval"}

    intents = {x["intent"] for x in it}
    orphan = intents - routable - UNROUTED_OK
    assert not orphan, f"这些意图没接路由也没进免路由白名单：{sorted(orphan)}"
    assert UNROUTED_OK <= intents, f"白名单里的意图已不在意图表：{sorted(UNROUTED_OK - intents)}"


# ---------------------------------------------------------------- 症状覆盖层 / 清点类数字
def test_症状覆盖层条数与文档一致():
    """`symptoms.jsonl` 的条目数与覆盖卡数由文件现算，锁**每一处**现状陈述。

    防的是铺开采集时加了条目、文档忘改（口语映射表条数漂移那类问题的翻版）。
    同一个事实现在写在四处、措辞各不相同，逐处锚原文——改一处漏一处必红：
    ① CLAUDE.md 二节（当前阶段，分号收尾接「c01 拆行」说明）
    ② CLAUDE.md 五节（目录说明）③ README 当前进度 ④ docs/06 症状覆盖层段
    ⑤ README 版本日志 V0.9 行：那是**带日期的历史记录**（铺开时追加新行、不回改旧行），
    故意不锁——锁了会逼人篡改历史。
    """
    p = ROOT / "data/processed/symptoms.jsonl"
    if not p.exists():
        pytest.skip("症状覆盖层未生成")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    n, cards = len(rows), len({r["card_id"] for r in rows})
    s = f"**{n} 条**（覆盖 **{cards} 张卡**）"
    assert s in CLAUDE, f"CLAUDE.md 五节未按现状回写症状条目数（应为 {s}）"
    assert s in README, f"README 当前进度未按现状回写症状条目数（应为 {s}）"
    # 二节措辞不同（分号 + c01 拆行说明），上面那条字符串盖不到，单独锚
    assert f"`data/processed/symptoms.jsonl` **{n} 条**（覆盖 **{cards} 张卡**；" in CLAUDE, \
        f"CLAUDE.md 二节未按现状回写症状条目数（应为 {n} 条 / {cards} 张卡）"
    # docs/06 的「张卡」没加粗，同数字换措辞也照样锁
    doc06 = (ROOT / "docs/06-全链路代码索引.md").read_text(encoding="utf-8")
    assert f"`data/processed/symptoms.jsonl` **{n} 条**（覆盖 {cards} 张卡；" in doc06, \
        f"docs/06 未按现状回写症状条目数（应为 {n} 条 / {cards} 张卡）"


def test_清点类文件数与磁盘现算一致():
    """`tests/` 与 `scripts/` 的文件个数必须与磁盘现算一致。

    这两个数历史上连错过两次（CLAUDE.md 的 tests：7→8→9），每次都是靠现算才发现的。
    """
    tests = len(list((ROOT / "tests").glob("test_*.py")))
    scripts = len(list((ROOT / "scripts").glob("*.py")))
    assert f"**{tests} 个**文件" in CLAUDE, f"CLAUDE.md 第五节的 tests 文件数过期（实际 {tests}）"
    assert f"`scripts/`（{scripts} 个）" in README, f"README 的 scripts 个数过期（实际 {scripts}）"
    doc06 = (ROOT / "docs/06-全链路代码索引.md").read_text(encoding="utf-8")
    assert f"`tests/` {tests} 个用例" in doc06, f"docs/06 的 tests 用例数过期（实际 {tests}）"

