# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/离线/test_golden_contract.py（T3 测试工程师：golden 测试集契约测试）

覆盖验收项（对应 t3 的 acceptance 与 设计/验收标准.md §3）：
  C1 golden 两份文件存在、可解析、16 题、id 与题面与 设计/工单需求原文.md §五 逐字一致
  C2 每题字段齐备（answer_keywords / evidence_pages / category / pdf / forbidden / traps / review）
  C3 每个证据页都在**真实 PDF** 中复核过：逐字证据串确实出现在该页（打开 PDF，不是抄）
  C4 图像题专项：id 6 页 72 两图 xref/尺寸/sha256 与 golden 登记一致；id 5 页 39 框图节点标签齐全
  C5 判分口径自检：正样本判对、负样本（含陷阱样本）全部判错
  C6 数值归一化等价：6,464.51≡6464.51、82.10%≡82.1%、1.5亿元≡15,000.00万元
  C7 「检索命中」≠「回答判对」：只有证据原文落在 chunk 里才算命中，引用页码不算
  C8 复核结论落盘（测试/测试数据/golden_复核结论.json）且每题有结论

运行：pwsh -NoProfile -File run_py.ps1 -m pytest 测试/离线/test_golden_contract.py -v
（离线：只用本机 PDF 与纯 Python 判定，不联网、不起服务、不调用 LLM）
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN_CANONICAL = ROOT / "研发" / "data" / "eval" / "golden_qa.jsonl"
GOLDEN_MIRROR = ROOT / "测试" / "测试数据" / "golden_qa_16.jsonl"
REVIEW_JSON = ROOT / "测试" / "测试数据" / "golden_复核结论.json"
SOURCE_MD = ROOT / "设计" / "工单需求原文.md"
RAW = ROOT / "研发" / "data" / "raw"

EXPECTED_ORDER = [5, 6, 1, 2, 3, 4, 260, 95, 33, 34, 957, 793, 795, 543, 531, 207]
PDF_NAME = {"招股说明书1": "招股说明书1.pdf", "招股说明书2": "招股说明书2.pdf"}


def _load_golden_lib():
    spec = importlib.util.spec_from_file_location(
        "golden_lib_under_test", ROOT / "测试" / "common" / "golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load_golden_lib()


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="session")
def records() -> list[dict]:
    return G.load_golden()


@pytest.fixture(scope="session")
def by_id(records) -> dict[int, dict]:
    return {r["id"]: r for r in records}


@pytest.fixture(scope="session")
def pdfs():
    fitz = pytest.importorskip("fitz")
    docs = {}
    for name, fn in PDF_NAME.items():
        p = RAW / fn
        if not p.exists():
            pytest.skip(f"缺少只读语料 {p}")
        docs[name] = fitz.open(p)
    yield docs
    for d in docs.values():
        d.close()


def norm_page(docs, pdf: str, page: int) -> str:
    return G.normalize(docs[pdf][page - 1].get_text("text"))


# ------------------------------------------------------------------ C1
def test_golden_files_exist_and_parse(records):
    """C1：两份 golden 文件存在、可解析、各 16 题。"""
    assert GOLDEN_CANONICAL.exists(), f"缺 {GOLDEN_CANONICAL}"
    assert GOLDEN_MIRROR.exists(), f"缺 {GOLDEN_MIRROR}"
    assert len(records) == 16, f"golden 必须 16 题，实际 {len(records)}"
    assert len(G.load_golden(GOLDEN_MIRROR)) == 16


def test_golden_mirror_is_identical():
    """C1：测试/测试数据 镜像与 研发/data/eval 主真值逐字节一致（禁止两份真值漂移）。"""
    a = GOLDEN_CANONICAL.read_bytes()
    b = GOLDEN_MIRROR.read_bytes()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest(), \
        "golden 主真值与镜像不一致——下游会读到第二份真值"


def test_questions_match_authoritative_source(records):
    """C1：题面与顺序逐字对齐 设计/工单需求原文.md §五（用户提示词附件副本）。"""
    text = SOURCE_MD.read_text(encoding="utf-8")
    m = re.search(r"```json\s*(\[.*?\])\s*```", text, re.S)
    assert m, "未在 设计/工单需求原文.md 找到 16 题 JSON 块"
    authoritative = json.loads(m.group(1))
    assert [q["id"] for q in authoritative] == [r["id"] for r in records] == EXPECTED_ORDER
    for q, r in zip(authoritative, records):
        assert q["question"] == r["question"], f"id {q['id']} 题面与权威题面不一致"
        assert r["order"] == EXPECTED_ORDER.index(r["id"]) + 1


# ------------------------------------------------------------------ C2
@pytest.mark.parametrize("rid", EXPECTED_ORDER)
def test_field_contract(by_id, rid):
    """C2：字段齐备（关键字/证据页/题型/语料/禁答项/陷阱/复核结论/工单3 兼容别名）。"""
    rec = by_id[rid]
    problems = G.validate_record_shape(rec)
    assert not problems, f"id {rid} 字段契约问题: {problems}"


def test_category_distribution(records):
    """C2：题型分类覆盖四类，且图像两题分别为 图像结构 / 图像图表。"""
    cats = {r["id"]: r["category"] for r in records}
    assert cats[5] == "图像结构" and cats[6] == "图像图表"
    assert set(r["category"] for r in records) == set(G.CATEGORIES)
    counts = {c: sum(1 for r in records if r["category"] == c) for c in G.CATEGORIES}
    assert counts["文本事实"] >= 8, counts


def test_pdf_corpus_mapping(records):
    """C2：语料归属正确（力源信息 6 题在 PDF2；兴图新科 10 题在 PDF1）。"""
    pdf2 = {r["id"] for r in records if r["pdf"] == "招股说明书2"}
    assert pdf2 == {5, 6, 1, 2, 3, 4}, pdf2
    assert all(r["corpus"] == ("pdf2" if r["pdf"] == "招股说明书2" else "pdf1") for r in records)


# ------------------------------------------------------------------ C3
@pytest.mark.parametrize("rid", EXPECTED_ORDER)
def test_evidence_pages_verified_on_real_pdf(by_id, pdfs, rid):
    """C3：每个证据页都在真实 PDF 中复核——逐字证据串/配对证据确实出现在该页。"""
    rec = by_id[rid]
    total = pdfs[rec["pdf"]].page_count
    for page in rec["evidence_pages"] + rec.get("supporting_pages", []) + rec.get("related_pages", []):
        assert 1 <= page <= total, f"id {rid}: 证据页 {page} 超出 {rec['pdf']} 共 {total} 页"
    facts = {f["page"]: f for f in rec["page_facts"]}
    for page in rec["evidence_pages"]:
        assert page in facts, f"id {rid}: 页 {page} 缺 page_facts 复核记录"
        assert facts[page]["verified"] is True, f"id {rid}: 页 {page} 未通过复核"
    pt_cache: dict[int, str] = {}

    def page_text(p: int) -> str:
        if p not in pt_cache:
            pt_cache[p] = norm_page(pdfs, rec["pdf"], p)
        return pt_cache[p]

    for ev in rec["retrieval_evidence"]:
        p = ev["page"]
        pt = page_text(p)
        if ev["kind"] in ("verbatim", "diagram_text_layer"):
            assert G.normalize(ev["text"]) in pt, \
                f"id {rid}: 页 {p} 未逐字包含证据原文（{ev['text'][:40]}…）"
        elif ev["kind"] == "chart_pairs":
            # 图像通道：本机无 OCR，故按「图像确实在该页 + 视觉复核后的图表数值」校验
            chart = rec.get("chart") or {}
            bar = next((f for f in chart.get("figures", []) if f["xref"] == 1692), None)
            assert bar, f"id {rid}: 缺条形图 xref=1692 的登记"
            assert chart["page"] == p
            assert 1692 in {i[0] for i in pdfs[rec["pdf"]][p - 1].get_images(full=True)}, \
                f"id {rid}: 页 {p} 上找不到条形图 xref=1692"
            for pair in ev["pairs"]:
                val = bar["data"].get(pair[0])
                assert val is not None and val.startswith(pair[1]), \
                    f"id {rid}: 图表数据与配对证据不一致 {pair} vs {val}"
        elif ev["kind"] == "pairs":
            for pair in ev["pairs"]:
                assert all(G.normalize(x) in pt for x in pair), \
                    f"id {rid}: 页 {p} 缺配对证据 {pair}"
        else:  # pragma: no cover
            pytest.fail(f"id {rid}: 未知证据类型 {ev['kind']}")


@pytest.mark.parametrize("rid", EXPECTED_ORDER)
def test_answer_keywords_grounded_in_evidence(by_id, pdfs, rid):
    """C3：answer_keywords 必须真的落在证据页上（防止凭空造关键字）；
    answer_keywords_any_of 每组至少有一种写法在证据页上有据。"""
    rec = by_id[rid]
    pt_all = "".join(norm_page(pdfs, rec["pdf"], p) for p in rec["evidence_pages"])
    for kw in rec["answer_keywords"]:
        if rec["category"] == "图像图表":
            chart = rec.get("chart", {})
            blob = json.dumps(chart, ensure_ascii=False) + pt_all
            assert G.normalize(kw.rstrip("%")) in G.normalize(blob), \
                f"id {rid}: 图表关键字 {kw} 在图表数据与证据页文本中均未见"
        else:
            assert G.page_carries(pt_all, kw), f"id {rid}: 关键字 {kw} 未出现在任何证据页"
    for gi, group in enumerate(rec.get("answer_keywords_any_of", []), 1):
        assert any(G.page_carries(pt_all, m) for m in group), \
            f"id {rid}: 任一组 {gi} 的全部写法在证据页中均无据: {group}"


# ------------------------------------------------------------------ C4
def test_chart_evidence_assets(by_id, pdfs):
    """C4：id 6 页 72 的两张图 xref/尺寸/sha256 与 golden 登记一致（可复现的像素级证据）。"""
    rec = by_id[6]
    chart = rec["chart"]
    assert chart["page"] == 72
    doc = pdfs["招股说明书2"]
    on_page = {info[0] for info in doc[71].get_images(full=True)}
    for fig in chart["figures"]:
        assert fig["xref"] in on_page, f"页 72 不含 xref {fig['xref']}"
        img = doc.extract_image(fig["xref"])
        assert (img["width"], img["height"]) == (fig["width"], fig["height"])
        assert hashlib.sha256(img["image"]).hexdigest()[:16] == fig["sha256_16"]
    bar = next(f for f in chart["figures"] if f["xref"] == 1692)
    assert bar["data"]["汽车"] == "14.0%" and bar["extremes"]["增长最快"] == "汽车 14.0%"
    assert bar["data"]["IC卡"] == "-2.0%" and bar["extremes"]["唯一负增长"] == "IC卡 -2.0%"
    pie = next(f for f in chart["figures"] if f["xref"] == 1693)
    assert pie["data"]["计算机"].startswith("2,515")
    assert G.normalize(chart["text_cross_check"]) in norm_page(pdfs, "招股说明书2", 72)


def test_diagram_evidence_assets(by_id, pdfs):
    """C4：id 5 页 39 框图：4 个部门 + 6 个销售处标签齐全，上海分公司与销售处不同层。"""
    rec = by_id[5]
    dia = rec["diagram"]
    pt = norm_page(pdfs, "招股说明书2", 39)
    for dept in dia["structure"]["销售部"]:
        assert G.normalize(dept) in pt, f"页 39 框图缺部门 {dept}"
    assert len(dia["structure"]["大客户销售部"]) == 6
    for office in dia["structure"]["大客户销售部"]:
        assert G.normalize(office) in pt, f"页 39 框图缺销售处 {office}"
    assert "上海分公司" in dia["structure"]["总经理"], "上海分公司必须登记在总经理层（禁答项依据）"
    assert G.normalize("上海分公司") in pt
    assert dia["rendered_page"]["path"]


# ------------------------------------------------------------------ C5
@pytest.mark.parametrize("rid", EXPECTED_ORDER)
def test_judge_samples(by_id, rid):
    """C5：判分口径自检——正样本判对，全部负样本（陷阱）判错。"""
    rec = by_id[rid]
    pos = G.judge_answer(rec, rec["judge_samples"]["positive"])
    assert pos["correct"], f"id {rid} 正样本被判错: {pos['reason']}"
    for i, neg in enumerate(rec["judge_samples"]["negatives"], 1):
        r = G.judge_answer(rec, neg)
        assert not r["correct"], f"id {rid} 负样本 {i} 被判对（判分漏洞）: {neg[:40]}…"


@pytest.mark.parametrize("rid", EXPECTED_ORDER)
def test_golden_answers_self_consistent(by_id, rid):
    """C5：golden 自己的 answer 必须能被自己的判分口径判对（真值自洽，防止关键字与答案互相矛盾）。"""
    rec = by_id[rid]
    r = G.judge_answer(rec, rec["answer"])
    assert r["correct"], f"id {rid}: golden 答案被判错（真值不自洽）: {r['reason']}"


def test_page_set_equalities(records):
    """C2：页级等式全域成立（16/16）——
    evidence_pages = complete + supporting；evidence_pages_union = complete + supporting + related。"""
    for r in records:
        complete = set(r["evidence_pages_complete"])
        supporting = set(r.get("supporting_pages", []))
        related = set(r.get("related_pages", []))
        assert r["evidence_pages"] == sorted(complete | supporting), \
            f"id {r['id']}: evidence_pages ≠ complete+supporting"
        assert r["evidence_pages_union"] == sorted(complete | supporting | related), \
            f"id {r['id']}: union 漏页 {sorted((complete | supporting | related) - set(r['evidence_pages_union']))}"
        assert complete and set(r["evidence_pages_primary"]) <= complete


def test_answer_completeness_policy(by_id):
    """C5：枚举完整性口径——要求逐项列全的题，漏项必须判错；
    而 answer_keywords_any_of 只接受「同一实体的不同写法」，不豁免完整性。"""
    # (1) 枚举题漏一项 → 判错（不得因 any_of 放宽）
    assert not G.judge_answer(
        by_id[5], "销售部由4个部门构成：渠道销售部、大客户销售部、国际贸易部；"
                  "大客户销售部由6个销售处构成：北京、广州、成都、深圳、武汉、珠海。")["correct"], \
        "漏『电话及网络销售部』应判错"
    assert not G.judge_answer(
        by_id[4], "不存在控制关系的关联方共7家：融冰投资、武汉博润、上海博润、听音投资、"
                  "联众聚源、力源贸易。")["correct"], "漏『普芯达』应判错"
    assert not G.judge_answer(
        by_id[2], "拟投资5个项目：仓储及物流中心3,393.40万元、研发中心1,526.38万元、"
                  "电子商务平台2,492.78万元、扩充产品种类和数量、其他与主营业务相关的营运资金。")["correct"], \
        "漏 9,000.00 万元应判错"
    # (2) 全列 + 名称写法变体 → 判对（any_of 生效，只放宽写法、不放宽完整性）
    ok_cases = [
        (4, "不存在控制关系的关联方共7家：融冰投资、武汉博润、上海博润、听音投资、"
            "联众聚源、力源贸易、普芯达。"),
        (5, "销售部由4个部门构成：电话网络销售部、渠道销售部、大客户销售部、国际贸易部；"
            "大客户销售部由6个销售处构成：北京、广州、成都、深圳、武汉、珠海。"),
        (2, "拟投资5个项目：仓储物流中心3,393.40万元、研发中心1,526.38万元、电子商务平台2,492.78万元、"
            "扩充产品种类和数量9,000.00万元、其他与主营业务相关的营运资金（金额未披露）。"),
        (795, "2014年12月，某情报指挥控制与通信网络一体化工程荣获国家科技进步一等奖。"),
        (95, "公司参与制定了全军第一个视频指挥系统技术标准，即《某视频技术规范1.0》。"),
    ]
    for rid, ans in ok_cases:
        assert G.judge_answer(by_id[rid], ans)["correct"], f"id {rid} 该判对却被判错: {ans[:40]}"


def test_min_hit_top_k_semantics(by_id):
    """C7：min_hit_top_k 精确语义——「前 K 条 chunk 内命中即算」，且函数按 K 截断。"""
    rec = by_id[1]
    ev_text = rec["retrieval_evidence"][0]["text"]
    k = rec["min_hit_top_k"]
    assert k == G.DEFAULT_MIN_HIT_TOP_K == 5
    filler = [{"page": 1, "text": f"无关 chunk {i}"} for i in range(k)]
    r = G.check_retrieval_hit(rec, filler + [{"page": 22, "text": ev_text}])
    assert not r["hit"], "证据落在第 K+1 条却被判命中——必须在『前 K 条』内"
    assert r["top_k"] == k and r["n_chunks_passed"] == k + 1 and r["n_chunks_considered"] == k
    r2 = G.check_retrieval_hit(rec, filler[:k - 1] + [{"page": 22, "text": ev_text}])
    assert r2["hit"] and r2["matched"][0]["chunk_index"] == k - 1
    # 严格模式：显式收窄到 3（生成阶段 Top-K=3）时，第 4 条的证据不算命中
    r3 = G.check_retrieval_hit(rec, filler[:3] + [{"page": 22, "text": ev_text}], top_k=3)
    assert not r3["hit"] and r3["top_k"] == 3


def test_id5_seven_offices_trap_rejected(by_id):
    """C5：id 5 —— 「6 城市 + 上海分公司 = 7 项却不写数字」的隐蔽错答必须判错；
    带否定澄清（上海分公司不计入销售处）的正答必须判对。"""
    rec = by_id[5]
    bad = ("销售部由4个部门构成：电话及网络销售部、渠道销售部、大客户销售部、国际贸易部；"
           "大客户销售部由6个销售处构成：北京、广州、成都、深圳、武汉、珠海、上海分公司。")
    r = G.judge_answer(rec, bad)
    assert not r["correct"], "把上海分公司列入销售处的答案被判对（判分漏洞）"
    assert any(f.get("keyword") == "上海分公司" for f in r["forbidden_fired"])
    ok = ("销售部由4个部门构成：电话及网络销售部、渠道销售部、大客户销售部、国际贸易部；"
          "大客户销售部由6个销售处构成：北京、广州、成都、深圳、武汉、珠海（上海分公司不计入销售处）。")
    assert G.judge_answer(rec, ok)["correct"], "带否定澄清的正确表述被误判为错"


def test_golden3_alias_fields_are_record_fields(records):
    """C2：工单3 兼容字段是**记录字段**（record["min_hit_top_k"]），不是 golden 模块顶层常量——
    契约测试在此固化调用约定，避免 T5/T6 写成 golden.min_hit_top_k 触发 AttributeError。"""
    for r in records:
        assert r["required_substrings"] == r["answer_keywords"]
        assert r["forbidden_substrings"] == r["forbidden_keywords"]
        assert r["evidence"] == r["evidence_verbatim"]
        assert r["should_be_unknown"] is False
        assert r["min_hit_top_k"] == G.DEFAULT_MIN_HIT_TOP_K == 5
        assert r["max_first_token_ms"] == G.DEFAULT_MAX_FIRST_TOKEN_MS == 3000
    for name in G.GOLDEN3_RECORD_FIELDS:
        assert not hasattr(G, name), f"{name} 是记录字段，不应在 golden 模块顶层暴露同名常量"
    assert hasattr(G, "DEFAULT_MIN_HIT_TOP_K") and hasattr(G, "DEFAULT_MAX_FIRST_TOKEN_MS")


def test_cross_document_confusions_rejected(by_id):
    """C5：跨文件串答必须判错（PDF1 与 PDF2 同字段不同值）。"""
    cases = [
        (1, "本次发行1,840万股，占发行后总股本的比例不低于25.00%。"),      # PDF1 兴图新科的发行数据
        (531, "法定代表人是赵马克。"),                                      # PDF2 力源信息的法定代表人
        (543, "注册资本为5,000.00万元。"),                                  # PDF2 力源信息的注册资本
        (3, "存在控制关系的关联方为程家明，持股42.35%，为公司控股股东。"),   # PDF1 的控股股东
    ]
    for rid, answer in cases:
        r = G.judge_answer(by_id[rid], answer)
        assert not r["correct"], f"id {rid} 跨文件串答被判对: {answer}"


def test_image_trap_answers_rejected(by_id):
    """C5：两处图像陷阱——上海分公司计入销售处、最快/负增长答反，必须判错。"""
    bad5 = ("销售部由5个部门构成：电话及网络销售部、渠道销售部、大客户销售部、国际贸易部和上海销售处；"
            "大客户销售部由6个销售处构成：北京、广州、成都、深圳、武汉、珠海。")
    assert not G.judge_answer(by_id[5], bad5)["correct"]
    bad6 = "增长率最快的是工控（10.5%）；负增长的是网络通信（-2.0%）。"
    assert not G.judge_answer(by_id[6], bad6)["correct"]


def test_negation_guard_allows_caveated_correct_answer(by_id):
    """C5：禁止误答项出现在否定语境时不判错（避免误杀带澄清语的正答）。"""
    answer = ("不存在控制关系的关联方共7家：融冰投资、武汉博润、上海博润、听音投资、联众聚源、"
              "力源贸易、普芯达；赵马克不属于不存在控制关系的关联方（他是控股股东）。")
    r = G.judge_answer(by_id[4], answer)
    assert r["correct"], f"否定语境的禁止项被误判为违规: {r['forbidden_fired']}"
    assert r["forbidden_guarded"], "应记录被护栏豁免的禁止项"


def test_fabricated_item5_amount_rejected(by_id):
    """C5：为 PDF2 募投第 5 项（原文占位符 [ ◆ ]）编造金额必须判错。"""
    bad = ("本次募集资金拟投资5个项目：仓储及物流中心3,393.40万元、研发中心1,526.38万元、"
           "电子商务平台2,492.78万元、扩充产品种类和数量9,000.00万元、"
           "其他与主营业务相关的营运资金5,000万元。")
    r = G.judge_answer(by_id[2], bad)
    assert not r["correct"]
    assert any("营运资金" in f.get("regex", "") for f in r["forbidden_fired"])


# ------------------------------------------------------------------ C6
def test_numeric_normalization_equivalence():
    """C6：数值归一化等价（逗号/全角/百分之/万元↔亿元）。"""
    assert G.normalize("6,464.51 万元") == G.normalize("6464.51万元")
    assert G.normalize("百分之82.10") == G.normalize("82.10%") == "82.10%"
    assert G.keyword_hit("占比为82.1%", "82.10%")[0]
    assert G.keyword_hit("金额6,464.51万元", "6464.51万元")[0]
    assert G.keyword_hit("使用1.5亿元用于补充流动资金", "15,000.00万元")[0]
    assert G.keyword_hit("注册资本5,520万元", "5,520.00万元")[0]
    assert not G.keyword_hit("占比为8.21%", "82.10%")[0]
    assert not G.keyword_hit("金额6,464.5万元", "6,464.51万元")[0]


def test_question_specific_numbers(by_id):
    """C6：关键数值题在归一化后判对（含千分位/单位/占比写法差异）。"""
    assert G.judge_answer(by_id[260], "报告期内军用领域收入分别为6464.51万元、14414.16万元、"
                                      "18780.67万元和4627.14万元。")["correct"]
    assert G.judge_answer(by_id[1], "本次发行1670万股，占发行后总股本比例为25.04%。")["correct"]
    assert G.judge_answer(by_id[207], "计划使用募集资金1.5亿元用于补充流动资金。")["correct"]


# ------------------------------------------------------------------ C7
def test_retrieval_hit_requires_evidence_in_chunk(by_id):
    """C7：检索命中必须以证据原文落在返回 chunk 中判定，『只引用页码』不算命中。"""
    rec = by_id[1]
    citation_only = [{"page": 22, "text": "参见 招股说明书2.pdf 第22页 的发行情况表"}]
    r = G.check_retrieval_hit(rec, citation_only)
    assert not r["hit"], "只引用页码却被判为检索命中——违反验收标准 §3.4 第 2 条"
    good = [{"page": 22, "text": "四、本次发行情况 股票种类 人民币普通股(A股) 发行股数 "
                                "1,670 万股，占发行后总股本的比例为25.04%"}]
    r2 = G.check_retrieval_hit(rec, good)
    assert r2["hit"] and r2["matched"][0]["page"] == 22


def test_retrieval_hit_separate_from_answer_judgement(by_id):
    """C7：命中与判对分别报告——命中但答错（或反之）都要如实记录。"""
    rec = by_id[1]
    chunks = [{"page": 22, "text": "发行股数 1,670 万股，占发行后总股本的比例为25.04%"}]
    hit = G.check_retrieval_hit(rec, chunks)
    wrong_answer = G.judge_answer(rec, "本次发行1,840万股，占比不低于25.00%。")
    assert hit["hit"] and not wrong_answer["correct"], "命中但答错的情形必须能被分别表达"


def test_retrieval_hit_chart_pairs(by_id):
    """C7：图像图表题的命中判定基于『行业-增长率』配对，单边出现不算命中。"""
    rec = by_id[6]
    half = [{"page": 72, "text": "2008年中国IC市场应用结构与增长(亿元) 汽车"}]
    assert not G.check_retrieval_hit(rec, half)["hit"]
    full = [{"page": 72, "text": "图表解析：汽车 14.0%；IC卡 -2.0%；工控 10.5%"}]
    r = G.check_retrieval_hit(rec, full)
    assert r["hit"] and r["matched"][0]["page"] == 72


def test_retrieval_hit_diagram_labels(by_id):
    """C7：框图题命中要求页 39 的部门/销售处标签串落在 chunk 中（列聚类重建后）。"""
    rec = by_id[5]
    bad = [{"page": 39, "text": "财场息销政物 报表区"}]
    assert not G.check_retrieval_hit(rec, bad)["hit"]
    good = [{"page": 39, "text": "电话及网络销售部渠道销售部大客户销售部国际贸易部珠海销售处"
                                "深圳销售处北京销售处武汉销售处广州销售处成都销售处"}]
    assert G.check_retrieval_hit(rec, good)["hit"]


# ------------------------------------------------------------------ C8
def test_strictness_asymmetry(by_id, records):
    """C5/C6：「数值宽松（只放格式、不放精度）、枚举严格」——逐条实测该规律，并确认报告已写明。
    数值：5,520.00万元 ≡ 5,520万元 ≡ 5520万元；15,000.00万元 ≡ 15,000万元 ≡ 1.5亿元；1,670万股 ≡ 1670万股。
    精度：6,464.5万元 ≠ 6,464.51万元。枚举：全称必须保留，截断/「等」概括即判错。"""
    num_ok = [
        (543, "注册资本为5,520.00万元。"), (543, "注册资本为5,520万元。"), (543, "注册资本为5520万元。"),
        (207, "使用15,000.00万元用于补充流动资金。"), (207, "使用15,000万元用于补充流动资金。"),
        (207, "使用1.5亿元用于补充流动资金。"),
        (1, "本次发行1,670万股，占发行后总股本的比例为25.04%。"),
        (1, "本次发行1670万股，占发行后总股本的比例为25.04%。"),
    ]
    for rid, ans in num_ok:
        assert G.judge_answer(by_id[rid], ans)["correct"], f"id {rid} 数值等价写法被误杀: {ans}"
    num_bad = [
        (260, "军用领域收入分别为6,464.5万元、14,414.16万元、18,780.67万元和4,627.14万元。"),  # 精度截断
        (543, "注册资本为5,520万美元。"),                                                      # 量纲不符
    ]
    for rid, ans in num_bad:
        assert not G.judge_answer(by_id[rid], ans)["correct"], f"id {rid} 精度/量纲错误被判对: {ans}"
    enum_bad = [
        # 「等」**替代**被省略的项 → 判错（缺 `其他与主营业务相关的营运资金`）
        (2, "拟投资5个项目：仓储及物流中心3,393.40万元、研发中心1,526.38万元、电子商务平台2,492.78万元、"
            "扩充产品种类和数量9,000.00万元等。"),
        # 名称截断 → 判错
        (2, "拟投资5个项目：仓储及物流中心3,393.40万元、研发中心1,526.38万元、电子商务平台2,492.78万元、"
            "扩充产品种类和数量9,000.00万元、其他与主营业务相关。"),
    ]
    for rid, ans in enum_bad:
        assert not G.judge_answer(by_id[rid], ans)["correct"], f"id {rid} 枚举截断/概括被判对"
    # 相反方向：**列全之后**尾随一个「等」不算错（没有省略任何项，不得误杀）
    assert G.judge_answer(
        by_id[2], "拟投资5个项目：仓储及物流中心3,393.40万元、研发中心1,526.38万元、电子商务平台2,492.78万元、"
                  "扩充产品种类和数量9,000.00万元、其他与主营业务相关的营运资金等。")["correct"], \
        "列全后尾随「等」属于措辞冗余，不应判错"
    # 数值关键字必须带单位（否则退化为子串匹配，精度口径失效）
    for r in records:
        for kw in r["answer_keywords"]:
            if any(ch.isdigit() for ch in kw):
                assert re.search(r"(万元|亿元|元|万股|亿股|股|%|倍)$", G.normalize(kw)), \
                    f"id {r['id']}: 数值关键字 `{kw}` 未带单位"
    text = (ROOT / "测试" / "测试数据" / "golden_复核报告.md").read_text(encoding="utf-8")
    assert "数值宽松" in text and "枚举严格" in text, "报告未写明严格度不对称口径"
    assert "不同源" in text, "报告未写明『答案字段与判分字段不同源』"


def test_three_layer_hit_semantics(by_id):
    """C7：三层命中口径（captain 裁定）——用 id 5 的真实 chunk 钉住三层差异：
    页 39 chunk（完整证据）：三层全 True；页 40 chunk（仅含 4 个部门句）：权威层 False、诊断层A True、诊断层B True；
    无关 chunk：三层全 False。三层不可合并（页级只看页码，任证层要求证据原文落块）。"""
    rec = by_id[5]
    diag = G.load_diagnostic_evidence()
    assert rec["id"] in diag, "诊断层证据文件缺 id 5"
    supporting_texts = [e["text"] for e in diag[5]["supporting"]]
    assert any("销售部下设渠道销售部" in t for t in supporting_texts), supporting_texts

    diag5 = {5: diag[5]}
    p39 = [{"page": 39, "text": rec["retrieval_evidence"][0]["text"]}]
    r39 = G.check_retrieval_hit_layers(rec, p39, diagnostic=diag5)
    assert (r39["hit_complete"], r39["hit_any_evidence"], r39["hit_page_level"]) == (True, True, True)
    p40 = [{"page": 40, "text": "销售部下设渠道销售部、电话及网络销售部、大客户销售部和国际贸易部。"}]
    r40 = G.check_retrieval_hit_layers(rec, p40, diagnostic=diag5)
    assert r40["hit_complete"] is False, "页 40 不含 6 个销售处证据，权威层必须为 False"
    assert r40["hit_any_evidence"] is True, "页 40 含 supporting 证据原文，诊断层A 应为 True"
    assert r40["hit_page_level"] is True, "页 40 ∈ evidence_pages，诊断层B 应为 True"
    none = [{"page": 38, "text": "（与本题无关的页）"}]
    rn = G.check_retrieval_hit_layers(rec, none, diagnostic=diag5)
    assert (rn["hit_complete"], rn["hit_any_evidence"], rn["hit_page_level"]) == (False, False, False)
    # 页级口径的虚高实例：页码在集合内但内容无关 → 页级 True、任证层 False（证明两层可分）
    fake = [{"page": 113, "text": "（页码在集合内但内容与证据无关）"}]
    rf = G.check_retrieval_hit_layers(rec, fake, diagnostic=diag5)
    assert rf["hit_page_level"] is True and rf["hit_any_evidence"] is False
    # 权威层与 check_retrieval_hit 完全一致（向后兼容）
    assert G.check_retrieval_hit(rec, p40)["hit"] is False
    assert G.check_retrieval_hit(rec, p39)["hit"] is True
    assert r39["authoritative_layer"] == "hit_complete"
    # 片段级（诊断层A）：captain 追踪到的真实分块 —— 支持性枚举被切到 p40-x1 / p41-x0 两块
    sup0 = supporting_texts[0]
    frags0 = G.clause_fragments(sup0, keywords=rec["answer_keywords"])
    for must in ("电话及网络销售部", "大客户销售部", "国际贸易部"):
        assert must in frags0, f"片段应含 `{must}`：{frags0}"
    half1 = [{"page": 40, "text": "销售部下设渠道销售部、电话及网络销售部、大客户销售部"}]
    r1 = G.check_retrieval_hit_layers(rec, half1, diagnostic=diag5)
    assert (r1["hit_complete"], r1["hit_any_evidence"], r1["hit_page_level"]) == (False, True, True)
    assert r1["matched_fragments"], "片段命中须留下片段明细"
    half2 = [{"page": 41, "text": "国际贸易部；负责海外市场销售"}]
    r2 = G.check_retrieval_hit_layers(rec, half2, diagnostic=diag5)
    # 证据原文（片段）在块里 → 诊断层A True；但页 41 不在 evidence_pages → 诊断层B False（两层独立）
    assert (r2["hit_complete"], r2["hit_any_evidence"], r2["hit_page_level"]) == (False, True, False), \
        "诊断层A 与诊断层B 必须相互独立：证据在原页外出现时 A=True 而 B=False"


def test_diagnostic_evidence_file_is_derived(records):
    """C2：诊断层证据文件存在、覆盖全部 supporting 页、且**不并入 golden 冻结哈希**。"""
    diag = G.load_diagnostic_evidence()
    assert diag, f"缺诊断层证据文件 {G.DIAGNOSTIC_EVIDENCE}"
    assert len(diag) == 16
    for r in records:
        it = diag[r["id"]]
        sup_pages = {e["page"] for e in it["supporting"]}
        # supporting 证据：页码必须落在 evidence_pages 内；且「不在完整证据页上的 supporting 页」== golden.supporting_pages
        assert sup_pages <= set(r["evidence_pages"]), \
            f"id {r['id']}: 诊断层 supporting 页 {sorted(sup_pages)} 不在 evidence_pages 内"
        assert sorted(sup_pages - set(r["evidence_pages_complete"])) == r["supporting_pages"], \
            f"id {r['id']}: 诊断层 supporting 页与 golden.supporting_pages 不一致"
        for e in it["supporting"]:
            assert e.get("text") or e.get("pairs"), f"id {r['id']}: supporting 证据缺原文"
    assert G.GOLDEN_CANONICAL.read_bytes() == G.GOLDEN_MIRROR.read_bytes()


def test_report_documents_actual_negation_cues(records):
    """C8：文档—实现一致性——复核报告必须列全实际的否定护栏触发词与窗口半径
    （captain 曾发现文档比实现窄：报告漏了『不属于』等，会误导 T5/T6）。"""
    text = (ROOT / "测试" / "测试数据" / "golden_复核报告.md").read_text(encoding="utf-8")
    for cue in G.NEGATION_CUES:
        assert cue in text, f"报告未列出实际否定护栏触发词：{cue}"
    assert str(G.GUARD_WINDOW) in text, "报告未写明否定护栏窗口半径"
    assert "min_hit_top_k" in text and "前" in text and "命中即算" in text, \
        "报告未写明 min_hit_top_k 的精确语义"
    assert "逐项" in text and "枚举" in text, "报告未写明枚举完整性口径"


def test_review_conclusions_recorded(records):
    """C8：逐题复核结论落盘，每题含结论/复核人/打开过的页，且全部 verified。"""
    assert REVIEW_JSON.exists(), f"缺复核结论 {REVIEW_JSON}"
    data = json.loads(REVIEW_JSON.read_text(encoding="utf-8"))
    assert data["n_questions"] == 16
    items = {it["id"]: it for it in data["items"]}
    assert sorted(items) == sorted(EXPECTED_ORDER)
    for r in records:
        it = items[r["id"]]
        assert it["verdict"] and it["note"] and it["agreement_with_captain"]
        assert it["pages_opened_and_verified"], f"id {r['id']} 缺逐页复核明细"
        assert all(p["verified"] for p in it["pages_opened_and_verified"]), \
            f"id {r['id']} 存在未通过复核的证据页"
        assert it["judge_selftest"]["positive_correct"] is True
        assert it["judge_selftest"]["negatives_all_wrong"] is True
        assert r["review"]["reviewed"] is True


def test_每个证据页都至少绑定一条证据(by_id):
    """C8：evidence_pages 每个页都必须能在 page_facts 里找到非 related 的证据条目。"""
    for rid, rec in by_id.items():
        roles = {f["page"]: f["role"] for f in rec["page_facts"]}
        for p in rec["evidence_pages"]:
            assert roles.get(p) in ("primary", "alt", "supporting"), \
                f"id {rid}: 页 {p} 被列为证据页但无对应证据条目"
        for p in rec["related_pages"]:
            assert roles.get(p) == "related", f"id {rid}: 页 {p} 的 related 角色不一致"
