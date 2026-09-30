# -*- coding: utf-8 -*-
"""生成层模板自检：锁「高危卡怎么被渲染成答案」这件事。

为什么单独来一层：这里的输出是农户照着配药的凭据，**渲染器错一格就是害人**。
检索层错顶多答非所问，生成层错会凭空给药方——2026-09-23 真发生过：
`chemicals` 为空的正文卡也走用药方案模板，渲染成
「【结论】黄瓜的7.1，可用下列药剂」+ 一个空的【用药方案】，
章节号被当成病虫害名，卡片真正的正文一个字都没出来。

不联网、不加载模型、不碰 Milvus：直接拿 data/chunks/chunks.jsonl 里的真实卡片喂给模板。
之所以能这么轻：`answer.py` 只在 main() 里才 import 检索层。
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]                 # 仓库根目录
sys.path.insert(0, str(ROOT / "src" / "generate"))         # 把生成层目录加入导入路径，便于 import answer

import answer as A  # noqa: E402  # 被测对象：生成层模板渲染模块

CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"         # 真实卡片语料（不 mock，直接喂模板）


@pytest.fixture(scope="module")
def rows():
    """加载全量真实卡片；语料未生成时跳过本模块所有用例。"""
    if not CHUNKS.exists():
        pytest.skip("chunks.jsonl 没生成（先跑 scripts/build_chunks.py）")
    return [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def hi_rows(rows):
    """needs_verification_hint=true 的卡——高危所有谈论的范围。"""
    return [r for r in rows if r["card"]["needs_verification_hint"]]


@pytest.fixture(scope="module")
def nochem(hi_rows):
    """条款型高危卡：有要求，没有药剂（2026-09-24 修的那批）。"""
    return [r for r in hi_rows if not r["card"].get("chemicals")]


@pytest.fixture(scope="module")
def withchem(hi_rows):
    """用药方案型高危卡：附录B 表B.1 那 25 张（本来就该走 chemicals 模板）。"""
    return [r for r in hi_rows if r["card"].get("chemicals")]


def _ask(row, **kw):
    """把一张卡当成 top-1 结果，跑完整的 compose（带来源与核实提示）。"""
    c = row["card"]
    return A.compose("测试问句", c["crop"], [{"chunk_id": row["chunk_id"]}], dry_run=True, **kw)


def test_两类高危卡都真实存在(nochem, withchem):
    """前置断言：语料里必须两类都有，否则下面这些用例等于没跑到。"""
    assert nochem, "语料里没有了无药剂的高危卡——本文件的用例失去意义，请检查解析产物"
    assert withchem, "语料里没有了带药剂的高危卡——附录B 解析可能坏了"


def test_无药剂高危卡不再渲染空用药方案(nochem):
    """本轮修的 bug：不许再出「可用下列药剂」+ 空的【用药方案】。"""
    for r in nochem:
        c = r["card"]
        out = A.render_high_risk([c], c["crop"])
        assert "可用下列药剂" not in out, f"{r['chunk_id']} 还在给不存在的药方"
        assert "【用药方案】" not in out, f"{r['chunk_id']} 渲染了空的【用药方案】"


def test_无药剂高危卡的正文必须逐字出现(nochem):
    """过去的症状是"正文一个字都不出来"，这条最要命——逐条核对全文。"""
    for r in nochem:
        c = r["card"]
        out = A.render_high_risk([c], c["crop"])
        texts = [(m.get("text") or "").strip() for m in c.get("measures") or []]
        texts = [t for t in texts if t]
        assert texts, f"{r['chunk_id']} 连 measures 都没有，渲染出来会是空的"
        for t in texts:
            assert t in out, f"{r['chunk_id']} 的原文条款没照抄进答案"


def test_章节号不再被当成病虫害名(nochem):
    """「【结论】黄瓜的7.1，可用下列药剂」——不再是「A的B，」这种结构。"""
    for r in nochem:
        c = r["card"]
        out = A.render_high_risk([c], c["crop"])
        head = out.splitlines()[0]
        assert f"的{c['subtype']}，" not in head, f"{r['chunk_id']} 又把章节号念成了标题"
        # 标题主语用卡自带的 knowledge_type，不新造词
        assert head.startswith(f"【结论】{c['crop']}的{c['knowledge_type']}要求如下"), head
        # 章节号降级成括号里的定位信息（仍可溯源，但不再是主语）
        if c["source"].get("section"):
            assert f"（第 {c['source']['section']} 条）" in head, head


def test_核实提示的内容跟着卡走(nochem, withchem):
    """两种措辞都必须带 VERIFY_HINT；无药剂时不许再说"剂量照抄"（名不副实）。"""
    for r in nochem:
        out = _ask(r)
        assert A.VERIFY_HINT in out, f"{r['chunk_id']} 漏了核实提示（违反核心约束 1）"
        assert "以上要求均照抄自国家标准原文" in out
        assert "安全间隔期均照抄自国家标准原文" not in out, f"{r['chunk_id']} 没有药剂却说剂量照抄"
    for r in withchem:
        out = _ask(r)
        assert A.VERIFY_HINT in out
        assert "农药名称、剂量、安全间隔期均照抄自国家标准原文" in out


def test_带药剂高危卡的渲染一字没变(withchem):
    """防回归：附录B 的用药方案路径不许被这次修改带歪（数值必须还逐字在答案里）。"""
    for r in withchem:
        c = r["card"]
        out = A.render_high_risk([c], c["crop"])
        assert "【用药方案】" in out, f"{r['chunk_id']} 丢了用药方案"
        for ch in c["chemicals"]:
            assert ch["product"] in out, f"{r['chunk_id']} 丢了药剂名 {ch['product']}"
            if ch.get("dose"):
                assert ch["dose"] in out, f"{r['chunk_id']} 丢了剂量"
            if ch.get("pre_harvest_interval"):
                assert ch["pre_harvest_interval"] in out, f"{r['chunk_id']} 丢了安全间隔期"


def test_高危卡不许过模型(nochem, monkeypatch):
    """条款型高危卡仍然由模板渲染，一个字都不能交给 DeepSeek 改写。

    手法：把 call_model 换成会抛异常的桩——真被调用到就红。
    """
    def _boom(*a, **k):
        raise AssertionError("高危卡不该调用模型")

    monkeypatch.setattr(A, "call_model", _boom)
    for r in nochem:
        assert _ask(r), f"{r['chunk_id']} 走了模型路径"


def test_precautions与measures同文时只念一遍(rows):
    """4.1.1 采购条款的 precautions 与 measures[0] 完全重复——必须去重。"""
    dup = [r for r in rows
           if (r["card"].get("precautions") or "").strip()
           and any((m.get("text") or "").strip() == r["card"]["precautions"].strip()
                   for m in r["card"].get("measures") or [])]
    assert dup, "语料里没有了 precautions 重复的卡，本用例失去意义"
    for r in dup:
        t = r["card"]["precautions"].strip()
        out = A.render_high_risk([r["card"]], r["card"]["crop"])
        assert out.count(t) == 1, f"{r['chunk_id']} 的同一段原文念了两遍"


def test_多张无药剂卡合并时按章节分小节(nochem):
    """同组多卡不能糊成一坨：每张卡先标章节号，再列条款。"""
    a, b = nochem[0]["card"], nochem[1]["card"]
    assert a["crop"] == b["crop"], "取两张同作物卡片才能测合并"
    out = A.render_high_risk([a, b], a["crop"])
    assert f"▸ {a['source']['section']}" in out
    assert f"▸ {b['source']['section']}" in out
    # 多卡时结论句不再挂单一章节号（不知道该挂谁的）
    assert out.splitlines()[0] == f"【结论】{a['crop']}的{a['knowledge_type']}要求如下"


def test_noplan提示仍然排在最前(withchem):
    """放宽场景：「暂无用药方案」必须排在答案最前面，农户第一眼看到。"""
    out = _ask(withchem[0], noplan=True)
    assert out.startswith(f"【注意】{A.NOPLAN_MSG}")


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


@pytest.fixture(scope="module")
def search_mod():
    """导入检索层（只为拿 normalize 与判据同源）。它按相对路径读 configs/，先切 CWD 到仓库根。"""
    old = os.getcwd()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src" / "retrieve"))
    try:
        import search
        yield search
    finally:
        os.chdir(old)


def test_口语病名问法答案零变化(sym_rows, search_mod):
    """点了**口语**病名的问法也必须零变化——判据要跑在归一化问句上（spec §5.1）。

    2026-09-28 最终审查实测：「大蒜小白虫打什么药」归一化后是「大蒜蓟马打什么药」，
    但判据此前拿的是**原始问句**——「小白虫」不在语料实体表里、过不了实体验证闸，
    于是被判成症状问法，农户点了病名却拿到对照口吻。
    上面那条 `test_病名问法答案零变化` 从卡 subtype 造问句，结构上测不到这个缺陷
    （subtype 本身就是标准名），所以这里单补一条口语的。
    """
    # 口语名要对上**真实挂了症状**的卡：大蒜「小白虫」→蓟马（c10）
    r = next((x for x in sym_rows
              if x["card"]["crop"] == "大蒜" and "蓟马" in (x["card"]["subtype"] or "")), None)
    assert r, "语料里没有大蒜蓟马症状卡，本用例失去意义（症状覆盖层变了？）"
    q = "大蒜小白虫打什么药"
    q2, _ = search_mod.normalize(q, "大蒜")
    assert "蓟马" in q2, f"口语应归一到病名（实际 {q2}）——归一化没生效则本用例测不到东西"
    out = A.compose(q, "大蒜", [{"chunk_id": r["chunk_id"]}], dry_run=True, nq=q2)
    assert "【症状对照】" not in out, f"点了口语病名却冒出了症状段：\n{out}"
    assert not out.startswith(A.SYM_HEAD), f"点了口语病名却用了对照口吻：\n{out}"


def test_症状判据与检索层同输入_口语病名家族(search_mod):
    """「小白虫→蓟马」「蒜蛆/黑蛆/根蛆→种蝇」这一族逐个过判据（喂归一化问句）。

    这几条正是最终审查实测的误判样本：口语名本身不在语料实体表里，
    拿原始问句去判 = 全部被判成症状问法。
    """
    for q in ("大蒜小白虫打什么药", "大蒜蒜蛆打什么药",
              "大蒜黑蛆打什么药", "大蒜根蛆打什么药"):
        q2, _ = search_mod.normalize(q, "大蒜")
        assert not A._is_symptom_query(q2), f"{q} 归一化后（{q2}）仍被判成症状问法"


def test_症状段点出病虫害名(sym_rows):
    """每条症状标出病虫害名 + 结论句点出卡片主语（2026-09-28 用户裁决，spec §5.2）。

    此前症状段只念症状文本：问「辣椒茎基黑褐斑…是什么病」，「疫病」一次都不出现；
    问「蒜苗白色弯曲斑道」，「豌豆潜叶蝇」一次都不出现——农户的问题等于没被回答。
    主语**只许来自症状条目自己的 `pest`**，不许拿卡的 subtype 顶：正文卡的 subtype
    是章节号（b6.2 → "6.2"），念出来就是「辣椒 6.2」这种垃圾。
    """
    for r in sym_rows:
        c = r["card"]
        out = A.compose("叶子上有白色的虫道，是什么虫", c["crop"],
                        [{"chunk_id": r["chunk_id"]}], dry_run=True)
        assert "【症状对照】" in out, f"{r['chunk_id']} 没渲染症状段"
        pests = []
        for s in r["symptoms"]:
            assert f"【{s['pest']}】" in out, f"{r['chunk_id']} 的症状条目没标出「{s['pest']}」"
            if s["pest"] not in pests:
                pests.append(s["pest"])
        assert f"（{c['crop']} {'、'.join(pests)}）" in out, \
            f"{r['chunk_id']} 结论句没点出卡片主语（应含「{c['crop']} {'、'.join(pests)}」）"
        # subtype 与病虫害名不一致时（正文卡就是这种），点名的必须是 pest 不是 subtype
        if c["subtype"] != "、".join(pests):
            assert f"（{c['crop']} {c['subtype']}）" not in out, \
                f"{r['chunk_id']} 结论句念的是卡的 subtype（{c['subtype']}），不是症状条目的病虫害名"


def test_症状提示只在症状段渲染时才加(sym_rows):
    """「症状描述来自外部文献…」这句**只在症状段真的渲染了**才许出现。

    与「没有药剂时不许说剂量照抄」同一条原则：措辞必须跟着内容走。
    这里 compose 两次——症状问法（该有）/ 口语病名问法（不该有）——
    谁把条件句改回无条件挂上，第二次就红。
    """
    r = next((x for x in sym_rows if x["card"]["needs_verification_hint"]), None)
    assert r, "语料里没有高危症状卡，本用例失去意义"
    c = r["card"]
    out_sym = A.compose("叶子上有白色的虫道，是什么虫", c["crop"],
                        [{"chunk_id": r["chunk_id"]}], dry_run=True)
    assert "【症状对照】" in out_sym
    assert "症状描述来自外部文献" in out_sym, "症状段渲染了却没挂那句来源提示"

    # 病名问法不吃覆盖层 → 同一张卡也不许挂那句（挂了就是不实）
    pest = (c.get("subtype") or "").split("、")[0].strip()
    out_by = A.compose(f"{c['crop']}{pest}打什么药", c["crop"],
                       [{"chunk_id": r["chunk_id"]}], dry_run=True)
    assert "【症状对照】" not in out_by
    assert "症状描述来自外部文献" not in out_by, "没渲染症状段却挂了症状来源提示"


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


def test_top1是普通卡时不许渲染别家的用药方案(sym_rows, rows):
    """变更 A（2026-09-29 用户裁决）：top-1 是普通卡 → **一张用药方案都不渲染**。

    本条原名 `test_症状段永远排在任何用药方案之前`，断言的是「症状段排在【用药方案】之前」。
    那条断言已被变更 A 取代：top-1（普通卡）不属于任何高危组，正确行为是**没有**
    【用药方案】——正是 collected #18 的验收口径「b6.2 正文 + **不带别家药方**」
    （top-1 已是正文卡 b6.2，c03 早疫/晚疫、c05 茶黄螨 仍在第 2、3 位，
    旧口径下农户照样拿到别家药方）。症状段本身照旧必须渲染。
    """
    r = next(x for x in sym_rows if not x["card"]["needs_verification_hint"])
    chem = [x for x in rows if x["card"].get("chemicals")]
    assert chem, "语料里没有带药卡，本用例失去意义"
    other = next(c for c in chem if c["card"]["crop"] == r["card"]["crop"])
    out = A.compose("叶子上有白色的虫道，是什么虫", r["card"]["crop"],
                    [{"chunk_id": r["chunk_id"]}, {"chunk_id": other["chunk_id"]}],
                    dry_run=True)
    assert "【症状对照】" in out, out
    assert "【用药方案】" not in out, f"top-1 是普通卡，却渲染了别家用药方案：\n{out}"
    assert other["card"]["chemicals"][0]["product"] not in out, \
        f"别家病虫害的药剂名（{other['card']['chemicals'][0]['product']}）漏进了答案：\n{out}"


def test_top1是高危卡时只渲染它那一组(withchem):
    """变更 A 的另一半：top-1 是高危卡 → 只渲染**它那一组**，别组不进答案。

    组口径＝(作物, 病虫害)：同组多卡（不同防治适期）仍然并排渲染（防回归），
    不同病虫害的组一律不渲染——「只渲染 top-1 所在那一组」的字面含义。
    """
    a = withchem[0]
    other = next(r for r in withchem
                 if r["card"]["crop"] == a["card"]["crop"]
                 and r["card"]["subtype"] != a["card"]["subtype"])
    out = A.compose("测试问句", a["card"]["crop"],
                    [{"chunk_id": a["chunk_id"]}, {"chunk_id": other["chunk_id"]}],
                    dry_run=True)
    assert "【用药方案】" in out, f"{a['chunk_id']} 自己的用药方案丢了：\n{out}"
    for ch in a["card"]["chemicals"]:
        assert ch["product"] in out, f"本组药剂 {ch['product']} 丢了"
    # 别组药剂名：先确认两组没有同名药剂，否则这条断言会假红
    shared = {ch["product"] for ch in a["card"]["chemicals"]} & \
             {ch["product"] for ch in other["card"]["chemicals"]}
    assert not shared, f"两组药剂同名（{shared}），本用例需要换一张卡来测"
    for ch in other["card"]["chemicals"]:
        assert ch["product"] not in out, \
            f"别组药剂 {ch['product']} 漏进了答案（变更 A 后不该渲染第二组）：\n{out}"
