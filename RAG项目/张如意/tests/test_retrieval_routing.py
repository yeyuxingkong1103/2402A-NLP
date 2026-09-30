# -*- coding: utf-8 -*-
"""检索层回归：路由劫持与实体名↔卡 subtype 不一致（2026-09-23）。

这两类 bug 的共同后果是**误兜底/答非所问**：
  ① 意图路由把「点了名」的病虫害自己的卡筛在门外——
     「辣椒苗期猝倒病怎么防治」被「怎么防」路由到 6.x 防治原则，绕开表B.1 的 c01；
     「防病治虫的人得有啥证」被「防病治虫」路由到 6.x，把 3.2.6.2 植保员条款筛掉
     （去掉路由后同一张卡 rerank 0.9429，路由过滤下只有 0.0585）。
  ② 农户口语的病虫害名与表B.1 的 subtype 不一致——
     大蒜说「潜叶蝇」（卡是「豌豆潜叶蝇」）→ 被判「语料中大蒜无此病虫害」硬兜底；
     黄瓜说「蓟马」（卡是「棕榈蓟马」）→ 判 missing。
  ③（2026-09-29）**症状问法判据**与重排 passage 的接线——判据一旦被改回
     「只看有没有病虫害实体」，问施药规矩的技术问句会重新被当成症状描述，
     方案卡的症状文本会把附录B 池顶过闸门（实测 auto 高危 53/53→52/53）；
     见本节末尾「③ 症状问法判据」一段。

**本文件的桩只锁控制流与判据接线**（哪个候选池、什么时候重试、重试失败是否仍兜底、
症状拼不拼进 passage），真实分数行为（RRF 融合、reranker 闸门）由 `eval/run_eval.py`
全量验证——它是本项目的回归网，改动检索层后必须重跑并对比 report.json（漏兜底必须仍为 0）。
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def S():
    """导入 search 模块。它按相对路径读 configs/ 与 data/chunks/，所以要先把 CWD 切到仓库根。"""
    old = os.getcwd()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src" / "retrieve"))
    try:
        import search
        yield search
    finally:
        os.chdir(old)


# ---------------------------------------------------------------- 桩 Searcher
class _Stub:
    """桩 Searcher：按 extra_expr 返回不同候选池，模拟 Milvus 的过滤效果。

    候选里的 `_score` 就是 rerank 分——真实 reranker 逐对精算，这里按池子写死，
    以便把「路由过滤下多少分、去掉路由后多少分」这一实测事实固化进用例。
    """

    def __init__(self, pool):
        self.pool = pool          # callable(extra_expr) -> [候选]
        self.calls = []           # 记录每次 hybrid 的 extra_expr，供断言「有没有重试」

    def hybrid(self, q, crop=None, topk=5, appendix_only=False, text_only=False,
               extra_expr=None):
        self.calls.append({"extra_expr": extra_expr, "text_only": text_only})
        return [dict(c) for c in self.pool(extra_expr)]

    def rerank_gate(self, q, cands):
        if not cands:
            return 0.0, None
        best = max(cands, key=lambda c: c["_score"])
        return best["_score"], best["chunk_id"]

    def rerank_sort(self, q, cands):
        """与真实 Searcher.rerank_sort 同口径：按分数降序返回 (最高分, 排序后候选)。"""
        if not cands:
            return 0.0, []
        ranked = sorted(cands, key=lambda c: c["_score"], reverse=True)
        return ranked[0]["_score"], ranked


def _card(chunk_id, subtype, score):
    """构造一个最小候选卡片：卡号 + 病虫/条款 subtype + rerank 分数（桩用）。"""
    return {"chunk_id": chunk_id, "subtype": subtype, "_score": score}


# ---------------------------------------------------------------- ① 路由劫持
def test_点名病虫害时_路由过滤不能把它的卡排除在外(S):
    """「辣椒苗期猝倒病怎么防治」→ 意图「怎么防」命中 control_principle。

    路由只放行 6.1/6.3/6.4/6.5（防治原则/农业防治/生物防治/物理防治），
    表B.1 的 c01《猝倒病、立枯病》进不了候选池，narrow 收窄不到东西只能退回原则条款。
    修法：路由表达式 **并上「点名实体自己的卡」**，不放弃路由。
    """
    def pool(extra_expr):
        # 路由表达式本身只放行 6.x 原则条款；并上实体子句后 c01 才进得来
        cards = [_card("GBZ26583-2011-p6-b6.1", "6.1", 0.50)]
        if extra_expr and "猝倒病" in extra_expr:
            cards.insert(0, _card("GBZ26583-2011-p20-c01", "猝倒病、立枯病", 0.90))
        return cards

    st, q2, intent, res, why = S.search(_Stub(pool), "辣椒苗期猝倒病怎么防治？", crop="辣椒")

    assert [r["chunk_id"] for r in res] == ["GBZ26583-2011-p20-c01"], \
        f"点了名的病虫害，它自己的卡必须在结果里（实际 {[r['chunk_id'] for r in res]}，{why}）"


def test_具体知识类型的路由_不被点名病虫害的卡挤掉(S):
    """「热水烫种防炭疽病」问的是**怎么操作**（物理防治 6.5.6 温汤浸种），不是炭疽病用药方案。

    并实体卡只该给「控制原则」这种泛问法用（它的模式就是「怎么防/防病治虫」这些泛词，
    容易劫持别的问题）。物理防治/主要防治对象这类路由是农户明确点名的知识类型，
    可信任——并进实体卡反而会被 narrow 把路由选中的卡挤掉（实测 auto 普通 −2、collected 普通 −1）。
    """
    def pool(extra_expr):
        cards = [_card("GBZ26581-2011-p6-b6.5.6", "6.5.6", 0.90)]      # 温汤浸种
        if extra_expr and "炭疽病" in extra_expr:
            cards.append(_card("GBZ26581-2011-p20-c04", "炭疽病", 0.95))
        return cards

    st, q2, intent, res, why = S.search(
        _Stub(pool), "黄瓜种子用五六十度的热水烫一下再泡温水，真能防炭疽病、角斑病和枯萎病这些毛病吗？具体咋操作啊？",
        crop="黄瓜")
    assert "GBZ26581-2011-p6-b6.5.6" in [r["chunk_id"] for r in res], \
        f"物理防治路由选中的卡不应被实体卡挤掉（实际 {[r['chunk_id'] for r in res]}，{why}）"


def test_问有哪些病虫害时_回主要防治对象而不是杂草方案(S):
    """「主要得防哪些病虫害和杂草」问的是**清单**（6.2 主要防治对象），不是杂草的用药方案。"""
    def pool(extra_expr):
        cards = [_card("GBZ26578-2011-p6-b6.2", "6.2", 0.90)]          # 主要防治对象清单
        if extra_expr and "杂草" in extra_expr:
            cards.append(_card("GBZ26578-2011-p20-c04", "杂草", 0.95))
        return cards

    st, q2, intent, res, why = S.search(
        _Stub(pool), "种大蒜的时候，主要得防哪些病虫害和杂草啊？", crop="大蒜")
    assert "GBZ26578-2011-p6-b6.2" in [r["chunk_id"] for r in res], \
        f"问清单应回 6.2，不是杂草方案（实际 {[r['chunk_id'] for r in res]}，{why}）"


def test_路由误判时_冠军营救_不再误兜底(S):
    """「种黄瓜的话，打药防病治虫的人得有啥证、受过啥培训才行啊？」

    实测：路由过滤到 6.x → rerank 0.0585 误兜底；去掉路由 → 同一张 3.2.6.2 卡 0.9429。
    2026-09-27 起由「冠军营救」处理：全库池 rerank 冠军若是正文卡且比路由池冠军
    高出 0.05 以上，判为路由劫持，直接把冠军插到队首——比原先「闸门失败后去路由
    重试」更早生效（重试机制仍保留为营救未触发时的后备）。
    """
    def pool(extra_expr):
        if extra_expr is not None:      # 被「防病治虫」路由到 6.x，把植保员条款筛掉了
            return [_card("GBZ26581-2011-p6-b6.5.1", "6.5.1", 0.0585)]
        return [_card("GBZ26581-2011-p4-b3.2.6.2", "3.2.6.2", 0.9429)]

    s = _Stub(pool)
    st, q2, intent, res, why = S.search(s, "种黄瓜的话，打药防病治虫的人得有啥证、受过啥培训才行啊？", crop="黄瓜")

    assert st == "ok", f"营救应直接命中（实际 {st}: {why}）"
    # 营救把冠军插到队首，路由池其余结果保留在后面
    assert [r["chunk_id"] for r in res][0] == "GBZ26581-2011-p4-b3.2.6.2"
    assert len(s.calls) == 2, "应恰好检索两次（带路由 1 次 + 营救全库 1 次）"


def test_去掉路由重试仍不过闸门时_照旧兜底(S):
    """重试不是「一定给答案」：去掉路由仍不过闸门 → 照旧说「暂无」。

    防的是修 ① 时把闸门一并放水——那是漏兜底（给农户错的药），比误兜底更危险。
    """
    def pool(extra_expr):
        return [_card("GBZ26581-2011-p6-b6.5.1", "6.5.1", 0.06)]

    st, q2, intent, res, why = S.search(_Stub(pool), "种黄瓜的话，打药防病治虫的人得有啥证、受过啥培训才行啊？", crop="黄瓜")

    assert st == "fallback_lowscore" and res == []
    # search() 返回五元组：(状态位, 归一化问句, 意图, 结果列表, 判定原因)


# ---------------------------------------------------------------- ② 实体名↔卡 subtype
def test_大蒜潜叶蝇归一到豌豆潜叶蝇(S):
    """大蒜 6.2 与表B.1 都写「豌豆潜叶蝇」，农户口语说「潜叶蝇」。

    不打通 → coverage 判「语料中大蒜无此病虫害」→ 硬兜底，
    而语料里明明有 c09《豌豆潜叶蝇》方案卡。
    """
    out, hits = S.normalize("大蒜潜叶蝇怎么防？", "大蒜")
    assert "豌豆潜叶蝇" in out, f"大蒜的潜叶蝇应归一到豌豆潜叶蝇（实际 {out}）"

    st, q2, intent, res, why = S.search(
        _Stub(lambda e: [_card("GBZ26578-2011-p20-c09", "豌豆潜叶蝇", 0.9)]),
        "大蒜潜叶蝇怎么防？", crop="大蒜")
    assert st != "fallback_nocoverage", f"有卡就不该硬兜底（实际 {st}: {why}）"


def test_黄瓜潜叶蝇仍归一到美洲斑潜蝇(S):
    """黄瓜侧已有条目不能被新加的大蒜条目抢走（同词不同作物的回归闸）。"""
    out, _ = S.normalize("黄瓜叶子上有潜叶蝇", "黄瓜")
    assert "美洲斑潜蝇" in out and "豌豆潜叶蝇" not in out, f"实际 {out}"


def test_黄瓜蓟马归一到棕榈蓟马(S):
    """黄瓜 6.2.2 与表B.1 都写「棕榈蓟马」，农户说「蓟马」按它召回。

    **人工定档 2026-09-23（选 A）**：黄瓜语料里只有这一个种，走兜底会让农户觉得
    系统连蓟马都不知道。是「属→种」推断（不是确定性改名），故：
      - 答案里必须出现具体种名——卡 subtype 就是「棕榈蓟马」，模板照卡渲染即满足；
      - auto 集旧期望 `黄瓜蓟马打什么药 → expect_fallback=True` 同步翻转（见 eval/datasets/auto.jsonl）。
    """
    out, _ = S.normalize("黄瓜蓟马打什么药", "黄瓜")
    assert "棕榈蓟马" in out, f"黄瓜的蓟马应归一到棕榈蓟马（实际 {out}）"

    st, q2, intent, res, why = S.search(
        _Stub(lambda e: [_card("GBZ26581-2011-p20-c07", "棕榈蓟马", 0.90)]),
        "黄瓜蓟马打什么药", crop="黄瓜")
    assert [r["chunk_id"] for r in res] == ["GBZ26581-2011-p20-c07"], \
        f"应召回棕榈蓟马方案卡（实际 {st}: {why}）"


def test_换茬连作重茬_归一到轮作(S):
    """农户说「换茬/连作/重茬」，语料里的规范表述是「轮作」（三份正文都有 6.3.1.2 轮作）。

    俗称本身语料 0 命中（与「蒜蛆→种蝇」同类：目标词是正文里的规范表述即可）。
    实测反事实：这两条 collected 失败问句归一后 b6.3.1.2 都排第 1（0.5678 / 0.8909，过闸门）；
    **加意图路由到农业防治反而更差**——辣椒池内 0.0759 < 闸门 0.1，会从「未召回」变误兜底。
    """
    cases = [
        ("换茬", "辣椒", "上一茬种辣椒的地能接着种辣椒吗？还是得换茬？"),
        ("连作", "辣椒", "这块地连作辣椒两年了，还能接着种吗？"),
        ("重茬", "大蒜", "大蒜重茬种行不行？"),
    ]
    for spoken, crop, q in cases:
        out, _ = S.normalize(q, crop)
        assert "轮作" in out and spoken not in out, f"{spoken} 应归一到轮作（实际 {out}）"


def test_大蒜蓟马不被黄瓜条目改写(S):
    """大蒜表B.1 单列「蓟马」（卡 subtype 就是「蓟马」）——映射必须按作物分开。"""
    out, _ = S.normalize("大蒜蓟马打什么药", "大蒜")
    assert out == "大蒜蓟马打什么药", f"大蒜的卡就叫「蓟马」，不该被黄瓜条目改写（实际 {out}）"


# ---------------------------------------------------------------- ③ 症状问法判据
# 2026-09-29 用户裁决：**症状问法 ＝ 无语料实体「且」无意图**（`search.is_symptom_query`）。
# 下面这组锁专治收窄这一轮的回归：判据一旦被改回「只看实体」，
# 问施药规矩的技术问句会重新被当成症状描述 → 方案卡的症状文本把附录B 池顶过闸门
# （实测 c05 池内最高分 0.0754→0.1156>0.1）→ 农户问「要不要轮换用药」拿到 5 张药方，
# 实测 auto 高危 53/53→52/53、MRR 50.9→49.9。
# 全部秒级、不加载模型（只读 configs/ + data/chunks/chunks.jsonl）。
_轮换用药问句 = "种辣椒打药的时候，是不是得换着用不同的药，免得虫子产生抗药性啊？"
_描述现象问句 = "蒜苗叶子上有一条条白色弯曲的斑道，是什么危害？"


def test_判据_技术问句不算症状问法(S):
    """「问要不要轮换用药」**没有病虫害实体，却被 chemical_control（「打药」）认领**。

    它问的是**施药规矩**（正文 6.3.4.4），不是症状描述——**意图是"这是个问句、
    不是一段描述"的证据**。判据只看实体的话，这类问句会拿到症状加成。
    """
    q2, _ = S.normalize(_轮换用药问句, "辣椒")
    # 前提先钉住：它确实**没有**语料实体、**有**意图——这正是判据两条件的判别面
    assert S.extract_entities(q2) & (S._MENTIONED | set(S._CARD_CROPS)) == set(), \
        f"本用例的前提是该问句无语料实体（实际 {q2}）"
    assert S.detect_intent(q2) is not None, f"本用例的前提是该问句有意图（实际 {q2}）"
    assert S.is_symptom_query(q2) is False, \
        f"问施药规矩的技术问句不该判成症状问法（实际 {q2}）"


def test_判据_描述现象的问句算症状问法(S):
    """农户描述现象：既说不出病名、也不带任何问法模式 → 两条件都成立。"""
    q2, _ = S.normalize(_描述现象问句, "大蒜")
    assert S.extract_entities(q2) & (S._MENTIONED | set(S._CARD_CROPS)) == set()
    assert S.detect_intent(q2) is None, f"本用例的前提是该问句无意图（实际 {q2}）"
    assert S.is_symptom_query(q2) is True, \
        f"描述现象的问句应判成症状问法（实际 {q2}）"


def test_passage_不拼症状时与卡本体逐字节相同(S):
    """198 张无症状卡（以及所有非症状问句）全靠这条：开关为 False 时
    `_rerank_passage` 必须返回 `bm25_text[:200]` 原串——多一个空格都会让它们的分数漂移。
    """
    cases = [
        {"bm25_text": "卡本体" * 100, "sym_text": "蛇形虫道" * 20},   # 两者都有（超 200 字）
        {"bm25_text": "卡本体", "sym_text": ""},                      # 无 症状
        {"bm25_text": "", "sym_text": "蛇形虫道"},                    # 空卡本体
        {"bm25_text": None, "sym_text": None},                        # 脏数据
        {},                                                           # 缺键
    ]
    for c in cases:
        want = (c.get("bm25_text") or "")[:200]
        got = S._rerank_passage(c, False)
        assert got == want, f"不拼症状时必须逐字节等于卡本体（{got!r} != {want!r}）"


def test_passage_拼症状时是追加不前置(S):
    """症状**追加**在卡本体之后：卡本体那段仍逐字保持，且症状文本在尾部。"""
    c = {"bm25_text": "卡本体" * 100, "sym_text": "白色弯曲的蛇形虫道"}
    out = S._rerank_passage(c, True)
    assert out.startswith(c["bm25_text"][:200])
    assert out.endswith("白色弯曲的蛇形虫道")
    # 没有症状文本的卡，开关开不开都一样（症状层只覆盖 5 张卡）
    assert S._rerank_passage({"bm25_text": "只有卡本体", "sym_text": ""}, True) == "只有卡本体"


class _StubReranker:
    """桩交叉编码器：只记录收到的 (query, passage) 对，不加载模型（秒级）。"""

    def __init__(self):
        self.pairs = None

    def compute_score(self, pairs, normalize=True):
        self.pairs = pairs
        return [0.5] * len(pairs)


def test_接线_rerank_sort按判据决定拼不拼症状(S):
    """判据对了但没接线＝没改（症状层会一整层失效）。绕过 __init__ 避免加载模型。"""
    s = S.Searcher.__new__(S.Searcher)      # 不跑 __init__：不连 Milvus、不载 BGE/reranker
    s.reranker = _StubReranker()
    cand = [{"chunk_id": "GBZ26578-2011-p20-c09", "bm25_text": "卡本体",
             "sym_text": "白色弯曲的蛇形虫道"}]

    # 症状问句 → passage 必须带上症状文本
    q2, _ = S.normalize(_描述现象问句, "大蒜")
    assert S.is_symptom_query(q2)
    s.rerank_sort(q2, cand)
    assert len(s.reranker.pairs) == 1
    assert "蛇形虫道" in s.reranker.pairs[0][1], "症状问句的 passage 必须含症状文本"

    # 技术问句 → passage 里一个症状字都不许有，且逐字节等于卡本体
    q2, _ = S.normalize(_轮换用药问句, "辣椒")
    assert not S.is_symptom_query(q2)
    s.rerank_sort(q2, cand)
    assert s.reranker.pairs[0][1] == "卡本体", \
        f"非症状问句的 passage 不该含症状（实际 {s.reranker.pairs[0][1]!r}）"


def test_结构_两处闸门都走同一个passage构造点(S):
    """`rerank_gate` / `rerank_sort` 历史上**各写了一遍** passage 构造，各改各的就漂了。

    现在两处都必须调 `_rerank_passage`，且都不许再自己切 `bm25_text`——用源码做结构断言，
    比行为断言更直接（行为测试可能因为两条路径恰好同分而漏判）。
    """
    import inspect
    src = inspect.getsource(S)
    assert src.count("def _rerank_passage(") == 1, "passage 构造点必须唯一"
    assert src.count("_rerank_passage(") == 3, \
        "`_rerank_passage` 应恰好 3 处出现：1 处定义 + rerank_gate / rerank_sort 各 1 处调用"
    for name in ("rerank_gate", "rerank_sort"):
        body = inspect.getsource(getattr(S.Searcher, name))
        assert "_rerank_passage(" in body, f"{name} 必须走唯一的 passage 构造点"
        assert 'bm25_text"][:200]' not in body, f"{name} 不许再自己切 bm25_text"
