# -*- coding: utf-8 -*-
"""Neo4j 图谱召回（多路召回的「路 3」）。

为什么是「规则抽取」而不是「LLM 抽取」
--------------------------------------
法条语料结构高度规整，规则抽取**准确率接近 100% 且零成本**；
LLM 抽取要对 14319 条逐条调用，耗时且会引入不可控的幻觉边。

⚠️ 但必须如实说明：规则的**覆盖率实测远低于设计时的预期**
---------------------------------------------------------
设计文档 §3.4 假设「依据《X》第Y条」是强模式。**实测该模式在本语料中不存在**
（覆盖 0/1500）。剥掉每条 chunk 开头的自我标识 `《本法名》第X条规定，` 之后，
真正的跨法引用只覆盖 **1.9%**（29/1500，共 38 处）。

因此本图建**两类边**：

    1. CITES     跨法引用 —— 稀疏（1.9%）但精确
                 例：《数据安全法》第31条「适用《网络安全法》的规定」
    2. ADJACENT  同法相邻条文 —— 覆盖 100%（176 部法律条文编号全部连续）
                 例：民法典第577条 ↔ 第578条

**这条路径的实际增益必须实测**（见 `scripts/eval_multirecall.py`），
不能因为「图上跑通了」就当成有收益 —— 向量检索本身可能已经召回了相邻条文。

图模型
------
    (:Article {uid, law_name, article_no, num, collection})
    (:Article)-[:CITES]->(:Article)
    (:Article)-[:ADJACENT]->(:Article)

`uid = "{law_name}|{article_no}"` 是主键，保证重复构建幂等。
"""
import re
import threading
import time
from collections import defaultdict

from neo4j import GraphDatabase

from ..core import config
from ..core.logging import get_logger

log = get_logger("graph_store")

_driver = None
_lock = threading.Lock()

# 中文数字的字符类。⚠️ 必须含「千」—— 「第一千一百四十七条」这类大条号
# 在民法典里占很大比例（民法典共 1260 条）。漏掉「千」会让**所有 1000 条以上
# 的法条静默解析失败**（2026-09-17 实测踩过：图谱路对第1147条、第1223条
# 完全失效，而节点其实存在于图中）。
_CN = "零〇一二三四五六七八九十百千"

# chunk 开头的自我标识：《本法名》第X条规定，……
# 必须剥掉，否则每条 chunk 都会「引用自己」，把图污染成 14319 条自环边。
_SELF_PREFIX = re.compile(rf"^《[^》]{{2,40}}》\s*第[{_CN}\d]+条\s*[，,：:]?\s*(?:规定[，,：:]?\s*)?")

# 正文中的《书名》
_CITE = re.compile(r"《([^》]{2,40})》")

# 「第X条」
_ART_NO = re.compile(rf"第([{_CN}\d]+)条")

_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def cn_to_int(s: str | None) -> int | None:
    """中文数字 -> int。「第十八条」「十八」「18」都能解析。"""
    if not s:
        return None
    s = str(s).strip().lstrip("第").rstrip("条").replace("〇", "零")
    if not s:
        return None
    if s.isdigit():
        return int(s)
    total = section = num = 0
    for ch in s:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        elif ch == "十":
            section += (num or 1) * 10
            num = 0
        elif ch == "百":
            section += (num or 1) * 100
            num = 0
        elif ch == "千":
            section += (num or 1) * 1000
            num = 0
        else:
            return None
    return total + section + num


def uid_of(law_name: str, article_no: str) -> str:
    """法条的唯一键。"""
    return f"{law_name or ''}|{article_no or ''}"


# ---------------------------------------------------------------- 连接
def get_driver():
    global _driver
    if _driver is None:
        with _lock:
            if _driver is None:
                log.info("连接 Neo4j: %s", config.NEO4J_URI)
                _driver = GraphDatabase.driver(
                    config.NEO4J_URI,
                    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        try:
            _driver.close()
        except Exception:
            pass
        _driver = None


def health() -> dict:
    t = time.time()
    try:
        d = get_driver()
        d.verify_connectivity()
        with d.session() as s:
            n = s.run("MATCH (a:Article) RETURN count(a) AS n").single()["n"]
        return {"ok": True, "ms": round((time.time() - t) * 1000), "articles": n}
    except Exception as e:
        return {"ok": False, "ms": round((time.time() - t) * 1000),
                "error": f"{type(e).__name__}: {str(e)[:150]}"}


# ---------------------------------------------------------------- 抽取
def extract_citations(law_name: str, text: str) -> list[str]:
    """从一条 chunk 的正文里抽出它引用的**其它**法律名。

    先剥掉开头的自我标识，否则每条都会引用自己。
    """
    body = _SELF_PREFIX.sub("", text or "", count=1)
    out = []
    for m in _CITE.finditer(body):
        name = m.group(1)
        if name and name != law_name:
            out.append(name)
    # 去重且保序
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


# ---------------------------------------------------------------- 建图
def build_graph(collection: str, batch: int = 500) -> dict:
    """从向量库读取 chunk，构建图谱。

    返回统计信息（含各类型边的实际条数 —— 用于如实报告覆盖率）。
    """
    from . import milvus_store
    from .retrieval import collection_names

    if config.VECTOR_STORE == "milvus":
        rows = _scroll_milvus(collection)
    else:
        rows = _scroll_qdrant(collection)

    if not rows:
        return {"articles": 0, "cites": 0, "adjacent": 0,
                "note": f"{collection} 无数据"}

    # ⚠️ 医疗语料（PDF 指南）的 chunk **没有 law_name / article_no 字段**，
    #    因此建不出节点 —— 图谱召回**对 kb_medical 不适用**。
    #    这不是 bug，是语料结构差异：法律语料按「第X条」切分并带条文号，
    #    指南语料按段落切分、没有可做图的层级标识。
    #    如实返回 note，避免上层以为「建了图但没生效」。
    if not any(r.get("law_name") and r.get("article_no") for r in rows):
        log.warning("%s 无 law_name/article_no 字段，图谱召回不适用", collection)
        return {"articles": 0, "cites": 0, "adjacent": 0,
                "note": f"{collection} 的 chunk 无条文号，不适用图谱"}

    driver = get_driver()
    cites = adjacent = 0

    with driver.session() as s:
        # 清空该 collection 的旧图（幂等重建）
        s.run("MATCH (a:Article {collection:$c}) DETACH DELETE a", c=collection)

        # ---- 节点 ----
        nodes = {}
        for r in rows:
            law, art = r.get("law_name"), r.get("article_no")
            if not law or not art:
                continue
            u = uid_of(law, art)
            nodes[u] = {"uid": u, "law_name": law, "article_no": art,
                        "num": cn_to_int(art) or 0, "collection": collection}
        s.run(
            "UNWIND $rows AS row MERGE (a:Article {uid: row.uid}) "
            "SET a.law_name=row.law_name, a.article_no=row.article_no, "
            "    a.num=row.num, a.collection=row.collection",
            rows=list(nodes.values()),
        )
        log.info("%s 节点写入 %d 个", collection, len(nodes))

        # ---- CITES 边（跨法引用）----
        cite_pairs = set()
        for r in rows:
            law, art = r.get("law_name"), r.get("article_no")
            if not law or not art:
                continue
            for target in extract_citations(law, r.get("text") or ""):
                cite_pairs.add((uid_of(law, art), target))
        if cite_pairs:
            s.run(
                "UNWIND $pairs AS p "
                "MATCH (a:Article {uid: p[0]}) "
                "MERGE (b:LawRef {name: p[1], collection: $c}) "
                "MERGE (a)-[:CITES]->(b)",
                pairs=[list(p) for p in cite_pairs], c=collection,
            )
            cites = len(cite_pairs)

        # ---- ADJACENT 边（同法相邻条文）----
        by_law = defaultdict(list)
        for n in nodes.values():
            by_law[n["law_name"]].append(n)
        adj_pairs = []
        for law, arts in by_law.items():
            arts.sort(key=lambda x: x["num"])
            for a, b in zip(arts, arts[1:]):
                if b["num"] == a["num"] + 1:      # 只在真正相邻时连边
                    adj_pairs.append([a["uid"], b["uid"]])
                    adjacent += 1
        if adj_pairs:
            # ⚠️ 必须写成两个独立 MATCH，不能写成 `MATCH (a), (b)` ——
            #    后者是笛卡尔积（Neo4j 会报 03N90 性能警告），1.4 万条边时实测慢一倍以上。
            s.run(
                "UNWIND $pairs AS p "
                "MATCH (a:Article {uid: p[0]}) "
                "MATCH (b:Article {uid: p[1]}) "
                "MERGE (a)-[:ADJACENT]->(b)",
                pairs=adj_pairs,
            )

    stats = {"articles": len(nodes), "cites": cites, "adjacent": adjacent,
             "laws": len(by_law)}
    log.info("%s 建图完成 | 节点 %d | CITES %d | ADJACENT %d | 涉及 %d 部法律",
             collection, stats["articles"], cites, adjacent, stats["laws"])
    return stats


def _scroll_milvus(collection: str, cap: int = 30000) -> list[dict]:
    """从 Milvus 读取全部 chunk 的文本与元数据（不含向量）。"""
    from . import milvus_store
    c = milvus_store.get_client()
    if not c.has_collection(collection):
        return []
    out, offset = [], 0
    fields = ["text", "law_name", "article_no"]
    while len(out) < cap:
        res = c.query(collection_name=collection, filter="",
                      output_fields=fields, limit=2000, offset=offset)
        if not res:
            break
        out.extend(res)
        if len(res) < 2000:
            break
        offset += len(res)
    return out


def _scroll_qdrant(collection: str, cap: int = 30000) -> list[dict]:
    from .qdrant_store import get_client
    c = get_client()
    if not c.collection_exists(collection):
        return []
    out, off = [], None
    while len(out) < cap:
        pts, off = c.scroll(collection_name=collection, limit=500,
                            offset=off, with_payload=True, with_vectors=False)
        if not pts:
            break
        out.extend([p.payload or {} for p in pts])
        if off is None:
            break
    return out


# ---------------------------------------------------------------- 召回
def graph_recall(query: str, collection: str | None = None, k: int = 10) -> list[dict]:
    """路 3：从 query 里识别法律实体，在图谱中做**精确**召回。

    与向量检索互补：向量检索擅长语义相似，但**不擅长精确定位**
    （问「民法典第五百七十七条」时，向量检索未必把该条排第一，而图谱是精确命中）。

    识别两类实体：
        《法名》       -> 该法被引用/引用的条文
        第X条          -> 配合 query 中出现的法名，或全局按条号匹配
    """
    driver = get_driver()
    hits: list[dict] = []

    law_names = [m.group(1) for m in _CITE.finditer(query)]
    art_nums = [n for n in (_ART_NO.findall(query))]
    art_ints = [x for x in (cn_to_int(a) for a in art_nums) if x]

    with driver.session() as s:
        # ① 指名道姓「《X》第Y条」-> 精确取该条
        if law_names and art_ints:
            for law in law_names:
                for num in art_ints[:3]:
                    res = s.run(
                        "MATCH (a:Article {collection:$c}) "
                        "WHERE a.law_name CONTAINS $law AND a.num = $num "
                        "OPTIONAL MATCH (a)-[:ADJACENT]->(nb:Article) "
                        "OPTIONAL MATCH (a)-[:CITES]->(lr:LawRef) "
                        "RETURN a, collect(DISTINCT nb.uid) AS nb, "
                        "       collect(DISTINCT lr.name) AS refs LIMIT $k",
                        c=collection, law=law, num=num, k=k,
                    )
                    hits.extend(_rows_to_hits(res, "法条精确命中"))

        # ② ⚠️ 刻意**不做**「只给法名 -> 返回若干条文」。
        #
        #    初版实现过这条，但它是错的：图里没有「哪条更重要」的信息，
        #    只能退回 `ORDER BY a.num LIMIT k`，即**永远返回该法的第一条、第二条…**
        #    实测问「《证券法》关于内幕交易的规定」时返回第一条/第二条/第三条 ——
        #    对用户毫无意义，还会污染融合结果。
        #
        #    「在某部法律内找语义相关条文」应由**文档限定路**（doc_scoped）承担，
        #    那才是正确的工具。图谱路只做它能做好的事：**精确定位**与**引用关系**。

        # ③ 只有条号 -> 全局按条号匹配（较宽，限量）
        if art_ints and not law_names:
            for num in art_ints[:2]:
                res = s.run(
                    "MATCH (a:Article {collection:$c}) WHERE a.num = $num "
                    "RETURN a, [] AS nb, [] AS refs LIMIT $k",
                    c=collection, num=num, k=k,
                )
                hits.extend(_rows_to_hits(res, "条号命中"))

        # ④ 沿 CITES 反向找：哪些条文引用了 query 提到的法
        if law_names:
            res = s.run(
                "MATCH (a:Article {collection:$c})-[:CITES]->(lr:LawRef) "
                "WHERE any(n IN $names WHERE lr.name CONTAINS n) "
                "RETURN a, [] AS nb, [lr.name] AS refs LIMIT $k",
                c=collection, names=law_names, k=k,
            )
            hits.extend(_rows_to_hits(res, "引用关系命中"))

    # 去重（按 uid），保留首次出现的来源说明
    seen, uniq = set(), []
    for h in hits:
        if h["uid"] in seen:
            continue
        seen.add(h["uid"])
        uniq.append(h)
        if len(uniq) >= k:
            break

    if uniq:
        log.info("图谱召回 %d 条（实体：法名%s 条号%s）",
                 len(uniq), law_names or "无", art_nums or "无")
    return uniq


def _rows_to_hits(rows, reason: str) -> list[dict]:
    out = []
    for r in rows:
        a = r["a"]
        out.append({
            "uid": a["uid"],
            "text": None,                # 文本由调用方从向量库补齐（图里只存关系）
            "source": a.get("law_name"),
            "law_name": a.get("law_name"),
            "article_no": a.get("article_no"),
            "score": None,
            "collection": a.get("collection"),
            "_graph_reason": reason,
            "_graph_neighbors": list(r.get("nb") or [])[:5],
            "_graph_refs": list(r.get("refs") or [])[:5],
        })
    return out


def stats(collection: str | None = None) -> dict:
    driver = get_driver()
    with driver.session() as s:
        where = " WHERE a.collection = $c" if collection else ""
        params = {"c": collection} if collection else {}
        n = s.run(f"MATCH (a:Article){where} RETURN count(a) AS n", **params).single()["n"]
        cites = s.run(f"MATCH (a:Article){where} MATCH (a)-[r:CITES]->() "
                      f"RETURN count(r) AS n", **params).single()["n"]
        adj = s.run(f"MATCH (a:Article){where} MATCH (a)-[r:ADJACENT]->() "
                    f"RETURN count(r) AS n", **params).single()["n"]
    return {"articles": n, "cites": cites, "adjacent": adj}
