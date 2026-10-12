# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""文档分类标签体系

工单任务二第二部分要求「文档分类标签体系」，任务三要求工作流
「根据分类标签路由到不同的解析器（如扫描型路由至 OCR 解析器）」。

所以标签不只是给人看的分类，**它带一个 `route` 字段，就是路由依据**：

| 标签 | route | 下游动作 |
|---|---|---|
| `Scan_PDF` | `ocr` | 走 OCR 解析 |
| `Text_PDF` | `text` | 直接抽文字 |
| `Mixed_PDF` | `hybrid` | 逐页决定 |
| `Empty_or_Broken` | `review` | 人工看（多半是坏文件） |
| `Long_Document` | `chunk_aggressive` | 切块要更碎，否则超上下文 |
| `Plain_Text` / `Tabular` | `text` / `table` | 对应解析器 |
| `Unclassified` | `review` | 兜底 |

## 规则顺序有讲究

配置里的 `tags.rules` 是**有序**的：先判最确定的（扫描型、坏文件），
后面的兜底。一个文档命中多条就都打上（比如「扫描型 + 超长」），
上限由 `max_tags_per_doc` 控制。

**第一条命中的标签决定主路由**（`route`），工作流按主路由挑解析器，
其余标签当参数用（比如 Long_Document 提示切块要更碎）。

## 「过短」不能打在扫描件上

`Empty_or_Broken`（字符数 < 50）的初衷是抓「解析失败或占位文件」。
但扫描件本来就没有文字层，字符数当然是 0 —— 实测 IMDR 那 1700 份扫描专利
**每一份都会被判成 Empty_or_Broken**，于是每份文档同时挂着 `Scan_PDF→ocr`
和 `Empty_or_Broken→review` 两个路由，工作流看到的路由是「ocr、review」，
不知道该送哪儿。

所以这条规则加了 `pdf_kind_not: scanned`：**已经被判为扫描型的，字符数少是
预期内的，不叫坏**。真正要抓的是「有文字层却几乎没抽出字」的文件。
"""


def _match(rule_when, facts):
    """一条规则的 when 条件是否满足。空 when = 兜底，永远命中。"""
    if not rule_when:
        return True
    for k, v in rule_when.items():
        if k == "pdf_kind":
            if facts.get("pdf_kind") != v:
                return False
        elif k == "pdf_kind_not":
            if facts.get("pdf_kind") == v:
                return False
        elif k == "ext_in":
            if facts.get("ext") not in v:
                return False
        elif k == "chars_lt":
            if not (facts.get("chars") is not None and facts["chars"] < v):
                return False
        elif k == "chars_gte":
            if not (facts.get("chars") is not None and facts["chars"] >= v):
                return False
        else:
            return False
    return True


def classify(facts, cfg):
    """给一个文档打标签。

    facts 需要这些键：ext, pdf_kind, chars
    返回 [{name, route, reason}, ...]

    ## 空 when 的规则是「兜底」，不是「永远命中」

    配置里的 `Unclassified` 规则 `when: {}` 一旦被当成普通规则参与匹配，
    **每份文档都会挂上它** —— 实测 30 份文档打出了 90 个标签，其中 30 个是
    Unclassified，标签分布直接失去意义。所以这里把空 when 的规则摘出来，
    只在**其他规则一条都没命中**时才用。
    """
    t = cfg["tags"]
    out, fallback = [], None
    for rule in t["rules"]:
        when = rule.get("when") or {}
        if not when:
            fallback = fallback or rule
            continue
        if _match(when, facts):
            out.append({"name": rule["name"], "route": rule.get("route", "review"),
                        "reason": _why(when, facts)})
        if len(out) >= t["max_tags_per_doc"]:
            break
    if not out:
        r = fallback or {"name": "Unclassified", "route": "review"}
        out.append({"name": r["name"], "route": r.get("route", "review"),
                    "reason": "没有其他规则命中"})
    return out


def _why(when, facts):
    if not when:
        return "兜底规则"
    bits = []
    for k, v in when.items():
        if k == "pdf_kind":
            bits.append(f"PDF 类型={facts.get('pdf_kind')}")
        elif k == "pdf_kind_not":
            bits.append(f"PDF 类型≠{v}")
        elif k == "ext_in":
            bits.append(f"扩展名={facts.get('ext')}")
        elif k == "chars_lt":
            bits.append(f"字符数 {facts.get('chars')} < {v}")
        elif k == "chars_gte":
            bits.append(f"字符数 {facts.get('chars')} ≥ {v}")
    return "；".join(bits)


def summarize(tagged):
    """汇总标签分布与路由去向。

    两个路由口径要分清，混了会让工作流算错：

    - `by_route`：**所有标签**涉及的解析器（一份文档可能算好几次）。
      用来看「一共要动用哪些解析器」，**不能拿来定路由**。
    - `by_primary`：**主路由**（每份文档只算第一条标签那一次）。
      这个才是「N 份文档分别送去哪儿」的答案，工作流读它。
    """
    from collections import Counter
    names, routes, primary = Counter(), Counter(), Counter()
    for _, tags in tagged:
        for t in tags:
            names[t["name"]] += 1
            routes[t["route"]] += 1
        primary[tags[0]["route"]] += 1
    total = len(tagged) or 1
    return {
        "by_tag": dict(names),
        "tag_ratios": {k: round(v / total, 4) for k, v in names.items()},
        "by_route": dict(routes),
        "by_primary": dict(primary),
    }
