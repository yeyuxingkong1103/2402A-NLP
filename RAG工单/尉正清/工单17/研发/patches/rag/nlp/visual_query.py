# 工单15 新增文件：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程
"""跨模态检索的查询理解与分数融合

## 要解决什么

IMDR 的「图文混合」问题习惯用**视觉引用**指代图纸，比如
「在文件中**第11页图3**中，编号13的部件相对于编号12的部件…」。
RAGFlow 原检索流程把整句话丢给 BM25 + 向量，于是出现两类故障：

- **故障1（检索无关）**：查询里的「部件」「编号」是泛词，会召回一堆
  *提到编号、但根本没有图* 的文本块（本工单实测：权利要求书那种
  「10．按权利要求1所述的装置…」段落得分很高），把真正的图纸块挤下去。
- **故障2（图文割裂）**：即便召回了图纸块，也没有任何机制把问题里的
  「图3」和「某一块讲的是图3」绑定起来 —— 模型只能靠猜。

## 怎么解决

两步，都只动检索打分，不改索引结构：

1. **解析视觉引用**：从问题里抽出图号（`图3`）和页码（`第11页`）。
2. **加权分数融合**：解析阶段已经给每个图纸块打过
   `【技术图纸】图N｜第P页` 的标记（见 `deepdoc/parser/figure_parser.py`
   的 `_mark_figures`）。检索时，凡是**带着被引用图号标记**的块，在融合分上
   加分。图号对上给 `figure_boost`，页码对上给 `page_boost`。

   这正是工单里写的「对结果进行融合（如加权分数…）」那种做法：视觉条件
   作为一路信号，以加权分数的形式并进最终排序分。

之所以用**标记匹配**而不是再跑一路向量检索：图号是个**确定性**信息
（问题说图3，就只有图3 那一块是对的），用字符串匹配既准又便宜，
不必再引入一路召回和它的不确定性权重。

## 外部配置

所有参数都能改，优先级：环境变量 > JSON 配置文件 > 内置默认值。

    JSON：`VISUAL_RAG_CONFIG`（默认 /ragflow/conf/visual_rag.json）
    环境变量：`VISUAL_RAG_<大写的配置名>`，例如 VISUAL_RAG_FIGURE_BOOST=0.8
"""
import json
import logging
import os
import re

DEFAULTS = {
    # 总开关
    "enabled": True,
    # 问题里提到图N，且块上带着「图N」标记时的加分
    "figure_boost": 0.5,
    # 问题里提到第P页，且块上标记的页码也是 P 时的**额外**加分
    "page_boost": 0.2,
    # 标记前缀，需与 figure_parser.py 的 _mark_figures 保持一致
    "marker": "【技术图纸】",
}

CONFIG_FILE = os.environ.get("VISUAL_RAG_CONFIG", "/ragflow/conf/visual_rag.json")

_config_cache = None


def load_config():
    """读配置。环境变量优先于 JSON 文件，JSON 文件优先于内置默认值。"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    cfg = dict(DEFAULTS)
    try:
        if CONFIG_FILE and os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, encoding="utf-8") as f:
                cfg.update(json.load(f))
            logging.info(f"[VisualRAG] 已加载配置 {CONFIG_FILE}: {cfg}")
    except Exception as e:
        logging.warning(f"[VisualRAG] 配置文件 {CONFIG_FILE} 读取失败，用默认值: {e}")

    for key in list(cfg.keys()):
        env = os.environ.get("VISUAL_RAG_" + key.upper())
        if env is None:
            continue
        try:
            if isinstance(cfg[key], bool):
                cfg[key] = env.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(cfg[key], int):
                cfg[key] = int(env)
            elif isinstance(cfg[key], float):
                cfg[key] = float(env)
            else:
                cfg[key] = env
        except ValueError:
            logging.warning(f"[VisualRAG] 环境变量 VISUAL_RAG_{key.upper()}={env!r} 解析失败，忽略")

    _config_cache = cfg
    return cfg


# 图号：图3、图 3、附图3、图三（中文数字只认到十）
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_FIG_RE = re.compile(r"(?:附)?图\s*([0-9]+|[一二三四五六七八九十])")
_PAGE_RE = re.compile(r"第\s*([0-9]+|[一二三四五六七八九十]+)\s*页")


def _to_int(s):
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + _CN_NUM.get(s[1], 0)
    if s.endswith("十"):
        return _CN_NUM.get(s[0], 0) * 10
    if "十" in s:
        a, b = s.split("十", 1)
        return _CN_NUM.get(a, 0) * 10 + _CN_NUM.get(b, 0)
    return _CN_NUM.get(s, 0)


def parse_visual_refs(question):
    """抽出问题里的视觉引用。返回 {"figures": [3], "pages": [11]}（可能为空）。"""
    if not question:
        return {"figures": [], "pages": []}
    figures, pages = [], []
    for m in _FIG_RE.finditer(question):
        n = _to_int(m.group(1))
        if n and n not in figures:
            figures.append(n)
    for m in _PAGE_RE.finditer(question):
        n = _to_int(m.group(1))
        if n and n not in pages:
            pages.append(n)
    return {"figures": figures, "pages": pages}


def marker_of(fig):
    return f"图{fig}"


def _marker_head(content, marker):
    """取出标记那一行，例如 `【技术图纸】图3/图2｜第10页｜第11页`。

    只在这一行里匹配，避免正文里偶然出现的「图3」误伤。
    """
    i = content.find(marker)
    if i < 0:
        return ""
    return content[i:i + 80].split("\n", 1)[0]


def visual_score(content, refs, cfg):
    """一个块因为「和问题引用的图纸对得上」该加多少分。

    页码是**闸门**：问题里写了「第11页」时，标记里不含第 11 页的块一律不加分。
    没有这道闸门会误伤 —— 实测有个「图例说明」文本块，正文里列了
    「图 1：…；图 2：…；图 3：…」的清单，靠图号也能匹配上，被加分后挤进
    第 2 名，反而把真正答对题目的正文块压了下去。
    """
    if not content or not refs:
        return 0.0, ""
    head = _marker_head(content, cfg.get("marker") or DEFAULTS["marker"])
    if not head:
        return 0.0, ""

    head_pages = {int(x) for x in re.findall(r"第\s*(\d+)\s*页", head)}
    head_figs = {x for x in re.findall(r"图\s*(\d+)(?![0-9])", head)}

    q_pages = set(refs.get("pages", []))
    q_figs = set(str(f) for f in refs.get("figures", []))

    if q_pages and not (q_pages & head_pages):
        return 0.0, ""

    score, why = 0.0, []
    hit_figs = q_figs & head_figs
    if hit_figs:
        score += float(cfg.get("figure_boost", 0.5))
        why.append("图" + "/图".join(sorted(hit_figs)))
    if q_pages & head_pages:
        score += float(cfg.get("page_boost", 0.2))
        why.append("第" + "/第".join(str(p) for p in sorted(q_pages & head_pages)) + "页")
    if not score:
        return 0.0, ""
    return score, "+".join(why)


def boost_scores(sim, sres, question, cfg=None):
    """把视觉条件下的加分并进融合分。

    sim    —— 已算好的融合分（list，顺序与 sres.ids 一致）
    返回 (新的 sim, 被加分的块数量)
    """
    cfg = cfg or load_config()
    if not cfg.get("enabled", True):
        return list(sim), 0

    refs = parse_visual_refs(question)
    if not refs["figures"] and not refs["pages"]:
        return list(sim), 0

    out = list(sim)
    hit = 0
    for i, cid in enumerate(sres.ids):
        content = sres.field.get(cid, {}).get("content_with_weight") or ""
        add, why = visual_score(content, refs, cfg)
        if add:
            out[i] += add
            hit += 1
            logging.info(f"[VisualRAG] 块 {str(cid)[:12]} 命中视觉引用（{why}），融合分 +{add}")
    if hit:
        logging.info(f"[VisualRAG] 问题 {question[:40]!r} 视觉引用 {refs}，共 {hit} 个块加分")
    return out, hit
