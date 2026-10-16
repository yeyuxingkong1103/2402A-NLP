# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""LLM 抽样抽取实体与关系（LightRAG 索引构建流程的第 ⑤ 步）。

复用（不另起一套）：
  * :mod:`lightrag.entity_schema` —— 类型封闭集 / 别名 / 约束 / 类型吸附
  * :mod:`lightrag.ollama_client`  —— Ollama 生成（qwen2.5:3b，keep_alive 可控）

硬纪律（来自「工单12 资源纪律」与 t2 任务书）：
  * 只跑小样本：块数 ≤ ``graph.sample_size``（默认 400），**绝不跑全量 3013 块**；
  * 单块 1 次 LLM 调用（``graph.batch_chunks = 1``）、**全程串行**，同一时刻只驻留 1 个模型；
  * 送 LLM 的正文截断到 ``graph.max_chunk_chars``（≤1000 字符）；
  * 单块失败重试 ≤ ``graph.max_retries_per_chunk``（默认 2）次，失败块写
    ``failed_chunks.jsonl`` 后继续，不中断整批；
  * 逐块 append 到 ``raw_extractions.jsonl``，支持 ``--resume`` 断点续跑；
  * **禁用** multiprocessing / ProcessPoolExecutor / joblib / Pool(（本模块只用单线程循环）。

抽样口径（与 ``设计/检索与评估方案.md`` §4 一致）：
  ① 只保留 ``graph.keep_doc_keys`` 中的文档（默认 招股说明书1无水 + 招股说明书2pd；
     剔除带水印重复件 招股说明书1pd）；
  ② 丢弃 ``content`` 长度 < ``graph.min_chunk_chars``（150）的块；
  ③ 优先队列 = 命中 16 题 ``expected_pages``（±``question_page_tolerance`` 页）的块，
     占比 ``question_page_quota_ratio``（默认 0.4）；
  ④ 其余按 ``doc_key`` 分层按块数占比配额，层内 ``table``/``image`` 优先（信息密度加权），
     并按等距跨取（spread）以覆盖整份文档；
  ⑤ 固定 ``sample_seed``（20250826）保证可复现。
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Cfg
from .entity_schema import (
    ENTITY_TYPES,
    RELATION_TYPES,
    PRONOUN_ALIASES,
    coerce_type,
    guess_entity_type,
    normalize_name,
    resolve_alias,
)
from .ollama_client import LLMError, OllamaClient

# 批处理上限（嵌入类通道用；本模块的生成通道固定 1 块/次）
MAX_BATCH = 16


def canonical_name(name: str) -> str:
    """实体名归并的**统一出口**：``normalize_name(resolve_alias(normalize_name(x)))``。

    为什么要在输出侧再规范化一次：``PRONOUN_ALIASES`` 的 canonical 值本身可能含空格
    （如 ``"赵马克(Mark Zhao)"``），而 :func:`normalize_name` 会去掉空白
    （→ ``"赵马克(MarkZhao)"``）。若不补这一步，输入「赵马克」映射为带空格形式、
    输入「赵马克(Mark Zhao)」映射为无空格形式，**同一主体会被拆成两个节点**。
    实测：schema 冒烟测试中 ``赵马克(Mark Zhao)`` 归一为 ``赵马克(MarkZhao)``。

    ⚠ 本函数是 graph 层的补齐，**未改动 entity_schema 的别名规则表**
    （t2 约束：该文件只允许改 docstring 的两处数字）。
    """
    return normalize_name(resolve_alias(normalize_name(name)))


# 易漏类型提示：来自 设计/实体关系Schema设计.md §1/§2 的语料实测
# （「schema 类型名」≠「语料用词」，不给样例则这几类必漏）
_EASY_MISS_HINTS = (
    "指标名（如「军用领域的收入」「注册资本」）→ 财务指标；"
    "「6,464.51万元」「1,670万股」→ 金额；"
    "「82.10%」「42.35%」→ 比例；"
    "「报告期内」「2014年12月」→ 日期；"
    "「全资子公司」「持股100%」→ 母子公司；"
    "「收入分别为…、…、…」→ 披露指标；"
    "「参与的某项工程」「荣获一等奖」→ 参与工程/获奖。"
)


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
def type_menu() -> str:
    """类型封闭集（紧凑写法，省 token：完整定义见 entity_schema.py）。"""
    ent = "|".join(ENTITY_TYPES)
    rel = "|".join(RELATION_TYPES)
    return f"实体类型：{ent}\n关系类型：{rel}\n易漏提示：{_EASY_MISS_HINTS}"


# 关系用法指引：试点实测「只给类型名、不给用法」时，模型把「本公司全体董监高」
# 判成「母子公司」（应为「任职」），16 条关系里 15 条被 relation_allowed 挡掉。
_RELATION_USAGE = (
    "【关系用法（按此选，别混用）】\n"
    "- 自然人在公司/机构任董监高 → 「任职」（**不要**用「母子公司」表示任职）\n"
    "- 公司持有另一公司股份/控股 → 「控股」或「持有股份」\n"
    "- 公司与其子公司/全资子公司 → 「母子公司」\n"
    "- 主体某项财务数据取某数值 → 「披露指标」（例：军用领域收入 → 披露指标 → 6,464.51万元）\n"
    "- 公司向客户卖产品 → 「销售」；向供应商买东西 → 「采购」\n"
    "- 参与制定标准 → 「制定标准」；参与工程 → 「参与工程」；获奖 → 「获奖」\n"
    "- 公司/产品属于某行业 → 「所属行业」；被列为关联方 → 「关联方」\n"
    "- 「投资」只用于设立/增资/股权转让这类**动作**，不要用来描述身份、任职或上市板块\n"
    "关系方向必须是「主体 → 客体」（自然人任职时自然人作 head、公司作 tail）。"
)

# 垃圾实体规则：试点实测污染源（地址、文档标题、中英文重复名、角色泛称）
_JUNK_RULE = (
    "【不要抽取】地址、电话、邮箱、页码、页眉页脚、文档标题（如「招股说明书」「申报稿」）、目录项。\n"
    "【不要把角色泛称当实体】只抽**具体专名**（公司全称、人名、机构名、产品名、行业名、"
    "金额、比例、日期、指标名、项目名、标准名）；**不要**抽「控股股东」「实际控制人」"
    "「董事」「监事」「高级管理人员」「管理层」「员工」「公司」这类角色词/泛称，"
    "也不要把整句或半句当实体。\n"
    "【同一主体】中英文名只保留中文正式名（如「武汉兴图新科电子股份有限公司」，"
    "不要另建「Wuhan Xingtu Xinke Electronics Co.,Ltd.」）。\n"
)


def build_prompt(chunk: dict[str, Any], *, max_chars: int = 1000,
                 max_entities: int = 10, max_relations: int = 8) -> str:
    """构造单块抽取提示词（正文截断到 max_chars ≤1000）。"""
    content = str(chunk.get("content") or "")
    if len(content) > max_chars:
        content = content[:max_chars] + "…（已截断）"
    page = chunk.get("page")
    ctype = chunk.get("type")
    return (
        "你是招股说明书信息抽取器。从给定文本块中抽取实体与关系，只输出一个 JSON 对象。\n\n"
        f"{type_menu()}\n\n"
        "要求：\n"
        "1. 只输出 JSON，不要解释、不要 Markdown 代码块。\n"
        "2. name 用文中原词（≤30 字）；type 必须来自上面的实体类型清单。\n"
        "3. 关系是四元组 head/rel/tail/evidence；rel 必须来自上面的关系类型清单；"
        "evidence 为支撑该关系的原文片段（≤40 字）。\n"
        f"4. 最多 {max_entities} 个实体、{max_relations} 个关系；原文没有的就不要编造；"
        "宁少勿长，整个 JSON 控制在 700 字以内（超出会被截断，导致整块作废）。\n"
        "5. 金额/比例/日期也抽成实体，并用「披露指标」关系连到其所属主体。\n\n"
        f"{_RELATION_USAGE}\n{_JUNK_RULE}\n"
        '输出格式：{"entities":[{"name":"","type":""}],'
        '"relations":[{"head":"","rel":"","tail":"","evidence":""}]}\n\n'
        f"文本块（{ctype}，第 {page} 页）：\n{content}"
    )


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _json_blob(text: str) -> str:
    """从 LLM 返回里抠出 JSON 主体（容错 Markdown 代码块与前后废话）。"""
    t = _FENCE.sub("", str(text or "")).strip()
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        return t[start:end + 1]
    return t


def _loads_lenient(blob: str) -> Any:
    """宽松 JSON 解析：先直解，再修尾逗号 / 中文引号后重试。"""
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", blob)
    fixed = fixed.replace("“", '"').replace("”", '"').replace("，", ",").replace("：", ":")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return json.loads(fixed)


def _salvage_objects(blob: str) -> dict[str, list[Any]]:
    """从**被截断**的 JSON 里抢救出已完整生成的对象。

    背景（实测）：qwen2.5:3b 输出常在中途被 max_tokens 截断（JSONDecodeError at char 2271~2953），
    整块记录报废。这里按段扫描，把每个**已闭合**的 ``{...}`` 当作完整对象收下，
    丢掉未闭合的尾巴 —— 只使用模型真实产出的内容，不做任何补全或编造。
    """
    out: dict[str, list[Any]] = {"entities": [], "relations": []}
    idx_e = blob.find('"entities"')
    idx_r = blob.find('"relations"')
    segments: list[tuple[str, str]] = []
    if idx_e >= 0:
        segments.append(("entities", blob[idx_e:idx_r if idx_r > idx_e else len(blob)]))
    if idx_r >= 0:
        segments.append(("relations", blob[idx_r:]))
    for key, seg in segments:
        depth = 0
        start = -1
        for i, ch in enumerate(seg):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        out[key].append(json.loads(seg[start:i + 1]))
                    except (json.JSONDecodeError, ValueError):
                        pass
                    start = -1
    return out


def _pick(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        val = item.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def is_junk(name: str, patterns: Sequence[str]) -> bool:
    """垃圾实体判定：命中任一正则即视为非知识性实体（地址/标题/页码/角色泛称…）。"""
    return any(re.search(p, name) for p in patterns if p)


# 内部别名（兼容本模块既有调用点）；graph_store 在归并阶段复用同一判定
_is_junk = is_junk


def parse_extraction(text: str, *, drop_patterns: Sequence[str] = ()) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把 LLM 返回解析为 (entities, relations)，并做类型吸附与合法性标记。

    返回的实体字段：``name``（规范化后）/ ``type``（已吸附到封闭集）/ ``type_exact``
    （原始类型是否本就合法）。关系字段：``head`` / ``rel`` / ``tail`` / ``evidence``
    / ``rel_exact``。非法项不丢弃，交 :mod:`lightrag.graph_store` 统一校验与统计；
    但 ``drop_patterns`` 命中的垃圾实体（地址/文档标题/页码）与其相关关系会被剔除，
    剔除数量记在返回记录里供 ``graph_meta.json`` 统计。
    """
    blob = _json_blob(text)
    if not blob:
        raise ValueError("LLM 返回为空，无法解析")
    salvaged = False
    try:
        data = _loads_lenient(blob)
    except (json.JSONDecodeError, ValueError):
        data = _salvage_objects(blob)               # 截断抢救：只收已闭合对象
        salvaged = True
        if not (data.get("entities") or data.get("relations")):
            raise ValueError("JSON 解析失败且无可抢救对象（疑似截断）") from None
    if isinstance(data, list):                      # 少数情况直接返回数组
        data = {"entities": data, "relations": []}
    if not isinstance(data, dict):
        raise ValueError(f"LLM 返回不是 JSON 对象：{type(data).__name__}")

    ents: list[dict[str, Any]] = []
    seen: set[str] = set()
    junk: set[str] = set()
    for item in (data.get("entities") or []):
        if isinstance(item, str):
            name, raw_type = item, ""
        elif isinstance(item, dict):
            name = _pick(item, "name", "entity", "实体", "名称")
            raw_type = _pick(item, "type", "类型", "entity_type")
        else:
            continue
        name = normalize_name(name)
        if not name or len(name) > 40:
            continue
        canon = canonical_name(name)
        if _is_junk(canon, drop_patterns):
            junk.add(canon)
            continue
        if canon in seen:
            continue
        seen.add(canon)
        exact = raw_type in ENTITY_TYPES
        ents.append({
            "name": canon,
            "raw_name": name,
            "type": coerce_type(raw_type, kind="entity") if raw_type else guess_entity_type(name),
            "type_raw": raw_type,
            "type_exact": exact,
        })

    rels: list[dict[str, Any]] = []
    junk_rel = 0
    for item in (data.get("relations") or []):
        if not isinstance(item, dict):
            continue
        head = canonical_name(_pick(item, "head", "source", "from", "头实体", "主体"))
        tail = canonical_name(_pick(item, "tail", "target", "to", "尾实体", "客体"))
        raw_rel = _pick(item, "rel", "relation", "关系", "type", "predicate")
        head = normalize_name(head)
        tail = normalize_name(tail)
        if not head or not tail or head == tail:
            continue
        if len(head) > 40 or len(tail) > 40:        # 与实体名长度上限保持一致，避免"句子当实体"
            continue
        if _is_junk(head, drop_patterns) or _is_junk(tail, drop_patterns):
            junk_rel += 1
            continue
        exact = raw_rel in RELATION_TYPES
        rels.append({
            "head": head,
            "rel": coerce_type(raw_rel, kind="relation") if raw_rel else "披露指标",
            "rel_raw": raw_rel,
            "rel_exact": exact,
            "tail": tail,
            "evidence": _pick(item, "evidence", "依据", "原文")[:80],
        })
    return ents, rels, {"junk_entities": len(junk), "junk_relations": junk_rel, "salvaged": salvaged}


# ---------------------------------------------------------------------------
# 抽样
# ---------------------------------------------------------------------------
def load_question_pages(question_file: str | Path, *, tolerance: int = 1) -> dict[str, set[int]]:
    """读 16 题问题集 → {doc 前缀: 展开 ±tolerance 的页码集合}。

    问题集的 ``doc`` 字段为「招股说明书1 / 招股说明书2」，对应块表 ``doc_key``
    前缀（招股说明书1无水 / 招股说明书1pd / 招股说明书2pd），故按前缀匹配。
    """
    out: dict[str, set[int]] = {}
    path = Path(question_file)
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = str(row.get("doc") or "").strip()
            if not doc:
                continue
            pages: set[int] = set()
            for p in (row.get("expected_pages") or []):
                try:
                    base = int(p)
                except (TypeError, ValueError):
                    continue
                for d in range(-tolerance, tolerance + 1):
                    if base + d > 0:
                        pages.add(base + d)
            out.setdefault(doc, set()).update(pages)
    return out


def _page_hit(chunk: dict[str, Any], question_pages: dict[str, set[int]]) -> bool:
    doc_key = str(chunk.get("doc_key") or "")
    try:
        page = int(chunk.get("page") or 0)
    except (TypeError, ValueError):
        return False
    for prefix, pages in question_pages.items():
        if doc_key.startswith(prefix) and page in pages:
            return True
    return False


def _type_rank(chunk: dict[str, Any], bonus_types: set[str]) -> int:
    return 0 if str(chunk.get("type")) in bonus_types else 1


def _spread(items: Sequence[dict[str, Any]], need: int) -> list[dict[str, Any]]:
    """等距跨取 need 个元素（覆盖整份文档，而非只取前几页）。"""
    if need <= 0 or not items:
        return []
    if need >= len(items):
        return list(items)
    if need == 1:
        return [items[0]]
    step = (len(items) - 1) / (need - 1)
    picked: list[dict[str, Any]] = []
    for i in range(need):
        picked.append(items[min(int(round(i * step)), len(items) - 1)])
    return picked


def _allocate(total: int, sizes: Sequence[int]) -> list[int]:
    """按最大余数法把 total 个名额按各层块数占比分配。"""
    if total <= 0 or not sizes:
        return [0] * len(sizes)
    sum_sizes = sum(sizes)
    if sum_sizes <= 0:
        return [0] * len(sizes)
    raw = [total * s / sum_sizes for s in sizes]
    base = [int(x) for x in raw]
    remain = total - sum(base)
    order = sorted(range(len(sizes)), key=lambda i: (raw[i] - base[i], sizes[i]), reverse=True)
    for i in order[:max(remain, 0)]:
        base[i] += 1
    return base


def select_sample(chunks: Sequence[dict[str, Any]], cfg: Cfg, *,
                  question_pages: dict[str, set[int]] | None = None,
                  limit: int | None = None,
                  logger: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按抽样口径挑块，返回 (选中块列表, 抽样说明 meta)。"""
    g = cfg.graph
    keep = [str(x) for x in (g.get("keep_doc_keys") or [])]
    exclude = {str(x) for x in (g.get("exclude_doc_keys") or [])}
    min_chars = int(g.min_chunk_chars)
    bonus = {str(x) for x in (g.get("density_bonus_types") or [])}
    seed = int(g.sample_seed)
    ratio = float(g.get("question_page_quota_ratio") or 0.0)
    cap = int(limit or g.sample_size)
    q_pages = question_pages or {}

    pool: list[dict[str, Any]] = []
    dropped: dict[str, int] = {"文档不在保留清单": 0, "文档在剔除清单": 0, "块过短": 0}
    for c in chunks:
        doc_key = str(c.get("doc_key") or "")
        if keep and doc_key not in keep:
            dropped["文档不在保留清单"] += 1
            continue
        if doc_key in exclude:
            dropped["文档在剔除清单"] += 1
            continue
        if len(str(c.get("content") or "")) < min_chars:
            dropped["块过短"] += 1
            continue
        pool.append(c)

    if not pool:
        return [], {"selected": 0, "pool": 0, "reason": "过滤后无可抽块", "dropped": dropped}

    total = min(cap, len(pool))
    rng = random.Random(seed)

    # ③ 优先队列：命中 16 题页
    hit = [c for c in pool if _page_hit(c, q_pages)]
    hit_sorted = sorted(hit, key=lambda c: (str(c.get("doc_key")), int(c.get("page") or 0), int(c.get("order") or 0)))
    hit_sorted.sort(key=lambda c: _type_rank(c, bonus))
    q_quota = min(len(hit_sorted), int(total * ratio))
    chosen: list[dict[str, Any]] = _spread(hit_sorted, q_quota)

    # ④ 其余按 doc_key 分层配额
    chosen_ids = {str(c.get("chunk_id")) for c in chosen}
    rest = [c for c in pool if str(c.get("chunk_id")) not in chosen_ids]
    layers: dict[str, list[dict[str, Any]]] = {}
    for c in rest:
        layers.setdefault(str(c.get("doc_key")), []).append(c)
    rest_total = total - len(chosen)
    keys = sorted(layers)
    quotas = _allocate(rest_total, [len(layers[k]) for k in keys])
    layer_picked: dict[str, int] = {}
    for key, quota in zip(keys, quotas):
        items = sorted(layers[key], key=lambda c: (int(c.get("page") or 0), int(c.get("order") or 0)))
        items.sort(key=lambda c: _type_rank(c, bonus))
        picked = _spread(items, quota)
        rng.shuffle(picked)                     # 打散层内顺序，避免同页连抽
        picked.sort(key=lambda c: (int(c.get("page") or 0), int(c.get("order") or 0)))
        chosen.extend(picked)
        layer_picked[key] = len(picked)

    chosen.sort(key=lambda c: (str(c.get("doc_key")), int(c.get("page") or 0), int(c.get("order") or 0)))
    meta = {
        "selected": len(chosen),
        "pool": len(pool),
        "chunk_total": len(chunks),
        "sample_size_cap": cap,
        "sample_seed": seed,
        "min_chunk_chars": min_chars,
        "density_bonus_types": sorted(bonus),
        "batch_chunks": int(g.batch_chunks),
        "max_chunk_chars": int(g.get("max_chunk_chars") or 1000),
        "keep_doc_keys": keep,
        "exclude_doc_keys": sorted(exclude),
        "exclude_reason": str(g.get("exclude_reason") or ""),
        "question_page_tolerance": int(g.get("question_page_tolerance") or 1),
        "question_page_quota_ratio": ratio,
        "question_page_hits_available": len(hit_sorted),
        "question_page_picked": len(chosen) - sum(layer_picked.values()),
        "layer_counts": layer_picked,
        "dropped": dropped,
    }
    if logger is not None:
        logger.info("抽样完成：全量 %d 块 → 池 %d 块 → 选中 %d 块（题页优先 %d）",
                    len(chunks), len(pool), len(chosen), meta["question_page_picked"])
    return chosen, meta


# ---------------------------------------------------------------------------
# 抽取
# ---------------------------------------------------------------------------
def extract_one(client: OllamaClient, chunk: dict[str, Any], cfg: Cfg, *,
                logger: Any = None) -> dict[str, Any]:
    """对单块调一次 LLM 并解析；异常由调用方兜底（本函数向上抛）。"""
    g = cfg.graph
    max_chars = int(g.get("max_chunk_chars") or 1000)
    max_e = int(g.get("max_entities_per_chunk") or 10)
    max_r = int(g.get("max_relations_per_chunk") or 8)
    drop = [str(p) for p in (g.get("drop_entity_patterns") or [])]
    prompt = build_prompt(chunk, max_chars=max_chars, max_entities=max_e, max_relations=max_r)
    t0 = time.perf_counter()
    text, stats = client.generate_with_stats(prompt, kind="extract")
    seconds = time.perf_counter() - t0
    ents, rels, pstats = parse_extraction(text, drop_patterns=drop)
    return {
        "chunk_id": chunk.get("chunk_id"),
        "doc_key": chunk.get("doc_key"),
        "page": chunk.get("page"),
        "type": chunk.get("type"),
        "ok": True,
        "seconds": round(seconds, 3),
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "entities": ents,
        "relations": rels,
        "junk_entities": pstats["junk_entities"],
        "junk_relations": pstats["junk_relations"],
        "salvaged": bool(pstats.get("salvaged")),
        "raw": text[:2000],
    }


def read_done_ids(raw_path: Path | str) -> set[str]:
    """读已完成的块 id（用于 --resume）。"""
    done: set[str] = set()
    path = Path(raw_path)
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = row.get("chunk_id")
            if cid:
                done.add(str(cid))
    return done


def run(chunks: Sequence[dict[str, Any]], client: OllamaClient, cfg: Cfg, *,
        raw_path: Path | str, failed_path: Path | str, resume: bool = True,
        time_budget_s: float | None = None, max_new: int | None = None,
        logger: Any = None) -> dict[str, Any]:
    """逐块串行抽取，边跑边 append；支持断点续跑、时间预算与总记录数上限。"""
    g = cfg.graph
    max_retry = int(g.get("max_retries_per_chunk") or 2)
    raw_path = Path(raw_path)
    failed_path = Path(failed_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    done = read_done_ids(raw_path) if resume else set()
    todo = [c for c in chunks if str(c.get("chunk_id")) not in done]

    stats: dict[str, Any] = {
        "chunks_selected": len(chunks),
        "skipped_by_resume": len(chunks) - len(todo),
        "attempted": 0, "succeeded": 0, "failed": 0,
        "produced_entities": 0, "produced_relations": 0,
        "junk_entities": 0, "junk_relations": 0, "salvaged": 0,
        "seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
        "stopped_by_budget": False, "stopped_by_target": False, "errors": [],
    }
    t_start = time.perf_counter()
    mode = "a" if resume else "w"

    with raw_path.open(mode, encoding="utf-8") as raw_fh, failed_path.open(mode, encoding="utf-8") as fail_fh:
        for i, chunk in enumerate(todo, start=1):
            if max_new is not None and (i - 1) >= int(max_new):
                stats["stopped_by_target"] = True
                if logger is not None:
                    logger.info("【进度】达到目标总量上限，本次再抽 %d 块后退出（--resume 可续跑）", i - 1)
                break
            if time_budget_s is not None and (time.perf_counter() - t_start) > float(time_budget_s):
                stats["stopped_by_budget"] = True
                if logger is not None:
                    logger.warning("达到时间预算 %.0fs，已完成 %d/%d 块，先落盘退出（--resume 可续跑）",
                                   float(time_budget_s), i - 1, len(todo))
                break

            stats["attempted"] += 1
            last_err = ""
            for attempt in range(max_retry + 1):
                try:
                    record = extract_one(client, chunk, cfg, logger=logger)
                    raw_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    raw_fh.flush()
                    stats["succeeded"] += 1
                    stats["produced_entities"] += len(record["entities"])
                    stats["produced_relations"] += len(record["relations"])
                    stats["seconds"] += float(record["seconds"])
                    stats["prompt_tokens"] += int(record["prompt_tokens"])
                    stats["completion_tokens"] += int(record["completion_tokens"])
                    stats["junk_entities"] += int(record.get("junk_entities") or 0)
                    stats["junk_relations"] += int(record.get("junk_relations") or 0)
                    if record.get("salvaged"):
                        stats["salvaged"] += 1
                    if logger is not None:
                        if i % 50 == 0 or i == len(todo):
                            logger.info("【进度】已完成 %d/%d 块｜实体=%d 关系=%d｜累计 %.1fs",
                                        i, len(todo), stats["produced_entities"],
                                        stats["produced_relations"], time.perf_counter() - t_start)
                        elif i % 10 == 0:
                            logger.info("进度 %d/%d  已抽实体=%d 关系=%d 累计耗时=%.1fs",
                                        i, len(todo), stats["produced_entities"],
                                        stats["produced_relations"], time.perf_counter() - t_start)
                    break
                except (LLMError, ValueError, KeyError, TypeError) as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    if attempt < max_retry:
                        time.sleep(float(cfg.llm.retry_sleep_s) * (attempt + 1))
                        continue
                    stats["failed"] += 1
                    stats["errors"].append({"chunk_id": chunk.get("chunk_id"), "error": last_err})
                    fail_fh.write(json.dumps({
                        "chunk_id": chunk.get("chunk_id"), "doc_key": chunk.get("doc_key"),
                        "page": chunk.get("page"), "type": chunk.get("type"),
                        "attempts": attempt + 1, "error": last_err,
                        "content_head": str(chunk.get("content") or "")[:120],
                    }, ensure_ascii=False) + "\n")
                    fail_fh.flush()
                    if logger is not None:
                        logger.warning("块 %s 抽取失败（%d 次尝试）：%s",
                                       chunk.get("chunk_id"), attempt + 1, last_err)
                    break

    stats["wall_seconds"] = round(time.perf_counter() - t_start, 3)
    stats["call_seconds"] = round(float(stats["seconds"]), 3)
    stats["errors"] = stats["errors"][:20]
    stats["remaining"] = max(len(todo) - stats["attempted"], 0)
    return stats


def unload_models(cfg: Cfg, *, logger: Any = None) -> list[str]:
    """显式卸载 Ollama 模型（keep_alive=0），确保跑完不残留驻留模型。"""
    released: list[str] = []
    # 含嵌入模型：检索阶段会加载 bge-m3，收尾必须一并卸载，否则会留下 0.66 GB 常驻
    models = {str(cfg.llm.extract_model), str(cfg.llm.gen_model), str(cfg.llm.embed_model)}
    for model in models:
        try:
            payload = json.dumps({"model": model, "keep_alive": 0}, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                f"{str(cfg.llm.base_url).rstrip('/')}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=25).read()
            released.append(model)
        except Exception as exc:  # noqa: BLE001 - 卸载失败不应阻断收尾
            if logger is not None:
                logger.warning("卸载 %s 失败：%s: %s", model, type(exc).__name__, exc)
    if logger is not None and released:
        logger.info("已卸载 Ollama 模型：%s", ", ".join(released))
    return released
